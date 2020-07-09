import argparse
import os
import os.path as osp

import pickle
from tqdm import tqdm
import numpy as np
import mmcv
import torch
from mmcv.parallel import DataContainer as DC
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmdet.core import encode_mask_results, tensor2imgs
from tools.fuse_conv_bn import fuse_module

from mmdet.apis import multi_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


def show_result(img,
                result,
                score_thr=0.3,
                bbox_color='green',
                text_color='green',
                thickness=1,
                font_scale=0.5,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None):
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    total_mask = torch.zeros(img.shape[:2], dtype=torch.bool)
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        np.random.seed(42)
        color_masks = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(max(labels) + 1)
        ]
        for i in inds:
            i = int(i)
            color_mask = color_masks[labels[i]]
            mask = segms[i]
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
            # if labels[i] != 0:
            #     continue
            # mask = segms[i]
            # total_mask = total_mask | torch.from_numpy(mask)
    # img[~total_mask] = 0
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
    # bboxes[:, 4] = 0
    # draw bounding boxes
    mmcv.imshow_det_bboxes(
        img,
        bboxes,
        labels,
        class_names=CLASSES,
        score_thr=score_thr,
        bbox_color=bbox_color,
        text_color=text_color,
        thickness=thickness,
        font_scale=font_scale,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file)

    if not (show or out_file):
        warnings.warn('show==False and out_file is not specified, only '
                      'result image will be returned')
        return img


def single_gpu_test(model,
                    img_data_list,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    for (img, img_meta) in img_data_list:
        data = {
            "img_metas": [DC([[img_meta]], cpu_only=True)],
            "img": [img.unsqueeze(0).contiguous()]
        }
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, osp.basename(img_meta['ori_filename']))
                else:
                    out_file = None

                show_result(
                    img_show,
                    result,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result, tuple):
            bbox_results, mask_results = result
            encoded_mask_results = encode_mask_results(mask_results)
            result = bbox_results, encoded_mask_results
        results.append(result)

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--input', help='input images to be processed', nargs='+')
    # parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    # parser.add_argument(
    #     '--format-only',
    #     action='store_true',
    #     help='Format the output results without perform evaluation. It is'
    #     'useful when you want to format the result to a specific format and '
    #     'submit it to the test server')
    # parser.add_argument(
    #     '--eval',
    #     type=str,
    #     nargs='+',
    #     help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
    #     ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    # parser.add_argument(
    #     '--gpu-collect',
    #     action='store_true',
    #     help='whether to use gpu to collect results.')
    # parser.add_argument(
    #     '--tmpdir',
    #     help='tmp directory used for collecting results from multiple '
    #     'workers, available when gpu-collect is not specified')
    # parser.add_argument(
    #     '--options', nargs='+', action=DictAction, help='arguments in dict')
    # parser.add_argument(
    #     '--launcher',
    #     choices=['none', 'pytorch', 'slurm', 'mpi'],
    #     default='none',
    #     help='job launcher')
    # parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def processing_one_image(file_path):
    img_meta = {}
    img_meta['filename'] = file_path
    img_meta['ori_filename'] = file_path
    img_meta['flip'] = False
    # 1. Read image
    file_client = mmcv.FileClient(backend='disk')
    img_bytes = file_client.get(file_path)
    orig_img = mmcv.imfrombytes(img_bytes, flag='color')  # BGR order
    img_meta['ori_shape'] = orig_img.shape
    # 2. Resize
    test_scale = (1333, 800)
    img, scale_factor = mmcv.imrescale(orig_img, test_scale, return_scale=True)
    # the w_scale and h_scale has minor difference
    # a real fix should be done in the mmcv.imrescale in the future
    new_h, new_w = img.shape[:2]
    h, w = orig_img.shape[:2]
    w_scale = new_w / w
    h_scale = new_h / h
    scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    img_meta['scale_factor'] = scale_factor
    img_meta['img_shape'] = img.shape
    # 3. Normalize
    # mean = np.array([102.9801, 115.9465, 122.7717], dtype=np.float32)
    # std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    mean = np.array([103.53 , 116.28 , 123.675], dtype=np.float32)
    std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    to_rgb = False
    img = mmcv.imnormalize(img, mean, std, to_rgb)
    img_meta['img_norm_cfg'] = dict(mean=mean, std=std, to_rgb=to_rgb)
    # 4. Pad
    img = mmcv.impad_to_multiple(img, divisor=32, pad_val=0)
    img_meta["pad_shape"] = img.shape
    # 5. ToTensor
    img = torch.from_numpy(img.transpose(2, 0, 1))
    return img, img_meta


def main():
    args = parse_args()

    # assert args.out or args.eval or args.format_only or args.show \
    #     or args.show_dir, \
    #     ('Please specify at least one operation (save/eval/format/show the '
    #      'results / save the results) with the argument "--out", "--eval"'
    #      ', "--format-only", "--show" or "--show-dir"')

    # if args.eval and args.format_only:
    #     raise ValueError('--eval and --format_only cannot be both specified')

    # if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
    #     raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # # init distributed env first, since logger depends on the dist info.
    # if args.launcher == 'none':
    #     distributed = False
    # else:
    #     distributed = True
    #     init_dist(args.launcher, **cfg.dist_params)

    # # build the dataloader
    # # TODO: support multiple images per gpu (only minor changes are needed)
    # dataset = build_dataset(cfg.data.test)
    # data_loader = build_dataloader(
    #     dataset,
    #     samples_per_gpu=1,
    #     workers_per_gpu=cfg.data.workers_per_gpu,
    #     dist=distributed,
    #     shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    imgs = []
    for file_path in args.input:
        imgs.append(processing_one_image(file_path))
    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, imgs, args.show, args.show_dir,
                              args.show_score_thr)
    # with open("im_names.pkl", "rb") as f:
    #     im_names = pickle.load(f)
    # for im_name in tqdm(im_names):
    #     file_path = osp.join("data/cuhk_sysu/Image/SSM", im_name)
    #     imgs = [processing_one_image(file_path)]
    #     outputs = single_gpu_test(model, imgs, args.show, args.show_dir,
    #                               args.show_score_thr)

    # if not distributed:
    #     model = MMDataParallel(model, device_ids=[0])
    #     outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
    #                               args.show_score_thr)
    # else:
    #     model = MMDistributedDataParallel(
    #         model.cuda(),
    #         device_ids=[torch.cuda.current_device()],
    #         broadcast_buffers=False)
    #     outputs = multi_gpu_test(model, data_loader, args.tmpdir,
    #                              args.gpu_collect)

    # rank, _ = get_dist_info()
    # if rank == 0:
    #     if args.out:
    #         print(f'\nwriting results to {args.out}')
    #         mmcv.dump(outputs, args.out)
    #     kwargs = {} if args.options is None else args.options
    #     if args.format_only:
    #         dataset.format_results(outputs, **kwargs)
    #     if args.eval:
    #         dataset.evaluate(outputs, args.eval, **kwargs)


if __name__ == '__main__':
    main()