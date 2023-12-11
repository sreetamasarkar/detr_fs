# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
# from engine import evaluate, train_one_epoch
from region_mask_generator import mae_vit_base_patch16, CNN

import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Iterable
from util import box_ops
import torch.nn.functional as F
from util.dice_score import dice_loss

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int) # For frame skipping, batch size also acts as the upper bound on number of skipped frames at a stretch
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--num_classes', type=int, default=None,
                        help="Number of classes in dataset+1")
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")


    # dataset parameters
    parser.add_argument('--dataset_file', default='kitti_tracking', choices=['coco', 'kitti_tracking'])
    parser.add_argument('--coco_path', type=str, default='/data1/COCO')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='results_fs_debug',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--region_size', default=16, type=int)
    return parser

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, region_size: int = 16):
    model.train()
    # criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('recall', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('precision', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('sparsity', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50


    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        frame_ids = torch.tensor([t['frame_id'] for t in targets]).to(device) 
        # outputs = model(samples)
        # ------------------ Extract region mask from targets ------------------   
        img_h, img_w = samples.tensors.shape[2:]
        bbox_batch = [t['boxes'] for t in targets]
        region_mask = []
        for i in range(len(bbox_batch)):
            rmask = torch.zeros((samples.tensors.shape[2:]))
            b = box_ops.box_cxcywh_to_xyxy(bbox_batch[i])
            b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).cuda()
            for bb in b:
                x1, y1, x2, y2 = bb
                rmask[int(y1):int(y2), int(x1):int(x2)] = 1
            region_mask.append(rmask.unsqueeze(0).unsqueeze(0))
        region_mask = torch.cat(region_mask, dim=0).to(device) # same shape as input image with one channel
        # ----------------------------------------------------------------------
        weight = torch.ones(1, region_mask.shape[1], region_size, region_size, device=device)
        y = F.conv2d(region_mask, weight, stride=region_size)
        y[y>0] = 1.0
        if 1 in frame_ids:
            first_frame_id = torch.where(frame_ids==1)[0][0]
            if model.new_mask is None:
                model.init_masks(samples.tensors)
            model.update_new_mask(region_mask[first_frame_id].unsqueeze(0))
        # y = y/(y.max()+1e-10)
        # y = y.flatten(1)
        # model.update_obj_token(y[-1].unsqueeze(-1))
        outputs = model(samples.tensors, frame_ids)
        # outputs = outputs.flatten(1)
        loss = criterion(outputs, y)
        # loss += dice_loss(F.sigmoid(outputs), y)
        # loss += dice_loss(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('Grad: {}', torch.sum(model.mask_frame.pixel_threshold.grad))
        tp = torch.sum((outputs>0.5) * y)
        fp = torch.sum((outputs>0.5) * (1-y))
        fn = torch.sum((outputs<=0.5) * y)
        recall = tp/(tp+fn+1e-10)
        precision = tp/(tp+fp+1e-10)
        sparsity = 1 - torch.sum(y)/(y.shape[0]*y.shape[1]*y.shape[2]*y.shape[3])
        metric_logger.update(loss=loss)
        metric_logger.update(recall=recall)
        metric_logger.update(precision=precision)
        metric_logger.update(sparsity=sparsity)
        model.update_last_mask(region_mask[-1].unsqueeze(0))
       
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    print("Averaged stats:", stats)
    return stats


@torch.no_grad()
def evaluate(model, criterion, data_loader, base_ds, device, output_dir, region_size, thresold=0.5):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('recall', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('precision', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('sparsity', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'


    for samples, targets in metric_logger.log_every(data_loader, 50, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        frame_ids = torch.tensor([t['frame_id'] for t in targets]).to(device) 
        # ------------------ Extract region mask from targets ------------------   
        img_h, img_w = samples.tensors.shape[2:]
        bbox_batch = [t['boxes'] for t in targets]
        region_mask = []
        for i in range(len(bbox_batch)):
            rmask = torch.zeros((samples.tensors.shape[2:]))
            b = box_ops.box_cxcywh_to_xyxy(bbox_batch[i])
            b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).cuda()
            for bb in b:
                x1, y1, x2, y2 = bb
                rmask[int(y1):int(y2), int(x1):int(x2)] = 1
            region_mask.append(rmask.unsqueeze(0).unsqueeze(0))
        region_mask = torch.cat(region_mask, dim=0).to(device) # same shape as input image with one channel
        # ----------------------------------------------------------------------
        weight = torch.ones(1, region_mask.shape[1], region_size, region_size, device=device)
        y = F.conv2d(region_mask, weight, stride=region_size)
        y[y>0] = 1.0
        if 1 in frame_ids:
            first_frame_id = torch.where(frame_ids==1)[0][0]
            if model.new_mask is None:
                model.init_masks(samples.tensors)
            model.update_new_mask(region_mask[first_frame_id].unsqueeze(0))
        # y = y/y.max()
        # y = y.flatten(1)
        # model.update_obj_token(y[-1].unsqueeze(-1))
        outputs = model(samples.tensors, frame_ids)
        # outputs = outputs.flatten(1)
        loss = criterion(outputs, y)
        # loss += dice_loss(F.sigmoid(outputs), y)
        # loss += dice_loss(outputs, y)
        # y[y>0] = 1.0
        # tp = torch.sum((F.sigmoid(outputs)>0.5) * y)
        # fp = torch.sum((F.sigmoid(outputs)>0.5) * (1-y))
        # fn = torch.sum((F.sigmoid(outputs)<=0.5) * y)
        tp = torch.sum((outputs>thresold) * y)
        fp = torch.sum((outputs>thresold) * (1-y))
        fn = torch.sum((outputs<=thresold) * y)
        recall = tp/(tp+fn+1e-10)
        precision = tp/(tp+fp+1e-10)
        sparsity = 1 - torch.sum(y)/(y.shape[0]*y.shape[1]*y.shape[2]*y.shape[3])
        metric_logger.update(loss=loss)
        metric_logger.update(recall=recall)
        metric_logger.update(precision=precision)
        metric_logger.update(sparsity=sparsity)
        model.update_last_mask(region_mask[-1].unsqueeze(0))
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    print("Averaged stats:", stats)
  
    return stats
    

def main(args):
    torch.autograd.set_detect_anomaly(True)
    if args.eval:
        args.outfile = args.output_dir + '/eval.txt'
    else:
        args.outfile = args.output_dir + '/output.txt'
    f = open(args.outfile, 'w')
    sys.stdout = f
    
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))


    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # model = mae_vit_base_patch16(img_size=(384,1280), in_chans=3, region_size=16)
    model = CNN(region_size=args.region_size)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
      
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
    #                               weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:            
        sampler_train = torch.utils.data.SequentialSampler(dataset_train) # For frame skipping
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


    base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        #TODO: remove strict=False and include specific conditions (strict=False used for loading last layer with different num of classes)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        
    # criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.BCELoss()
    if args.eval:
        test_stats = evaluate(model, criterion, data_loader_val, base_ds, device, args.output_dir, region_size=args.region_size)
        # region_mask = model.maskregion.region_mask.squeeze(0).detach().cpu().numpy()
        # plt.imsave(args.output_dir + '/region_mask.png', region_mask, cmap=cm.gray)
        return
    best_loss = 10000
    loss_list, loss_bbox_list, loss_ce_list, loss_giou_list = [], [], [], []
    loss_frame_count_list = []
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, region_size=args.region_size)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(model, criterion, data_loader_val, base_ds, device, args.output_dir, args.region_size)
        loss_list.append(test_stats['loss'])

        best_checkpoint_path = output_dir / 'best_checkpoint.pth'
        if test_stats['loss'] < best_loss:
            best_loss = test_stats['loss']
            utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, best_checkpoint_path)
            
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Minimum loss {}'.format(best_loss))
    # pixel_threshold = model.mask_frame.pixel_threshold.squeeze(0).detach().cpu().numpy()
    # plt.imsave(args.output_dir + 'pixel_threshold.png', pixel_threshold/pixel_threshold.max(), cmap=cm.gray)
    # delta = model.mask_frame.delta.squeeze(0).detach().cpu().numpy()
    # plt.imsave(args.output_dir + 'delta.png', delta/delta.max(), cmap=cm.gray)
    if not args.eval:
        plt.plot(loss_list, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(args.output_dir + '/loss.png')
    print('Training time {}'.format(total_time_str))
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
