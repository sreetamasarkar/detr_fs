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
from engine import evaluate, train_one_epoch
from models import build_model

import matplotlib.pyplot as plt
from matplotlib import cm

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_threshold', default=1e-5, type=float)
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
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Frame skipping
    parser.add_argument('--frame_skipping', action='store_true',
                        help="Train while skipping frames if the flag is provided")  
    parser.add_argument('--period', default=None, type=int, help="periodic frame skipping period") # set batch size as a multiple of period
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--frame_count_loss_coef', default=1, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='kitti_tracking', choices=['coco', 'kitti_tracking'])
    parser.add_argument('--coco_path', type=str, default='/data1/COCO')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
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
    return parser


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

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if not args.frame_skipping:
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
    else:
        # Freeze weight parameters except for maskregion network
        for name, parameter in model.named_parameters():
            if 'maskregion' in name:
                print('Setting grad false for: {}'.format(name))
                parameter.requires_grad_(False)

        # Remove mask region network from parameters before loading checkpoint
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and "maskregion" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and "maskregion" not in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
      
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        if not args.frame_skipping:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
        else:
            sampler_train = torch.utils.data.SequentialSampler(dataset_train) # For frame skipping
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    # if args.frame_skipping:
        # Add decision threshold and pixel threshold to optimizer parameters after loading checkpoint
        # Run one step to get pixel threshold parameter
        # for samples, targets in data_loader_train:
        #     samples = samples.to(device)
        #     frame_ids = torch.tensor([t['frame_id'] for t in targets]).to(device)
        #     outputs = model(samples, frame_ids)
        #     break

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
            if checkpoint["model"]["class_embed.weight"].shape[0] != args.num_classes:
                print("=> WARNING: pretrained model has {} classes but "
                      "the config file specifies {} classes. "
                      "Proceeding anyway...".format(
                          checkpoint["model"]["class_embed.weight"].shape[0], args.num_classes))
                # Remove class weights
                del checkpoint["model"]["class_embed.weight"]
                del checkpoint["model"]["class_embed.bias"]
            # SaveOGH
            # torch.save(checkpoint,
                    # 'detr-r50_no-class-head.pth')
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        #TODO: remove strict=False and include specific conditions (strict=False used for loading last layer with different num of classes)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        # if args.frame_skipping:           
        #     # param_dicts_threshold = {
        #     #     "params": [p for n, p in model_without_ddp.named_parameters() if "threshold" in n or "scores" in n and p.requires_grad],
        #     #     "lr": args.lr_threshold,
        #     # }
        #     # Train mask region network together with detr
        #     param_dicts_threshold = {
        #         "params": [p for n, p in model_without_ddp.named_parameters() if "maskregion" in n and p.requires_grad],
        #         "lr": args.lr,
        #     }
        #     optimizer.add_param_group(param_dicts_threshold)

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, args.frame_skipping, args.period)
        # if args.output_dir:
        #     utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        # pixel_threshold = model.mask_frame.pixel_threshold.squeeze(0).detach().cpu().numpy()
        # plt.imsave(args.output_dir + 'pixel_threshold.png', pixel_threshold/pixel_threshold.max(), cmap=cm.gray)
        # delta = model.mask_frame.delta.squeeze(0).detach().cpu().numpy()
        # plt.imsave(args.output_dir + '/delta.png', delta/delta.max(), cmap=cm.gray)
        # masked_delta = model.mask_frame.delta_masked.squeeze(0).detach().cpu().numpy()
        # plt.imsave(args.output_dir + '/delta_masked.png', masked_delta/masked_delta.max(), cmap=cm.gray)
        region_mask = model.maskregion.region_mask.squeeze(0).detach().cpu().numpy()
        plt.imsave(args.output_dir + '/region_mask.png', region_mask, cmap=cm.gray)
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
            args.clip_max_norm, args.frame_skipping)
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

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args.frame_skipping
        )
        loss_list.append(test_stats['loss'])
        loss_bbox_list.append(test_stats['loss_bbox'])
        loss_ce_list.append(test_stats['loss_ce'])
        loss_giou_list.append(test_stats['loss_giou'])
        if args.frame_skipping:
            loss_frame_count_list.append(test_stats['loss_frame_count'])

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

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Minimum loss {}'.format(best_loss))
    # pixel_threshold = model.mask_frame.pixel_threshold.squeeze(0).detach().cpu().numpy()
    # plt.imsave(args.output_dir + 'pixel_threshold.png', pixel_threshold/pixel_threshold.max(), cmap=cm.gray)
    # delta = model.mask_frame.delta.squeeze(0).detach().cpu().numpy()
    # plt.imsave(args.output_dir + 'delta.png', delta/delta.max(), cmap=cm.gray)
    if not args.eval:
        plt.plot(loss_list, label='Loss')
        plt.plot(loss_bbox_list, label='Bbox Loss')
        plt.plot(loss_ce_list, label='CE Loss')
        plt.plot(loss_giou_list, label='GIOU Loss')
        if args.frame_skipping:
            plt.plot(loss_frame_count_list, label='Frame Count Loss')
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
