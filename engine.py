# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, frame_skipping: bool = False):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    if frame_skipping: 
        # For frame skipping
        processed_frames = 0
        total_frames = 0
        # loss_epoch = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # outputs = model(samples)
        # For frame skipping
        if frame_skipping:
            frame_ids = torch.tensor([t['frame_id'] for t in targets]).to(device)
            outputs = model(samples, frame_ids, train=True, targets=targets)
            processed_frames += torch.sum(outputs['frame_mask']).item()
            total_frames += len(outputs['frame_mask'])
        else:
            outputs = model(samples)
        loss_dict = criterion(outputs, targets)
       
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # loss_epoch += losses
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # print('Grad: {}', torch.sum(model.mask_frame.pixel_threshold.grad))
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # loss_epoch /= len(data_loader)
    # print("Loss epoch: {}".format(loss_epoch))
    # optimizer.zero_grad()
    # loss_epoch.backward()
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", {k: meter.global_avg for k, meter in metric_logger.meters.items()})
    if frame_skipping: 
        # For frame skipping: print processed frame count
        print("Processed frames: {}/{}".format(processed_frames, total_frames))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, frame_skipping=False, period=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('region_sparsity', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    if frame_skipping: 
        # For frame skipping
        processed_frames = 0
        total_frames = 0
        object_to_image_gt = {}
        object_to_image_dt = {}
        region_skipped = 0
    for samples, targets in metric_logger.log_every(data_loader, 50, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

         # For frame skipping
        if frame_skipping:
            # mapping track ids to the first image that it appears in
            image_to_objects = {t['image_id']: t['track_ids'] for t in targets}
            for image_id, object_ids in image_to_objects.items():
                for object_id in object_ids:
                    if object_id.item() not in object_to_image_gt:
                        object_to_image_gt[object_id.item()] = image_id.item()

            frame_ids = torch.tensor([t['frame_id'] for t in targets]).to(device) 
            # outputs = model(samples, frame_ids, period)
            outputs = model(samples, frame_ids, train=False, targets=targets)
            processed_frames += torch.sum(outputs['frame_mask']).item()
            total_frames += len(outputs['frame_mask'])
            # skipped region
            region_mask = outputs['region_mask']
            region_sparsity = 1 - torch.sum(region_mask)/torch.prod(torch.tensor(region_mask.shape))
            loss_dict, object_to_image_dt = criterion(outputs, targets, object_to_image_dt)
        else:
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(region_sparsity=region_sparsity)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        # if frame_skipping:
        #     targets = [targets[i] for i in torch.nonzero(outputs['frame_mask']).flatten()]
        #     results = [results[i] for i in torch.nonzero(outputs['frame_mask']).flatten()]
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", {k: meter.global_avg for k, meter in metric_logger.meters.items()})
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    if frame_skipping: 
        # Region SKipping
        # print("Average portion of frames skipped: {}".format(region_skipped / len(data_loader)))
        # For frame skipping: print processed frame count
        print("Processed frames: {}/{}".format(processed_frames, total_frames))
        print('Ground truth:', object_to_image_gt)
        print('Detection:', object_to_image_dt)
        delay, missed_tracks = 0, 0
        for track_id in object_to_image_gt.keys():
            if track_id in object_to_image_dt:
                assert object_to_image_gt[track_id] <= object_to_image_dt[track_id], "track appears later in ground truth than in detection"
                delay += object_to_image_dt[track_id] - object_to_image_gt[track_id]
            else:
                missed_tracks += 1
        print("Average delay: {}".format(delay / (len(object_to_image_gt))))
        print("Missed tracks: {}".format(missed_tracks))
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
