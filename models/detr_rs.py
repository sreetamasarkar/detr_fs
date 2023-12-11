# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch import nn
import math
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer
from region_mask_generator import CNN
class ThresholdFunc(torch.autograd.Function):

	@staticmethod
	def forward(ctx, input):
		# ctx.save_for_backward(input)
		out = torch.zeros_like(input).cuda()
		out[input > 1.0] = 1.0
		return out

	@staticmethod
	def backward(ctx, grad_output):	
		# input, = ctx.saved_tensors
		grad_input = grad_output.clone()
		# grad = STDB.alpha * torch.exp(-1*last_spike)**STDB.beta
		return grad_input

class GetSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten(start_dim=1).sort()
        j = int((1 - k) * scores[0].numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten(start_dim=1)
        for i in range(len(idx)):
            flat_out[i,idx[i,:j]] = 0
            flat_out[i,idx[i,j:]] = 1
        # flat_out[idx[:,:j]] = 0
        # flat_out[idx[:,j:]] = 1
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None
            
class MaskRegion(nn.Module): 
    def __init__(self, sparsity=0.5, threshold=0.5):
        super(MaskRegion, self).__init__()
        self.sparsity = sparsity
        self.threshold = threshold
        # initialize the scores
        self.scores = None
        # self.scores = self.register_parameter('scores', None)
        self.last_batch = None
        self.region_size = 16
        self.region_mask_generator = CNN(region_size=16)
        self.region_mask_generator.load_state_dict(torch.load('results_fs_debug4/checkpoint.pth')['model'])

    def init_scores(self, input):
        b,c,h,w = input.tensors.size()
        # self.scores = nn.Parameter(input.tensors.new(input.tensors.size()[2:]).unsqueeze(0).normal_(0, 1)) # pixel level scores
        # ------------------------- initialize scores with kaiming uniform -------------------------------
        # self.scores = nn.Parameter(input.tensors.new(1,h//self.region_size,w//self.region_size).normal_(0, 1)) # region level scores
        # nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5)) # initialize scores with kaiming uniform
        # ------------------------------------initialize scores with heatmap-------------------------------
        with open('datasets/hm_train.npy', 'rb') as f:
            hm_train = np.load(f)
        self.init_value = torch.tensor(hm_train/hm_train.max(), device=input.tensors.device, dtype=torch.float)
        # init_value = torch.clip(init_value, min=0.01) # avoid zeros
        self.scores = self.init_value.clone().detach().unsqueeze(0) # pixel threshold with 3 channels
        if self.region_mask_generator.new_mask is None:
            self.region_mask_generator.init_masks(input.tensors)
        self.region_mask_generator.update_new_mask(self.scores) 
        # self.scores = nn.Parameter(self.init_value.unsqueeze(0)) # pixel threshold with 1 channel 
        # self.scores = nn.Parameter(self.init_value[::self.region_size,::self.region_size].unsqueeze(0)) # region threshold with 1 channel 
        # self.scores = nn.Parameter(self.init_value[::self.region_size,::self.region_size].unsqueeze(0), requires_grad=False) # region threshold with 1 channel 
        # --------------------------------------------------------------------------------------------------
        # # self.pixel_threshold = nn.Parameter(0.5*torch.ones_like(input.tensors[0])) # pixel threshold with 3 channels 
        # # self.pixel_threshold = nn.Parameter(0.8*torch.ones_like(input.tensors[0][0].unsqueeze(0))) # pixel threshold with 1 channel  
        # self.delta = torch.zeros_like(input.tensors[0][0].unsqueeze(0))
        # self.delta_masked = torch.zeros_like(input.tensors[0][0].unsqueeze(0))
        self.region_mask = torch.zeros_like(input.tensors[0][0].unsqueeze(0))

    def update_last_batch(self, input):
        self.last_batch = input.tensors
        
    def get_prev_input(self, cur_input, frame_ids):
        prev_input_batch = []
        if self.last_batch is None:
            prev_input = 0
        else:
            prev_input = self.last_batch[-1]
        for i in range(len(cur_input)):
            if frame_ids[i] == 1:
                prev_input = torch.zeros_like(cur_input[i])
            delta = cur_input[i] - prev_input
            pixel_mask = torch.abs(delta) > self.pixel_threshold 
            pixels_above_threshold = torch.sum(pixel_mask)/torch.prod(torch.tensor(pixel_mask.shape))
            frame_dec = pixels_above_threshold > self.decision_threshold
            prev_input_batch.append(prev_input.unsqueeze(0))
            if frame_dec == 1: # update prev_input only if the frame is processed
                prev_input = cur_input[i]
            
        prev_input_batch = torch.cat(prev_input_batch, dim=0)
        return prev_input_batch
    
    def update_scores(self, input, output, frame_ids, gamma=0.9, num_classes=4):
        img_h, img_w = input.tensors.shape[2:]
        b = box_ops.box_cxcywh_to_xyxy(output['pred_boxes'])
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).cuda()
        region_mask = []
        for i in range(len(b)):
            # if frame_ids[i] == 1:
            #     self.scores = 0.0 * self.scores
            keep = (output['pred_logits'][i].softmax(-1).topk(1)[1] != num_classes).squeeze(-1)
            # x = torchvision.ops.nms(b[i], output['pred_logits'][i].softmax(-1).max(-1).values, iou_threshold=0.1)
            rmask = torch.zeros(input.tensors.shape[2:])
            for bb in b[i][keep]:
                x1, y1, x2, y2 = bb
                rmask[int(y1):int(y2), int(x1):int(x2)] = 1
            region_mask.append(rmask.unsqueeze(0).unsqueeze(0))
        region_mask = torch.cat(region_mask, dim=0).to(input.tensors.device) # same shape as input image with one channel
        #     self.scores += (gamma ** (len(frame_ids)-i)) * rmask
        # self.scores += self.init_value
        # idx = self._get_src_permutation_idx(indices)
        # src_boxes = outputs['pred_boxes'][idx]
        self.region_mask_generator.update_last_mask(region_mask[-1].unsqueeze(0))

    def forward(self, input, frame_ids, targets):
        if self.scores is None:
            self.init_scores(input)
        #------------- Masking using delta -----------------
        # cur_input = input.tensors
        # # get previous input
        # if self.last_batch is None:
        #     self.last_batch = torch.zeros(input.tensors.shape).cuda()

        # prev_input = torch.cat((self.last_batch[-1].unsqueeze(0), cur_input[:-1]), dim=0)
        # prev_input[frame_ids == 1] = 0.0 # for first frame of each sequence, prev_input is set to 0
        # delta = cur_input - prev_input # prev_input is the last processed frame, cur_input is the frame being currently processed 
        # # Region level masking
        # weight = torch.ones(1, input.tensors.shape[1], self.region_size, self.region_size, device=input.tensors.device)
        # y = F.conv2d(delta, weight, stride=self.region_size) # convolve with a kernel of size region_size to obtain total delta in region
        # y_masked = y * self.scores
        # mask = GetSubnet.apply(y_masked.abs(), self.sparsity) # obtain region wise mask
        # region_mask = torch.repeat_interleave(torch.repeat_interleave(mask, self.region_size, dim=2), self.region_size, dim=3) # expand to image dimension
        # # self.region_mask += torch.sum(region_mask, dim=0)
        # # Pixel level masking
        # # masked_delta = delta * self.scores
        # # mask = GetSubnet.apply(masked_delta.abs(), self.sparsity)
        # self.update_last_batch(input)
        # ------------------------------------------------------
        #------------- Mask learning on direct input -----------------
        scores = self.region_mask_generator(input.tensors, frame_ids)
        # mask = GetSubnet.apply(scores.abs(), self.sparsity)
        mask = scores > self.threshold # do not constrain to a fixed sparsity for every image
        region_mask = torch.repeat_interleave(torch.repeat_interleave(mask, self.region_size, dim=2), self.region_size, dim=3) # expand to image dimension
        proxy_mask = GetSubnet.apply(self.scores.abs(), self.sparsity)
        if 1 in frame_ids:
            first_frame_id = torch.where(frame_ids==1)[0][0]
            region_mask[first_frame_id:] = proxy_mask.repeat(region_mask.shape[0]-first_frame_id, 1, 1, 1)
        # region_mask[-1] = torch.ones_like(region_mask[-1]) # always process the first frame of the batch
        region_mask[-1] = proxy_mask
        # ------------------------------------------------------------
        #-------------- Dynamic Masking with output of previous batch-----------------------
        # weight = torch.ones(1, self.scores.shape[0], self.region_size, self.region_size, device=input.tensors.device)
        # region_level_scores = F.conv2d(self.scores, weight, stride=self.region_size) # pixel level to region level
        # mask = GetSubnet.apply(region_level_scores.abs(), self.sparsity) # region level mask
        # region_mask = torch.repeat_interleave(torch.repeat_interleave(mask, self.region_size, dim=1), self.region_size, dim=2) # expand to image dimension
        # if 1 in frame_ids:
        #     first_frame_id = (frame_ids == 1).nonzero(as_tuple=True)[0].item()
        #     if first_frame_id > 0:
        #         input.tensors[:first_frame_id] = input.tensors[:first_frame_id] * region_mask
        #     init_mask = F.conv2d(self.init_value.unsqueeze(0), weight, stride=self.region_size)
        #     mask = GetSubnet.apply(init_mask.abs(), self.sparsity)
        #     region_mask_init = torch.repeat_interleave(torch.repeat_interleave(mask, self.region_size, dim=1), self.region_size, dim=2) # expand to image dimension
        #     input.tensors[first_frame_id:] = input.tensors[first_frame_id:] * region_mask_init
        # else:
        #     input.tensors = input.tensors * region_mask
        # ----------------------------------------------------------------------------------
        #-------------- Masking using Original Labels -----------------------
        # img_h, img_w = input.tensors.shape[2:]
        # bbox_batch = [t['boxes'] for t in targets]
        # region_mask = []
        # for i in range(len(bbox_batch)):
        #     rmask = torch.zeros((input.tensors.shape[2:]))
        #     b = box_ops.box_cxcywh_to_xyxy(bbox_batch[i])
        #     b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).cuda()
        #     for bb in b:
        #         x1, y1, x2, y2 = bb
        #         rmask[int(y1):int(y2), int(x1):int(x2)] = 1
        #     region_mask.append(rmask.unsqueeze(0).unsqueeze(0))
        # region_mask = torch.cat(region_mask, dim=0)
        # input.tensors = input.tensors * region_mask.to(input.tensors.device)
        # -------------------------------------------------------------------
        # self.region_mask += torch.sum(region_mask, dim=0) # uncomment to save the mask during inference
        input.tensors = input.tensors * region_mask
        return input, region_mask
        # cur_input = input.tensors
        # prev_input = self.get_prev_input(cur_input, frame_ids)
        # # if self.last_batch is None:
        # #     self.last_batch = torch.zeros(input.tensors.shape).cuda()

        # # prev_input = torch.cat((self.last_batch[-1].unsqueeze(0), cur_input[:-1]), dim=0)
        # # prev_input[frame_ids == 1] = 0.0 # for first frame of each sequence, prev_input is set to 0
        # x = cur_input - prev_input # prev_input is the last processed frame, cur_input is the frame being currently processed 
        # self.delta += torch.sum(torch.max(torch.abs(x), dim=1, keepdim=True)[0], dim=0)
        # # mask = torch.max(torch.abs(x), dim=1, keepdim=True)[0]
        # # pixel_mask = torch.max(torch.abs(x), dim=1, keepdim=True)[0] > pixel_threshold # calculate absolute pixel wise differences 
        # # tiled_mask = torch.repeat_interleave(pixel_mask, dim=1, repeats=x.shape[1])
        # # pixel_mask = torch.abs(x) > self.pixel_threshold 
        # # Masking
        # # pixel_mask = torch.max(torch.abs(x), dim=1, keepdim=True)[0] * self.pixel_threshold
        # # Thresholding
        # pixel_mask = self.threshold_func(torch.max(torch.abs(x), dim=1, keepdim=True)[0] / self.pixel_threshold) # pixel threshold with 1 channel
        # self.delta_masked += torch.sum(torch.max(pixel_mask, dim=1, keepdim=True)[0], dim=0)
        # # pixel_mask = self.threshold_func(torch.abs(x) / self.pixel_threshold) # pixel threshold with 3 channels 
        # pixels_above_threshold = torch.sum(pixel_mask, dim=(1,2,3))/torch.prod(torch.tensor(pixel_mask.shape[1:]))
        # frame_mask = self.threshold_func(pixels_above_threshold/self.decision_threshold) 
        # frame_mask[frame_ids == 1] = 1.0 # always process the first frame of a sequence
        # frame_mask[0] = 1.0 # always process the first frame of the batch
        # # TODO: save output of last batch and use it for the next batch to remove this condition
        # self.update_last_batch(input, frame_mask)
        # return frame_mask
 
    

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.maskregion = MaskRegion(sparsity=0.6, threshold=0.1) # sparsty is the percentage of pixels to be preserved
        self.last_batch = None

    def update_last_batch(self, input):
        self.last_batch = input.tensors

    def filter_frames(self, out, frame_mask, train=False):
        # if not train:
        logits = out['pred_logits']
        bboxes = out['pred_boxes']
        # if frame_mask[0] == 0:
        #     logits[0] = self.last_output['pred_logits'][-1]
        #     bboxes[0] = self.last_output['pred_boxes'][-1]
        #     frame_mask[0] = 1.0
        # indices of the processed frames
        true_indices = torch.nonzero(frame_mask).flatten()
        # Create an index tensor with a shift of 1 to find the end index of each segment with same output
        end_indices = torch.roll(true_indices, shifts=-1) 
        end_indices[-1] = len(logits) # last index is the length of the output
        # Replace output of the skipped frames with the output of the last processed frame
        out['pred_logits'] = logits[true_indices.repeat_interleave(end_indices - true_indices)]
        out['pred_boxes'] = bboxes[true_indices.repeat_interleave(end_indices - true_indices)]
        # else:
            # # out['pred_logits'] = out['pred_logits'] * frame_mask.unsqueeze(1).unsqueeze(2)
            # # out['pred_boxes'] = out['pred_boxes'] * frame_mask.unsqueeze(1).unsqueeze(2)
            # logits = out['pred_logits']
            # bboxes = out['pred_boxes']
            # true_indices = torch.nonzero(frame_mask).flatten()
            # # Create an index tensor with a shift of 1 to find the end index of each segment with same output
            # end_indices = torch.roll(true_indices, shifts=-1) 
            # end_indices[-1] = len(logits) # last index is the length of the output
            # mask = torch.tensor(frame_mask, dtype=torch.bool).unsqueeze(1).unsqueeze(2)
            # # new_batch_size = int(torch.sum(frame_mask).item())
            # # out_shape = (new_batch_size, out['pred_logits'].shape[1], out['pred_logits'].shape[2])
            # masked_logits = torch.masked_select(logits, mask)
            # masked_bboxes = torch.masked_select(bboxes, mask)
            # out['pred_logits'] = torch.repeat_interleave(masked_logits, (end_indices-true_indices)).view(logits.shape)
            # out['pred_boxes'] = torch.repeat_interleave(masked_bboxes, (end_indices-true_indices)).view(bboxes.shape)
        return out
    
    def filter_input(self, input, frame_mask):
        true_indices = torch.nonzero(frame_mask).flatten()
        # Create an index tensor with a shift of 1 to find the end index of each segment with same output
        end_indices = torch.roll(true_indices, shifts=-1) 
        end_indices[-1] = len(frame_mask) # last index is the length of the output
        input.tensors = input.tensors[true_indices.repeat_interleave(end_indices - true_indices)]
        return input
    

    def mask_region(self, input, region_size=8, region_threshold=0.005, frame_ids=None):
        # heatmap of objects in the training set
        with open('datasets/hm_train.npy', 'rb') as f:
            hm_train = np.load(f)
        hm_train = torch.tensor(hm_train/hm_train.max(), device=input.tensors.device, dtype=torch.float)
        cur_input = input.tensors
        # get previous input
        if self.last_batch is None:
            self.last_batch = torch.zeros(input.tensors.shape).cuda()
            self.region_mask = torch.zeros_like(input.tensors[0][0].unsqueeze(0))

        prev_input = torch.cat((self.last_batch[-1].unsqueeze(0), cur_input[:-1]), dim=0)
        prev_input[frame_ids == 1] = 0.0 # for first frame of each sequence, prev_input is set to 0
        delta = cur_input - prev_input # prev_input is the last processed frame, cur_input is the frame being currently processed 

        masked_delta = delta * hm_train
        weight = torch.ones(1, input.tensors.shape[1], region_size, region_size, device=input.tensors.device)
        y = F.conv2d(masked_delta, weight, stride=region_size) 
        y_masked = (torch.abs((y/y.max()))>region_threshold).to(torch.float) # decide which regions to keep
        region_mask = torch.repeat_interleave(torch.repeat_interleave(y_masked, region_size, dim=2), region_size, dim=3) # expand to image dimension
        self.region_mask += torch.sum(region_mask, dim=0)
        self.update_last_batch(input)
        input.tensors = input.tensors * region_mask
        region_skipped = torch.sum(y_masked == 0)/torch.prod(torch.tensor(y_masked.shape))
        return input, region_skipped
    

    def forward(self, samples: NestedTensor, frame_ids=None, period=None, train=False, targets=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # if period is None:
        #     frame_mask = self.mask_frame(samples, frame_ids)
        # else:
        #     # frame_mask = torch.ones_like(frame_ids)
        #     frame_mask = torch.zeros_like(frame_ids)
        #     # Define the indices for the first frame of each sequence
        #     first_frame_indices =  torch.nonzero(frame_ids == 1).flatten()
        #     # Set zeros at regular intervals
        #     # frame_mask[period-1::period] = 0
        #     frame_mask[::period] = 1
        #     # Restart the pattern at specified indices
        #     # for idx in first_frame_indices:
        #     #     frame_mask[idx:] = 1.0
        #     #     frame_mask[idx+period-1::period] = 0
        #     for idx in first_frame_indices:
        #         frame_mask[idx:] = 0
        #         frame_mask[idx::period] = 1
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # Region Skipping
        frame_mask = torch.ones_like(frame_ids)
        # samples, region_skipped = self.mask_region(samples, frame_ids=frame_ids)
        samples, region_mask = self.maskregion(samples, frame_ids, targets)
        features, pos = self.backbone(samples)

        # frame_mask = self.mask_frame(features[0], frame_ids)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        # out = self.filter_frames(out, frame_mask, train=train)
        out['frame_mask'] = frame_mask
        self.maskregion.update_scores(samples, out, frame_ids) # dynamic mask
        out['region_mask'] = region_mask
        # self.update_last_output(out)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_frame_count(self, outputs, targets, indices, num_boxes):
        """Compute the fraction of frames processed by the network"""
        frame_mask = outputs['frame_mask']
        frame_count = torch.sum(frame_mask, dim=0)
        total_frames = len(targets)
        loss_frame_count = frame_count/total_frames
        losses = {'loss_frame_count': loss_frame_count}
        return losses
    
    def fs_metrics(self, outputs, targets, indices, num_boxes, object_to_image_dt=None):
        # Filter the matched predictions and find class labels
        frame_id = [target['image_id'] for i, target in enumerate(targets) if outputs['frame_mask'][i] == 1.0]
        pred_labels = [outputs['pred_logits'][i,ind[0],:].topk(1)[1] for i, ind in enumerate(indices)]
        pred_labels = [lab.squeeze() for i, lab in enumerate(pred_labels) if outputs['frame_mask'][i] == 1.0]
        target_labels = [target['labels'] for target in targets]
        target_labels = [lab[ind[1]] for i, (lab, ind) in enumerate(zip(target_labels, indices)) if outputs['frame_mask'][i] == 1.0]
        target_ids = [target['track_ids'] for i, target in enumerate(targets) if outputs['frame_mask'][i] == 1.0]
        pred_ids = [target_id[target_labels[i] == pred_labels[i]] for i, target_id in enumerate(target_ids)]
        img_to_obj_id = {frame_id[i]: pred_ids[i] for i in range(len(frame_id))}
        for image_id, object_ids in img_to_obj_id.items():
            for object_id in object_ids:
                if object_id.item() not in object_to_image_dt:
                    object_to_image_dt[object_id.item()] = image_id.item()
        return object_to_image_dt
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'frame_count': self.loss_frame_count
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, object_to_image_dt=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        if object_to_image_dt is not None:
            object_to_image_dt = self.fs_metrics(outputs, targets, indices, num_boxes, object_to_image_dt)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if object_to_image_dt is not None:
            return losses, object_to_image_dt
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    if args.num_classes is not None:
        print('Building a DETR model with %s classes' % args.num_classes)
        num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_frame_count'] = args.frame_count_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'frame_count']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
