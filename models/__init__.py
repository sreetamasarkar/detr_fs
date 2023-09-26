# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .detr_fs import build as build_fs

def build_model(args):
    if args.frame_skipping:
        return build_fs(args)
    else:
        return build(args)
