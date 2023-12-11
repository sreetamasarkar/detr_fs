Train Region Mask Generator
```
python train_region_mask_generator.py --dataset_file kitti_tracking --num_classes 4 --output_dir results_fs_debug --lr 0.001 --epochs 50 --batch_size 4
```

Train DETR with Region Mask Generator
```
CUDA_VISIBLE_DEVICES=1 python main.py --no_aux_loss --dataset_file kitti_tracking --num_classes 4 --resume results1/checkpoint.pth --output_dir results_region_skip_cnn_sparsity0.5_t0.2 --lr 1e-5 --lr_backbone 1e-6 --lr_threshold 0.001 --epochs 100 --batch_size 4 --frame_skipping  --frame_count_loss_coef 0.0
```