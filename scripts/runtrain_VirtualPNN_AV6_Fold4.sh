python train.py \
--dataroot ../PNN_finetune/cv_splits_AV6/fold_4 \
--name VirtualPNN_AV6_Fold4 \
--continue_train \
--pretrained_name aiffpe \
--CUT_mode CUT \
--checkpoints_dir ../checkpoints \
--lr_G 0.00005 \
--lr_D 0.00002 \
--epoch 5 \
--n_epochs 20 \
--n_epochs_decay 20 \
--batch_size 8 \
--load_size 256 \
--crop_size 256 \
--self_regularization 0 \
--display_id -1 \
--gpu_id 3 \
--lambda_VGG 2 \
--use_vessel_patches \
--vessel_patch_ratio 0.3 \
--lambda_anti_vessel 5 \
--lambda_NCE 1



