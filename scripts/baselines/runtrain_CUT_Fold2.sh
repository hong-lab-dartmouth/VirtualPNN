python train.py \
--dataroot ../PNN_finetune/cv_splits/fold_2 \
--name CUT_Fold2 \
--continue_train \
--pretrained_name aiffpe_weights \
--model cut \
--CUT_mode CUT \
--checkpoints_dir checkpoints \
--epoch 5 \
--n_epochs 20 \
--n_epochs_decay 20 \
--batch_size 8 \
--load_size 256 \
--crop_size 256 \
--self_regularization 0 \
--gpu_id 1







