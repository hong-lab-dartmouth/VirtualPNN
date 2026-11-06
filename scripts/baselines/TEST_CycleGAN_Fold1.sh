RDIR="checkpoints/CycleGAN_Fold1_TEST"

# python test.py \
# --dataroot ../PNN_finetune/cv_splits/fold_1 \
# --name CycleGAN_Fold1 \
# --checkpoints_dir checkpoints \
# --model cycle_gan \
# --epoch latest \
# --phase test \
# --eval \
# --gpu_ids 1 \
# --load_size 256 \
# --crop_size 256 \
# --no_flip \
# --batch_size 1 \
# --num_test 2005 \
# --results_dir $RDIR

# FID=$(python -m pytorch_fid ${RDIR}/test_latest/images/fake_B ${RDIR}/test_latest/images/real_B)
# echo "FID for ${RDIR}: ${FID}" > ${RDIR}/FID.txt
# echo "FID for ${RDIR}: ${FID}"

# python ../reconstruct_noa.py --patch_dir ${RDIR}/test_latest/images/fake_B --output_dir ${RDIR}/test_latest/images/reconstructed --patch_size 256 --image_size 1536


RDIR="checkpoints/CycleGAN_Fold1_TEST_vessel"

# python test.py \
# --dataroot ../PNN_finetune/cv_splits/fold_1/vessel \
# --name CycleGAN_Fold1 \
# --checkpoints_dir checkpoints \
# --model cycle_gan \
# --epoch latest \
# --phase test \
# --eval \
# --gpu_ids 0 \
# --load_size 256 \
# --crop_size 256 \
# --no_flip \
# --batch_size 1 \
# --num_test 2005 \
# --results_dir ${RDIR}

# python ../reconstruct_noa.py --patch_dir ${RDIR}/test_latest/images/fake_B --output_dir ${RDIR}/test_latest/images/reconstructed_vessel --patch_size 256 --image_size 1536
python ../vessel_signal.py --patch_folder ${RDIR}/test_latest/images/reconstructed_vessel --out_file ${RDIR}/vessel_signals.txt
