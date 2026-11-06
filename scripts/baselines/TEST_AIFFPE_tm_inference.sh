#!/bin/bash

# Loop over folds 1 to 5
for FOLD in 1 2 3 4 5; do
    echo "Running inference for Fold ${FOLD}"

    RDIR="checkpoints/AIFFPE_Fold${FOLD}_tumorTEST"

    # Inference
    python test.py \
        --dataroot ../PNN_finetune/Dec25_tumor_patches \
        --name AIFFPE_Fold${FOLD} \
        --checkpoints_dir checkpoints \
        --model cut \
        --CUT_mode CUT \
        --epoch latest \
        --phase test \
        --eval \
        --gpu_ids 1 \
        --load_size 256 \
        --crop_size 256 \
        --no_flip \
        --batch_size 1 \
        --num_test 2688 \
        --results_dir ${RDIR}

    # Reconstruct full cores
    python ../reconstruct_noa.py \
        --patch_dir ${RDIR}/test_latest/images/fake_B \
        --output_dir ${RDIR}/test_latest/images/tm_reconstructed \
        --patch_size 256 \
        --image_size 1536

    # Compute off-target signal
    python ../vessel_signal.py \
        --patch_folder ${RDIR}/test_latest/images/tm_reconstructed \
        --out_file ${RDIR}/tumor_signals_AIFFPE_Fold${FOLD}.txt

done
