# Inference: regenerate the model outputs from H&E

Run this once per model x fold, then compute metrics as below. With the released weights the
outputs are deterministic (`--eval`, `--no_flip`, batch size 1).

## 1. Get the weights
```
./fetch_weights.sh            # downloads the archive + sha256-verifies against WEIGHTS_MANIFEST.csv
```
This lays the 25 generators out as `weights/<name>/latest_net_G.pth`, where `<name>` is the
checkpoint name `test.py` expects:

| Released file | `--name` | `--model` |
|---|---|---|
| VirtualPNN-P+AV_Fold{k}_net_G.pth | VirtualPNN_AV6_Fold{k} | virtualpnn |
| VirtualPNN-P_Fold{k}_net_G.pth | VirtualPNN_noAV_Fold{k} | virtualpnn |
| CUT_Fold{k}_net_G.pth | CUT_Fold{k} | virtualpnn |
| AI-FFPE_Fold{k}_net_G.pth | AIFFPE_Fold{k} | virtualpnn |
| CycleGAN_Fold{k}_net_G.pth (= G_A) and CycleGAN_Fold{k}_net_G_B.pth | CycleGAN_Fold{k} | cycle_gan |

`--model cycle_gan` in test mode loads both generators, `G_A` (H&E to virtual aggrecan) and `G_B` (the reverse
direction), so the archive holds both and `fetch_weights.sh` writes them as `latest_net_G_A.pth` and
`latest_net_G_B.pth`. The virtual stain is `fake_B`.

## 2. Run the generator (main model, fold 1 shown; verbatim in `scripts/TEST_VirtualPNN_AV6_Fold1.sh`)

`--dataroot` is reproduced as the authors ran it and points at their cross-validation splits;
substitute your own directory in the CUT layout (`testA/` H&E, `testB/` aggrecan). `--num_test`
is the size of that fold's test split; set it to your own count. To run on images that have no
paired aggrecan channel, use `infer.py` instead; see the README.
```
RDIR=results/VirtualPNN_AV6_Fold1
python test.py \
  --dataroot ../PNN_finetune/cv_splits_AV6/fold_1 \
  --name VirtualPNN_AV6_Fold1 --checkpoints_dir weights \
  --model virtualpnn --CUT_mode CUT --epoch latest --phase test --eval --no_flip \
  --load_size 256 --crop_size 256 --batch_size 1 --num_test 2005 --results_dir $RDIR
```
Baselines and the ablation use the same call with their `--name`/`--model` from the table
(CycleGAN: `--model cycle_gan`). `../reconstruct_noa.py` and `../vessel_signal.py` in the TEST scripts are
`postprocess/reconstruct_noa.py` and `postprocess/signal_ratio.py` of this repository.

## 3. Post-process into cores and metrics
```
# stitch 256px patches into 1536px cores
python postprocess/reconstruct_noa.py \
  --patch_dir $RDIR/test_latest/images/fake_B \
  --output_dir $RDIR/test_latest/images/reconstructed --patch_size 256 --image_size 1536

# FID (Figure 3)
python -m pytorch_fid $RDIR/test_latest/images/fake_B $RDIR/test_latest/images/real_B

# signal_ratio: the stained-pixel fraction inside a circular tissue mask. One script serves
# both the vessel and the tumor cores; point --patch_folder at the reconstructed set you want.
#
# vessel (Figure 4A): run test.py on the fold's /vessel split first, then:
python postprocess/signal_ratio.py \
  --patch_folder $RDIR/test_latest/images/reconstructed_vessel --out_file $RDIR/vessel_signals.txt

# tumor negative controls (supplementary figure): the same script over the reconstructed tumor cores
python postprocess/signal_ratio.py \
  --patch_folder $RDIR/test_latest/images/reconstructed_tumor --out_file $RDIR/tumor_signals.txt
```

## 4. Your own H&E
To apply the model to your own H&E, use `infer.py`. It takes ordinary image files and does not need the
`testA`/`testB` layout or a ground-truth aggrecan channel.
