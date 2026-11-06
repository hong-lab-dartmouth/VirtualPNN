"""Inference on a cross-validation test split, as used for the paper's results.

Loads a trained generator from --checkpoints_dir/--name and writes, for each of the first --num_test
test patches, the H&E input (real_A), the virtual aggrecan image (fake_B) and the real aggrecan patch
(real_B) under --results_dir/<phase>_<epoch>/images/, with an HTML index. It computes no metric: the
per-fold commands in scripts/ run postprocess/reconstruct_noa.py, pytorch_fid and
postprocess/signal_ratio.py on these folders (INFERENCE.md).

Example (fold 1 of the main model; scripts/TEST_VirtualPNN_AV6_Fold1.sh is the verbatim command):
    python test.py --dataroot <fold dir with testA/ and testB/> --name VirtualPNN_AV6_Fold1 \
        --checkpoints_dir weights --model virtualpnn --CUT_mode CUT --epoch latest --phase test \
        --eval --no_flip --load_size 256 --crop_size 256 --batch_size 1 --num_test 2005 --results_dir <out>

CycleGAN checkpoints use --model cycle_gan. For H&E images without the testA/testB layout, use infer.py.
See options/base_options.py and options/test_options.py for every option.
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    train_dataset = create_dataset(util.copyconf(opt, phase="test"))
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.results_dir, opt.phase, opt.epoch))

    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
    webpage.save()  # save the HTML
