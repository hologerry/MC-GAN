################################################################################
# MC-GAN
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# By Samaneh Azadi
################################################################################


import time
import os
from options.test_options import TestOptions

opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html

import torch

opt.nThreads = 1  # test code only supports nThreads=1
opt.batchSize = 1  # test code only supports batchSize=1
opt.serial_batches = True  # no shuffle
opt.stack = True
opt.use_dropout = False
opt.use_dropout1 = False

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch + '+' + opt.which_epoch1))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (
    opt.name, opt.phase, opt.which_epoch + '+' + opt.which_epoch1))

l1_loss_file = os.path.join(opt.results_dir, opt.phase, "l1_loss.txt")
cnt = 0
mean_l1_loss = 0.0

# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    real_B = visuals['real_B_t']
    fake_B = visuals['fake_B_t']
    with torch.no_grad():
        test_l1_loss = torch.nn.functional.l1_loss(fake_B, real_B)
        mean_l1_loss += test_l1_loss.item()
    cnt += 1
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()

mean_l1_loss /= cnt
with open(l1_loss_file, "w") as f:
    f.write(str(mean_l1_loss))
