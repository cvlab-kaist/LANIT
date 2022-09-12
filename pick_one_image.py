"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import shutil
from collections import OrderedDict
from tqdm import tqdm

import numpy as np
import torch

#from metrics.fid import calculate_fid_given_paths
#from metrics.lpips import calculate_lpips_given_images
from core.data_loader import get_eval_loader, get_filePathEval_loader
from core import utils
from core import solver
import torch.nn.functional as F
import torchvision.transforms as transforms
import clip
from PIL import Image
import cv2


def generate_sample():
    files = cv2.imread(os.path.join('./test_results', name ,'final_2', 'latent_3_{}.jpg'.format(i)), cv2.IMREAD_COLOR)#Image.open(os.path.join('./test_results', args.name, 'reference_{}.jpg'.format(i)))
    #os.makedirs(os.path.join('test_results', name, 'final_{}'.format(i), 'src'), exist_ok=True)
    #os.makedirs(os.path.join('test_results', name, 'final_{}'.format(i), 'ref'), exist_ok=True)
    os.makedirs(os.path.join('test_results', name, 'final_{}'.format(i), 'fake'), exist_ok=True)

    step = 256
    for lst in range(len(cp_list)):
        p_ref = int(cp_list[lst][0])
        p_src = int(cp_list[lst][1])
        
        x_ref = files[p_src*step:p_src*step+step, :step, :] 
        x_src = files[:step, p_ref*step:p_ref*step+step, :]
        x_fake = files[p_src*step:p_src*step+step, p_ref*step:p_ref*step+step, :] 
        
        save = str(i).zfill(2)
        
        cv2.imwrite(os.path.join('test_results', name, 'final_{}'.format(i), 'src', '%s_%s.png' %(lst,save)), x_src)
        #cv2.imwrite(os.path.join('test_results', name, 'final_{}'.format(i), 'ref', '%s_%s.png' %(lst,save)), x_ref)
        cv2.imwrite(os.path.join('test_results', name, 'final_{}'.format(i), 'fake', '%s_%s.png' %(lst,save)), x_fake)


if __name__ == '__main__':
    name = 'celeb_10_top_diff_2'
    i = 15
    cp_list = [[1,3]]
    generate_sample()




