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

from core.data_loader import get_eval_loader, get_filePathEval_loader
from core import utils
from core import solver
import torch.nn.functional as F
import clip


@torch.no_grad()
def calculate_metrics(nets, args, prompt_idx, prompt, base_template, mode):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clip_model, preprocess = clip.load('ViT-B/32', device="cpu", jit=False) 
    """ freeze the network parameters """
    for clip_param in clip_model.parameters():
        clip_param.requires_grad = False
    clip_model = clip_model.to(device)

    print('Calculating evaluation metrics...')

    domains = []

    if ("animal" in args.dataset) or ("celeb"in args.dataset) or ("food" in args.dataset):
        domains_list = os.listdir(args.val_img_dir)
        domains_list.sort() 
        for idx, i in enumerate(prompt_idx):
            domains.append(domains_list[i])
        num_domains = len(domains) 

        print('Number of domains: %d' % num_domains)


        tasks = domains 
        path_fakes = []
        for i in range(len(tasks)):
            """ tasks 별 directory 생성 """
            task = tasks[i]
            if mode == 'reference':
                path_fake = os.path.join(args.eval_dir, args.name,'reference', task)
            else:
                path_fake = os.path.join(args.eval_dir, args.name,'latent', task)
            shutil.rmtree(path_fake, ignore_errors=True)
            os.makedirs(path_fake)
            path_fakes.append(path_fake)
            
        lpips_dict = OrderedDict()
        for trg_idx, trg_domain in tqdm(enumerate(domains), desc="trg_num", total=len(domains)):
            src_domains = [x for x in domains if x != trg_domain]

            path_ref = os.path.join(args.val_img_dir, trg_domain)
            loader_ref = get_eval_loader(root=path_ref,
                                        img_size=args.img_size,
                                        batch_size=args.val_batch_size*5,
                                        imagenet_normalize=False,
                                        drop_last=True)
            iter_ref = iter(loader_ref)
            x_ref = next(iter_ref).to(device)
                
            for src_idx, src_domain in tqdm(enumerate(src_domains), desc="src_num", total=len(src_domains)):
                """ source dataloader """
                path_src = os.path.join(args.val_img_dir, src_domain)
                loader_src = get_eval_loader(root=path_src,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            imagenet_normalize=False)

                N = 5 
                #generate 10 outputs from the same input
                group_of_images = []
                if mode == 'latent':
                    z_trg = torch.randn(args.val_batch_size*N, args.latent_dim).to(device)
                try:
                    x_src = next(iter_src).to(device)
                    x_ref = next(iter_ref).to(device)
                except:
                    iter_src = iter(loader_src)
                    x_src = next(iter_src).to(device)
                    iter_ref = iter(loader_ref)
                    x_ref = next(iter_ref).to(device)  

                if args.use_base:
                    sim_ref_base = solver.get_sim_from_clip(args, x_ref, clip_model, base_template, device=device).float()
                    sim_src_base = solver.get_sim_from_clip(args, x_src, clip_model, base_template, device=device).float()

                    if args.use_prompt:
                        sim_ref =solver.cal_clip_loss(args, nets, x_ref , clip_model, prompt, device=device)
                        sim_src =solver.cal_clip_loss(args, nets, x_src , clip_model, prompt, device=device)
                    else:
                        sim_ref =solver.get_sim_from_clip(args, x_ref , clip_model, prompt, device=device)
                        sim_src =solver.get_sim_from_clip(args, x_src , clip_model, prompt, device=device)
                    
                    _ , y_trg = solver.get_label_from_sim(args, prompt_idx, sim_src, sim_ref, sim_src_base, sim_ref_base)
                else:
                    sim_src, y_org, sim_ref, y_trg = solver.get_unsup_labels(args, nets, x_src, x_ref, clip_model, prompt, prompt_idx, base_template, device, norm=norm, detach=False)

                
                """ get style code """
                if mode == 'latent':
                    s_trg = nets.mapping_network(z_trg, y_trg)
                else:
                    s_trg = nets.style_encoder(x_ref, y_trg)
                """ end style code extraction """

                x_src = x_src.repeat(N,1,1,1)
                """ end """

                """ generation """
                x_fake = nets.generator(x_src, s_trg)
                group_of_images.append(x_fake)

                #save generated images to calculate FID later
                """ save images """
                for num_img_idx in range(args.val_batch_size * N):
                    path_fake = path_fakes[trg_idx]
                    filename = os.path.join(
                        path_fake,
                    '%.2i2%.2i_%.2i.png' % (src_idx, trg_idx,  num_img_idx))
                    utils.save_image(x_fake[num_img_idx], ncol=1, filename=filename)

            # delete dataloaders
            del loader_src
            if mode == 'reference':
                del loader_ref
                del iter_ref
