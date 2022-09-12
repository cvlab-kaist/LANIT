"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
from tqdm import trange
import functools
import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision.transforms.functional as FF

from core.utils import *
from core.model import build_model
from core.checkpoint import CheckpointIO
import core.utils as utils
from core.utils import clip_normalize
from metrics.eval import calculate_metrics
from visualizer import Visualizer

import clip
from copy import deepcopy

class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        """
        self._set_attr_prompt(self.args):

        return
            self.template: "a photo of the {}." (dataset마다 다름.)
            self.prompt: 우리가 사용하기로 설정한 prompt들. ex) ["fox", "dog", ... etc]
            self.prompt_idx: 해당 prompt가 all_prompt의 몇 번째에 존재하는지에대한 index. ex) [2, 1, 0, ...] (데이터셋에서 폴더 순서.)
            args.num_domains: 우리가 사용하는 prompt의 개수. (=len(prompt))
        """
        self.template, self.prompt, self.prompt_idx, args.num_domains, self.base_template = self._set_attr_prompt(self.args)
        self.prompt = [self.template.format(x) for x in self.prompt] # template과 결합하여 문장으로 만듬. ex) ["a photo of the fox." , ... , ]

        """ network definition """
        self.nets, self.nets_ema = build_model(self.args)
            
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if self.args.mode == 'train':
            self.optims = Munch()
            self.scheduler = Munch()
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                if net=='mapping_network':
                    lr = self.args.m_lr
                elif "promptLearner" in net:
                    lr = self.args.p_lr
                else:
                    lr = self.args.lr

                """ optimizer & shceduler"""
                if net=="promptLearner":
                    self.optims[net] = torch.optim.SGD(
                        params=self.nets[net].parameters(),
                        lr = 1e-5,
                        )
                    self.scheduler[net] = lr_scheduler.CosineAnnealingLR(self.optims[net], T_max=20, eta_min=1e-7, verbose=True)

                else:
                    self.optims[net] = torch.optim.Adam(
                        params=self.nets[net].parameters(),
                        lr = lr,
                        #lr=args.f_lr if net == 'mapping_network' else args.lr,
                        betas=[args.beta1, args.beta2],
                        weight_decay= args.weight_decay)

            if args.use_prompt:
                self.ckptios = [
                        CheckpointIO(ospj(args.checkpoint_dir, args.name, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
                        CheckpointIO(ospj(args.checkpoint_dir, args.name, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema),
                        CheckpointIO(ospj(args.checkpoint_dir, args.name, '{:06d}_optims.ckpt'), **self.optims),
                        CheckpointIO(ospj(args.checkpoint_dir, args.name, '{:06d}_shceduler.ckpt'), **self.scheduler),
                    ]
            else:
                self.ckptios = [
                                CheckpointIO(ospj(args.checkpoint_dir, args.name, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
                                CheckpointIO(ospj(args.checkpoint_dir, args.name, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema),
                                CheckpointIO(ospj(args.checkpoint_dir, args.name, '{:06d}_optims.ckpt'), **self.optims),
                            ]
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, args.name, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema)]

        """ network definition이 모두 끝났으면, device 위치로 network를 올린다. """
        self.to(self.device)

        """ network initialization을 진행한다. """
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            print(name)
            if ('ema' not in name) and ('fan' not in name) and ('prompt' not in name)  and ('Prompt' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)
        
        """ initialization이 끝난 후에, pretrained clip을 load """
        self.clip_model, preprocess = clip.load('ViT-B/32', device="cpu", jit=False)
        self.clip_model = self.clip_model.to(self.device)
        """ freeze the network parameters """
        for clip_param in self.clip_model.parameters():
            clip_param.requires_grad = False

    def _save_checkpoint(self, step):
        """ checkpoint 저장 """
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        """ checkpoint 불러오기 """
        for ckptio in self.ckptios:
            ckptio.load(self.args, step)

    def _reset_grad(self):
        """ gradient 초기화 """
        for optim in self.optims.values():
            optim.zero_grad()

    def _set_attr_prompt(self, args):
        """ define template, prompt, prompt_idx. class initialization. """
        template, prompt, prompt_idx, base_template = get_prompt_and_att(args)
        num_domains = len(prompt)
        return template, prompt, prompt_idx, num_domains, base_template

    def get_sim_mask(self, args, x_real, x_ref):
        """ if args.use_prompt then, sim_vals are not detached else detached. """
        sim_val_src = cal_clip_loss(args, self.nets, x_real, self.clip_model, self.prompt, device=self.device) # (batch, num_prompt)
        sim_val_ref = cal_clip_loss(args, self.nets, x_ref, self.clip_model, self.prompt, device=self.device)  # (batch, num_prompt)
        
        """ base prompt is not learned """
        sim_val_src_base = get_sim_from_clip(args, x_real, self.clip_model, self.base_template, device=self.device) # (batch, 1)
        sim_val_ref_base = get_sim_from_clip(args, x_ref, self.clip_model, self.base_template, device=self.device)  # (batch, 1)
        
        """
        abc = torch.Tensor([ [1,2,3], [4,5,6], [7,8,9]  ])
        torch.where(abc>5)
        (tensor([1, 2, 2, 2]), tensor([2, 0, 1, 2]))
        torch.where(abc>5)[0]
        tensor([1, 2, 2, 2])
        torch.where(abc>5)[1]
        tensor([2, 0, 1, 2])
        """

        """ To filter data that all the similarity values are lower than zero """
        src_mask = torch.unique( torch.where( (sim_val_src-sim_val_src_base)>0)[0] )

        if args.use_base:
            if args.multi_hot:
                ref_mask = torch.unique( torch.where( (sim_val_ref-sim_val_ref_base)>0)[0] ) 
        else:
            if args.multi_hot:
                ref_mask = torch.unique( torch.where( (sim_val_ref)>0)[0] )
        return sim_val_src, sim_val_ref, sim_val_src_base , sim_val_ref_base, src_mask, ref_mask

    def get_data_iter(self, args, loaders):
        """ iter 당 data를 가져오는 함수 """
        #import pdb; pdb.set_trace()
        try:
            x_src, real_y_src = next(self.train_src)
            x_ref, x_ref_2, real_y_ref = next(self.train_ref)
        except:
            self.train_src = iter(loaders.src)
            self.train_ref = iter(loaders.ref)
            x_src, real_y_src = next(self.train_src)
            x_ref, x_ref_2, real_y_ref = next(self.train_ref)
        
        """ define latent vecotrs """
        z_trg = torch.randn(x_src.size(0), args.latent_dim).to(self.device)
        z_trg_2 = torch.randn(x_src.size(0), args.latent_dim).to(self.device)

        """ data to device """
        x_real = x_src.to(self.device)
        x_ref = x_ref.to(self.device)
        x_ref_2 = x_ref_2.to(self.device)

        b = args.batch_size
        if args.use_base:
            sim_val_src, sim_val_ref, sim_val_src_base , sim_val_ref_base, src_mask, ref_mask = self.get_sim_mask(args, x_real, x_ref)

            """ To do. hard coded"""
            while (len(ref_mask) < 8) :
                try:
                    x_ref, x_ref_2, real_y_ref = next(self.train_ref)
                except:
                    self.train_ref = iter(loaders.ref)
                    x_ref, x_ref_2, real_y_ref = next(self.train_ref)

                x_ref = x_ref.to(self.device)
                _, sim_val_ref, _, _,_,ref_mask = self.get_sim_mask(args, x_real, x_ref)

            return x_real[:b], x_ref[ref_mask[:b]], x_ref_2[:b], z_trg[:b], z_trg_2[:b], real_y_src[:b], real_y_ref[ref_mask[:b]], sim_val_src[:b], sim_val_ref[ref_mask[:b]], sim_val_src_base[:b] , sim_val_ref_base[ref_mask[:b]]
        else:
            return x_real[:b], x_ref[:b], x_ref_2[:b], z_trg[:b], z_trg_2[:b], real_y_src[:b], real_y_ref[:b]                

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims
        prompt = self.prompt
        base_template =self.base_template
        prompt_idx = self.prompt_idx
        clip_model = self.clip_model

        if args.use_prompt and args.step2:
            scheduler = self.scheduler

        """ data loader """
        self.train_src = iter(loaders.src)
        self.train_ref = iter(loaders.ref)
        self.val_src = iter(loaders.val_src)
        self.val_ref = iter(loaders.val_ref)

        """ start training """
        # resume training if necessary
        if args.resume_iter > 0:
            """ if resume_iter is larger than 0, then load checkpoint """
            self._load_checkpoint(args.resume_iter)

        
        """ scheduling a weight of style diversification loss """
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        visualizer = Visualizer(args)   # create a visualizer that display/save images and plots
        args.visualizer = visualizer

        """ generation process """
        if args.step2:
            batch_iter = (args.num_domains*300)//(args.batch_size)
            args.total_iters = args.resume_iter + int( batch_iter*20 )
            

        for i in range(args.resume_iter, args.total_iters):
            """ get image and label at this iteration """

            """ gradient of reference text prompt is controlled here. detach option. """
            """ if args.use_prompt, detach=False """
            if args.use_base:
                """ if args.use_prompt then, sim_vals are detached. """
                x_real, x_ref, x_ref_2, z_trg, z_trg_2, real_y_src, real_y_ref, sim_real, sim_ref, sim_base_real, sim_base_ref = self.get_data_iter(args, loaders)
                y_org, y_trg = get_label_from_sim(args, prompt_idx, sim_real, sim_ref, sim_base_real, sim_base_ref)
            else:
                # default norm: None
                x_real, x_ref, x_ref_2, z_trg, z_trg_2, real_y_src, real_y_ref  = self.get_data_iter(args, loaders)
                sim_real, y_org, sim_ref, y_trg = get_unsup_labels(args, nets, x_real, x_ref, clip_model, prompt, prompt_idx, base_template, self.device, detach=False)

            """ Discriminator step1. update discriminator """
            d_loss, d_losses_latent = compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=z_trg, device=self.device)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            """ Discriminator step2. mapping network & Update Discriminator """
            d_loss, d_losses_ref = compute_d_loss(nets, args, x_real, y_org, y_trg, x_ref=x_ref, device=self.device)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            """ Generator step1. Update Mapping net & Style encoder & Generator """
            g_loss, g_losses_latent = compute_g_unsup_loss(nets, args, clip_model, prompt, base_template, prompt_idx, x_real, y_org, y_trg, sim_ref = sim_ref, z_trgs=[z_trg, z_trg_2], device=self.device)

            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()
            if args.step2:
                optims.promptLearner.step()
                
            """ Generator step2. Update only Generator """
            if args.step2:
                sim_real, y_org, sim_ref, y_trg = get_unsup_labels(args, nets, x_real, x_ref, clip_model, prompt, prompt_idx, base_template, self.device, detach=False)
            
            g_loss, g_losses_ref = compute_g_unsup_loss(nets, args, clip_model, prompt, base_template, prompt_idx, x_real, y_org, y_trg, sim_ref = sim_ref, x_refs=[ x_ref , x_ref_2 ], device=self.device)

            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            if args.step2:
                optims.promptLearner.step()
                if i%batch_iter==0.:
                    scheduler.promptLearner.step()
    
            """ compute moving average of network parameters """
            moving_average(args, nets.generator, nets_ema.generator, beta=0.999)
            moving_average(args, nets.style_encoder, nets_ema.style_encoder, beta=0.999)
            moving_average(args, nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            if args.use_prompt:
                moving_average(args, nets.promptLearner, nets_ema.promptLearner, beta=0.999)

            """ weight decaying of the style diversity sensitive loss """
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            """ plotting current losses on command window """
            if (i+1) % args.print_every == 0:
                visualizer.reset() 
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                        ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds

                if args.step2:
                    # prompt learning 할경우 해당 loss 추가.
                    all_losses["P/map"] = g_losses_latent["prompt"]
                    all_losses["P/ref"] = g_losses_ref["prompt"]

                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

                visualizer.plot_current_losses(i, float(i) / len(loaders.src), all_losses)

            """ save checkpoint """
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)
            
            """ sampling in evry args.sample_every """
            with torch.no_grad():
                # generate images for debugging
                if (i+1) % args.sample_every == 0:
                    # validation에 사용할 이미지와 레이블을 뽑는다.
                    try:
                        inputs_val_src, real_y_src = next(self.val_src)
                        inputs_val_ref, _, real_y_ref_val = next(self.val_ref)
                    except StopIteration:
                        self.val_src = iter(loaders.val_src)
                        self.val_ref = iter(loaders.val_ref)
                        inputs_val_src, real_y_src_val = next(self.val_src)
                        inputs_val_ref, _, real_y_ref_val = next(self.val_ref)

                    inputs_val_src = inputs_val_src.to(self.device)
                    inputs_val_ref = inputs_val_ref.to(self.device)

                    sim_val_src, y_val_src, sim_val_ref, y_val_ref = get_unsup_labels(args, nets, inputs_val_src, inputs_val_ref, clip_model, prompt, prompt_idx, base_template, self.device)
                    y_val_src = y_val_src.to(self.device)
                    y_val_ref = y_val_ref.to(self.device)

                    """ end of loading validation data """
                    os.makedirs( os.path.join(args.sample_dir, args.name),  exist_ok=True)
                    utils.debug_image(nets_ema, args, inputs_val_src, inputs_val_ref, y_val_src, y_val_ref, step=i+1)

    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        clip_model = self.clip_model
        prompt = self.prompt
        prompt_idx = self.prompt_idx

        os.makedirs(ospj(args.result_dir, args.name), exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        train_src = iter(loaders.src)
        train_ref = iter(loaders.ref)

        src,_ = next(train_src) 
        ref,_, _ = next(train_ref) 

        sim_real, y_org, sim_ref, y_ref = get_unsup_labels(args, nets_ema, src.to(self.device), ref.to(self.device), clip_model, prompt, prompt_idx, self.base_template, self.device)

        if args.infer_mode == 'reference':
            fname = ospj(args.result_dir, args.name, 'reference.jpg')
            print('Working on {}...'.format(fname))
            utils.translate_using_reference(nets_ema, args, src.to(self.device), ref.to(self.device), y_ref.to(self.device), fname)

        else:
            fname = ospj(args.result_dir, args.name, 'latnet.jpg')
            print('Working on {}...'.format(fname))
            utils.translate_using_latent(nets_ema, args, src.to(self.device), ref.to(self.device), y_ref.to(self.device), fname)

    @torch.no_grad()
    def evaluate(self):
        prompt = self.prompt
        args   = self.args
        nets_ema = self.nets_ema
        infer_mode = args.infer_mode
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, self.prompt_idx, self.prompt, self.base_template, mode=infer_mode)

def cross_entropy(label, predict):
    """ 0 """
    ce_loss = -1. * (( label*torch.log(predict) + (1. - label)*torch.log(1. - predict) ))
    #ce_loss = -1. * ( label*torch.log(predict) )
    return ce_loss

def domain_c_loss(args, sim_real, y_real, sim_fake, y_fake, device="cuda"):
    """ make labels """
    if args.multi_hot: #y = idx 
        if args.use_base and args.zero_cut:
            return torch.mean(  cross_entropy(y_real, sim_fake) )
        else:
            batch_idx = [ [i] for i in range(sim_real.shape[0]) ]
            real_label = torch.zeros_like(sim_real).to(device)
            real_label[batch_idx, y_real] = 1.
            fake_label = torch.zeros_like(sim_fake).to(device)
            fake_label[batch_idx, y_fake] = 1.
            return torch.mean(  cross_entropy(real_label, sim_fake) )

def compute_p_loss(args, sim_real, y_real, sim_fake, y_fake, device="cuda"):
    if args.step2:
        """ when do promptleanring """
        if args.multi_hot: #y = idx 
            if  args.use_base and args.zero_cut:
                loss = torch.mean( cross_entropy(y_fake, sim_real) ) + torch.mean( cross_entropy(y_real, sim_fake) )
            else:
                batch_idx = [ [i] for i in range(sim_real.shape[0]) ]
                real_label = torch.zeros_like(sim_real).to(device)
                real_label[batch_idx, y_real] = 1.
                fake_label = torch.zeros_like(sim_fake).to(device)
                fake_label[batch_idx, y_fake] = 1.
                loss = torch.mean( cross_entropy(fake_label, sim_real) ) + torch.mean( cross_entropy(real_label, sim_fake) ) 
        return loss

def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, device="cuda"):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1) 
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg) #(8,16)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)

        x_fake = nets.generator(x_real, s_trg)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0) 

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())

def get_label_from_sim_one(args, sim_fake, sim_fake_base, device="cuda"):
    """
    : get labels from clip similarity of one image.
    abc
    tensor([[1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]]

    # topk
        torch.topk(abc, k=3, dim=-1)
        torch.return_types.topk(
        values=tensor([[3., 2., 1.],
                [6., 5., 4.],
                [9., 8., 7.]]),
        indices=tensor([[2, 1, 0],
                [2, 1, 0],
                [2, 1, 0]]))

    """
    if args.use_base:
        if args.multi_hot:
            """ 0 or sim"""
            zero = torch.FloatTensor([0.]).to(device)
            fake_flag = torch.where( (sim_fake-sim_fake_base)<=zero, zero, sim_fake )

            """ topk indexing """
            _, y_fake = torch.topk(fake_flag, args.topk, dim=-1) # topk

            if args.zero_cut:
                """ 0 or 1: one-hot encoding """
                y_fake_fin = torch.zeros(sim_fake.shape)

                for i in range(y_fake_fin.size(0)):
                    # batch
                    for k in range(y_fake.size(1)):
                        # num_topk
                        fake_true = y_fake[i][k] # topk index
                        if sim_fake[i][fake_true] != 0.:
                            y_fake_fin[i][fake_true] = 1.
                y_fake = y_fake_fin 

    else:
        if args.multi_hot:
            _, y_fake = torch.topk(sim_fake, args.topk, dim=-1)

    y_fake = y_fake.to(device)
    return y_fake

def get_label_from_sim(args, prompt_idx, sim_val_src, sim_val_ref, sim_val_src_base=None, sim_val_ref_base=None, device="cuda"):
    """
    get_label_from_sim 함수를 src, ref input에 대하여 각각 한번씩 실행후 반환해주는 함수.
    """
    y_val_src = get_label_from_sim_one(args, sim_val_src, sim_val_src_base, device=device) 
    y_val_ref = get_label_from_sim_one(args, sim_val_ref, sim_val_ref_base, device=device)
    return y_val_src, y_val_ref

#   get_unsup_labels(args, nets, x_real,         x_ref,          clip_model, prompt, prompt_idx, base_template, self.device, norm=norm)
def get_unsup_labels(args, nets, inputs_val_src, inputs_val_ref, clip_model, prompt, prompt_idx, base_template, device, input_fake=None, detach=True):
    if input_fake is not None :
        """ 생성된 이미지에 대해서만 similarity, label 값을 계산함 """
        """ base는 prompt learning 안된 원래 clip model에서 feature를 얻어내려고함. """
        """ 생성된 이미지로는 prompt learning하지 않음, detach=True """
        sim_fake = cal_clip_loss(args, nets, input_fake, clip_model, prompt, detach=detach, device=device)
        sim_fake_base = get_sim_from_clip(args, input_fake, clip_model, base_template, device=device) #cal_clip_loss(args, nets, input_fake, clip_model, base_template, device=device)

        y_fake = get_label_from_sim_one(args, sim_fake, sim_fake_base, device=device) 

        sim_fake = sim_fake.to(device)
        y_fake = y_fake.to(device)
        return sim_fake, y_fake

    else:
        """ supversied가 아니면. """
        """ 생성 쪽에서는 prompt쪽 update에 영향을 끼치지 않기위해 detach 진행 """
        sim_val_src = cal_clip_loss(args, nets, inputs_val_src, clip_model, prompt, detach=detach, device=device)
        sim_val_ref = cal_clip_loss(args, nets, inputs_val_ref, clip_model, prompt, detach=detach,device=device)
        sim_val_src_base = None
        sim_val_ref_base = None

        if args.use_base:
            """ base는 clip으로 부터 추출해 학습 안되게함 """
            #get_sim_from_clip(args, x, clip_model, prompt, device="cuda"):
            sim_val_src_base = get_sim_from_clip(args, inputs_val_src, clip_model, base_template, device=device)#cal_clip_loss(args, nets, inputs_val_src, clip_model, base_template, device=device)
            sim_val_ref_base = get_sim_from_clip(args, inputs_val_ref, clip_model, base_template, device=device)#cal_clip_loss(args, nets, inputs_val_src, clip_model, base_template, device=device)

        # default: norm=None
        y_val_src, y_val_ref = get_label_from_sim(args, prompt_idx, sim_val_src, sim_val_ref, sim_val_src_base, sim_val_ref_base)

        return sim_val_src, y_val_src, sim_val_ref, y_val_ref
        
def compute_g_unsup_loss(nets, args, clip_model, prompt, base_template, prompt_idx, x_real, y_org, y_trg, x_refs= None, sim_ref=None, z_trgs=None, device="cuda"):
    loss_adv = torch.tensor([0.]).to(device) 
    loss_recon = torch.tensor([0.]).to(device)
    loss_sty = torch.tensor([0.]).to(device)
    loss_cnt = torch.tensor([0.]).to(device)
    loss_cyc = torch.tensor([0.]).to(device)
    loss_dc = torch.tensor([0.]).to(device)
    loss_ds = torch.tensor([0.]).to(device)
    loss_text_ds = torch.tensor([0.]).to(device)
    loss_p = torch.tensor([0.]).to(device)

    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    x_fake = nets.generator(x_real, s_trg)
    #x_fake_detach = nets.generator(x_real, s_trg.detach())

    out = nets.discriminator(x_fake, y_trg)
    loss_adv += adv_loss(out, 1)

    #                                               dummy   dummy
    sim_fake, y_fake = get_unsup_labels(args, nets, x_fake, x_fake, clip_model, prompt, prompt_idx, base_template, device, input_fake=x_fake)
    #sim_fake_detach, y_fake_detach = get_unsup_labels(args, nets, x_fake_detach, x_fake_detach, clip_model, prompt, prompt_idx, base_template, device, input_fake=x_fake_detach)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty += torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive(ds) loss
    if args.ds:
        if z_trgs is not None:
            s_trg2 = nets.mapping_network(z_trg2, y_trg)

            x_fake2 = nets.generator(x_real, s_trg2)
            x_fake2 = x_fake2.detach()
            loss_ds += torch.mean(torch.abs(x_fake - x_fake2))

    # clip_similarity loss
    if args.dc and not args.step2:
        loss_dc += 2*domain_c_loss(args, sim_ref, y_trg, sim_fake, y_fake, device)

    if args.use_prompt and args.step2 and z_trgs is None:
        p_loss = compute_p_loss(args, sim_ref, y_trg, sim_fake, y_fake, device=device)
        loss_p += p_loss

    s_org = nets.style_encoder(x_real, y_org)

    #cycle_consistency
    if args.cycle:
        x_cyc = nets.generator(x_fake, s_org)
        loss_cyc += torch.mean(torch.abs(x_cyc - x_real))  

    """ total loss summation 과정. """
    loss = loss_adv + args.lambda_sty*loss_sty

    if args.ds:
        """ diversity loss 는 minus 로 """
        loss -= args.lambda_ds * loss_ds
    if args.cycle:
        loss += args.lambda_cyc * loss_cyc
    if args.dc:
        # domain consistency loss
        loss += args.lambda_dc * loss_dc
    
    if args.use_prompt and args.step2:
        loss += loss_p
            
    return loss, Munch(adv=loss_adv.item(),
                            cnt=loss_cnt.item(),
                            sty=loss_sty.item(),
                            cyc=loss_cyc.item(),
                            dc = loss_dc.item(),
                            ds=loss_ds.item(),
                            recon = loss_recon.item(),
                            prompt = loss_p.item())

def moving_average(args, model, model_test, beta=0.999):
    # for param, param_test in zip(model.parameters(), model_test.parameters()):
    #     param_test.data = torch.lerp(param.data.detach().cpu(), param_test.data, beta)
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)

def adv_loss(logits, target):
    #  logits.shape = (batch, num_topk) 
    assert target in [1, 0]
    logits = logits.view(-1) # (batch*num_topk)
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

def get_sim_from_clip(args, x, clip_model, prompt, device="cuda"):
    """ input denormalize """
    x = (x+1.)/2.
    """ resize to 224 """
    x = F.interpolate(x, size=224, mode='bicubic', align_corners=True)
    """ clip noralization """
    x = clip_normalize(x, device)

    image_features = clip_model.encode_image( x )#clip_normalize(x, args.device))
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    tokens = clip.tokenize(prompt).to(device)
    text_feature = clip_model.encode_text(tokens).detach()
    text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

    return image_features @ text_feature.t()

def cal_clip_loss(args, nets, x, clip_model, prompt, detach=False, device="cuda"):
    """ input denormalize """
    x = (x+1.)/2.
    """ resize to 224 """
    x = F.interpolate(x, size=224, mode='bicubic', align_corners=True)
    """ clip noralization """
    x = clip_normalize(x, device)

    image_features = clip_model.encode_image( x )#clip_normalize(x, args.device))
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    if args.use_prompt:
        """ use promptLearner """
        text_feature = nets.promptLearner(clip_model)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        if detach:
            """ detach not to propagat to the text encoder. """
            text_feature = text_feature.detach()
    else:
        tokens = clip.tokenize(prompt).to(device)
        text_feature = clip_model.encode_text(tokens).detach()
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

    return image_features @ text_feature.t()