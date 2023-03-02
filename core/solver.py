import os
import sys
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
from torch.utils.tensorboard import SummaryWriter

from core.utils import *
from core.model import build_model
from core.checkpoint import CheckpointIO
import core.utils as utils
from core.utils import clip_normalize
from metrics.eval import calculate_metrics
from metrics.test_fid import main_fid
import lpips

import clip
from template import * 
from copy import deepcopy

class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda') 

        self.template, self.prompt, self.prompt_idx, args.num_domains, self.base_template = self._set_attr_prompt(self.args)
        self.lpips_loss = lpips.LPIPS().eval().requires_grad_(False).to(self.device)

        if self.args.text_aug:
            self.template, self.base_templates = get_templates(args.dataset)
        else:
            self.prompt = [self.template.format(x) for x in self.prompt]

        """ network definition """
        self.nets, self.nets_ema = build_model(self.args)
            
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        log_dir = os.path.join(args.checkpoint_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.summary_writer = SummaryWriter(log_dir=log_dir)

        if self.args.mode == 'train':
            self.optims = Munch()
            if args.step2 and args.use_scheduler:
                self.scheduler = Munch()

            for net in self.nets.keys():
                """ get learning rate """
                if net == 'fan':
                    continue
                if net=='mapping_network':
                    lr = self.args.m_lr
                elif "promptLearner" in net:
                    lr = self.args.p_lr
                else:
                    lr = self.args.lr

                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr = lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay= args.weight_decay)

                if net=="promptLearner" and self.args.use_scheduler:
                    self.scheduler[net] = lr_scheduler.StepLR(self.optims[net], step_size=4000, gamma=0.5)


            if args.step2:
                self.ckptios = [
                        CheckpointIO(ospj(args.checkpoint_dir, args.name, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
                        CheckpointIO(ospj(args.checkpoint_dir, args.name, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema),
                        CheckpointIO(ospj(args.checkpoint_dir, args.name, '{:06d}_optims.ckpt'), **self.optims),
                    ]
                if self.args.use_scheduler:
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

        """ models to the GPU """
        self.to(self.device)

        """ network initialization. """
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            print(name)
            if ('ema' not in name) and ('fan' not in name) and ('prompt' not in name)  and ('Prompt' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

        """ after initialization, load pretrained clip """
        self.clip_model, preprocess = clip.load('ViT-B/32', device="cpu", jit=False)
        self.clip_model = self.clip_model.to(self.device)
        """ freeze the network parameters """
        for clip_param in self.clip_model.parameters():
            clip_param.requires_grad = False

    def _save_checkpoint(self, step):
        """ save checkpoint """
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        """ load checkpoints """
        for ckptio in self.ckptios:
            ckptio.load(self.args, step)

    def _reset_grad(self):
        """ zero initialization gradient """
        for optim in self.optims.values():
            optim.zero_grad()

    def _set_attr_prompt(self, args):
        """ define template, prompt, prompt_idx. class initialization. """
        template, prompt, prompt_idx, base_template = get_prompt_and_att(args)
        num_domains = len(prompt)
        return template, prompt, prompt_idx, num_domains, base_template

    def get_sim_mask(self, args, x_real, x_ref, text_feat, base_text_feat):
        """ get similarity between image and domain prompts."""
        sim_val_src = cal_clip_loss(args, self.nets, x_real, self.clip_model, self.prompt, text_feat, base_text_feat, device=self.device) # (batch, num_prompt)
        sim_val_ref = cal_clip_loss(args, self.nets, x_ref, self.clip_model, self.prompt, text_feat, base_text_feat, device=self.device)  # (batch, num_prompt)
        
        """ get similarity between image and base prompt. """
        sim_val_src_base = cal_clip_loss(args, self.nets, x_real, self.clip_model, self.base_template, text_feat, base_text_feat, is_prefix=True, device=self.device) # (batch, 1)
        sim_val_ref_base = cal_clip_loss(args, self.nets, x_ref, self.clip_model, self.base_template, text_feat, base_text_feat, is_prefix=True, device=self.device)  # (batch, 1)

        """ To filter data that all the similarity values are lower than zero """
        src_mask = torch.unique( torch.where( (sim_val_src-sim_val_src_base)>0)[0] )
        ref_mask = torch.unique( torch.where( (sim_val_ref-sim_val_ref_base)>0)[0] ) 

        return sim_val_src, sim_val_ref, sim_val_src_base , sim_val_ref_base, src_mask, ref_mask

    def get_data_iter(self, args, loaders, text_feat, base_text_feat):
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
        x_src = x_src.to(self.device)
        x_ref = x_ref.to(self.device)
        x_ref_2 = x_ref_2.to(self.device)

        b = args.batch_size
        if args.use_base:
            sim_val_src, sim_val_ref, sim_val_src_base , sim_val_ref_base, src_mask, ref_mask = self.get_sim_mask(args, x_src, x_ref, text_feat, base_text_feat)

            """ To do. hard coded"""
            while (len(ref_mask) < b) or (len(src_mask) < b) :
                try:
                    x_src, real_y_src = next(self.train_src)
                    x_ref, x_ref_2, real_y_ref = next(self.train_ref)
                except:
                    self.train_src = iter(loaders.src)
                    self.train_ref = iter(loaders.ref)

                    x_src, real_y_src = next(self.train_src)
                    x_ref, x_ref_2, real_y_ref = next(self.train_ref)

                x_src = x_src.to(self.device)
                x_ref = x_ref.to(self.device)
                sim_val_src, sim_val_ref, sim_val_src_base , sim_val_ref_base, src_mask, ref_mask = self.get_sim_mask(args, x_src, x_ref, text_feat, base_text_feat)

            return x_src[:b], x_ref[:b], x_ref_2[:b], z_trg[:b], z_trg_2[:b], real_y_src[:b], real_y_ref[:b], sim_val_src[:b], sim_val_ref[:b], sim_val_src_base[:b] , sim_val_ref_base[:b]
        else:
            return x_src[:b], x_ref[:b], x_ref_2[:b], z_trg[:b], z_trg_2[:b], real_y_src[:b], real_y_ref[:b]                 

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims
        prompt = self.prompt
        base_template =self.base_template
        prompt_idx = self.prompt_idx
        clip_model = self.clip_model
        lpips_loss = self.lpips_loss

        if args.step2 and args.use_scheduler:
            scheduler = self.scheduler

        """ data loader """
        self.train_src = iter(loaders.src)
        self.train_ref = iter(loaders.ref)
        self.val_src = iter(loaders.val_src)
        self.val_ref = iter(loaders.val_ref)

        """ fixed base text feat """
        if args.base_fix:
            base_text_feat = nets.promptLearner.module.init_base_mean_embed(clip_model, device="cuda").clone().detach()

        """ load checkpoint """
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)
        
        """ scheduling a weight of style diversification loss """
        initial_lambda_ds = args.lambda_ds

        """ start training """
        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):

            """ get text and base-text feature"""
            text_feat      = nets.promptLearner.module.init_mean_embed(clip_model, device="cuda")
            if not args.base_fix:
                base_text_feat = nets.promptLearner.module.init_base_mean_embed(clip_model, device="cuda")

            if args.use_base:
                x_real, x_ref, x_ref_2, z_trg, z_trg_2, real_y_src, real_y_ref, sim_real, sim_ref, sim_base_real, sim_base_ref = self.get_data_iter(args, loaders, text_feat, base_text_feat)
                y_org, y_trg = get_label_from_sim(args, prompt_idx, sim_real, sim_ref, sim_base_real, sim_base_ref)
            else:
                x_real, x_ref, x_ref_2, z_trg, z_trg_2, real_y_src, real_y_ref  = self.get_data_iter(args, loaders, text_feat, base_text_feat)
                sim_real, y_org, sim_ref, y_trg = get_unsup_labels(args, nets, x_real, x_ref, clip_model, prompt, prompt_idx, base_template, text_feat, base_text_feat, self.device, detach=False)

            if (0. in torch.sum(y_trg, dim=-1)) or (0. in torch.sum(1-y_trg, dim=-1)):
                continue
            if (0. in torch.sum(y_org, dim=-1)) or (0. in torch.sum(1-y_org, dim=-1)):
                continue
            
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
            g_loss, g_losses_latent = compute_g_unsup_loss(nets, lpips_loss, args, clip_model, prompt, base_template, text_feat, base_text_feat, prompt_idx, x_real, y_org, y_trg, sim_ref = sim_ref, z_trgs=[z_trg, z_trg_2], device=self.device)
            
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()
            if args.step2:
                optims.promptLearner.step()
                if args.use_scheduler:
                    scheduler.promptLearner.step()
                
            """ Generator step2. Update only Generator """
            if args.step2:
                sim_real, y_org, sim_ref, y_trg = get_unsup_labels(args, nets, x_real, x_ref, clip_model, prompt, prompt_idx, base_template, text_feat, base_text_feat, self.device, detach=True)
                if (0. in torch.sum(y_trg, dim=-1)) or (0. in torch.sum(y_org, dim=-1)):
                    non_zero_idx = (torch.sum(y_trg, dim=-1) != 0) and (torch.sum(y_org, dim=-1) != 0)
                    x_real  = x_real[non_zero_idx]
                    y_org   = y_org[non_zero_idx]
                    y_trg   = y_trg[non_zero_idx]
                    sim_ref = sim_ref[non_zero_idx]
                    x_ref   = x_ref[non_zero_idx]
                    x_ref_2 = x_ref_2[non_zero_idx]

            g_loss, g_losses_ref = compute_g_unsup_loss(nets, lpips_loss, args, clip_model, prompt, base_template, text_feat, base_text_feat, prompt_idx, x_real, y_org, y_trg, sim_ref = sim_ref, x_refs=[ x_ref , x_ref_2 ], device=self.device)

            self._reset_grad()
            g_loss.backward()
            optims.generator.step()

    
            """ compute moving average of network parameters """
            moving_average(args, nets.generator, nets_ema.generator, beta=0.999)
            moving_average(args, nets.style_encoder, nets_ema.style_encoder, beta=0.999)
            moving_average(args, nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            if args.step2 or args.text_aug:
                moving_average(args, nets.promptLearner, nets_ema.promptLearner, beta=0.999)
                
            """ weight decaying of the style diversity sensitive loss """
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            """ plotting current losses on command window """
            if (i+1) % args.print_every == 0:
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
                    all_losses["P/map"] = g_losses_latent["prompt"]
                    all_losses["P/ref"] = g_losses_ref["prompt"]

                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

                for key, value in all_losses.items():
                    self.summary_writer.add_scalar( '{}'.format(key), value , i)

            """ save checkpoint """
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)
            
            """ sampling in evry args.sample_every """
            with torch.no_grad():
                if (i+1) % args.sample_every == 0:
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

                    sim_val_src, y_val_src, sim_val_ref, y_val_ref = get_unsup_labels(args, nets, inputs_val_src, inputs_val_ref, clip_model, prompt, prompt_idx, base_template, text_feat, base_text_feat, self.device)
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

        text_feat      = nets_ema.promptLearner.module.init_mean_embed(clip_model, device="cuda")
        base_text_feat = nets_ema.promptLearner.module.init_base_mean_embed(clip_model, device="cuda")

        os.makedirs(ospj(args.result_dir, args.name), exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        train_src = iter(loaders.src)
        train_ref = iter(loaders.ref)

        src,_ = next(train_src) 
        ref,_, _ = next(train_ref) 

        sim_real, y_org, sim_ref, y_ref = get_unsup_labels(args, nets_ema, src.to(self.device), ref.to(self.device), clip_model, prompt, prompt_idx, self.base_template, text_feat, base_text_feat, self.device)

        if args.infer_mode == 'reference':
            fname = ospj(args.result_dir, args.name, 'reference.jpg')
            print('Working on {}...'.format(fname))
            utils.translate_using_reference(nets_ema, args, src.to(self.device), ref.to(self.device), y_ref.to(self.device), fname)

        else:
            fname = ospj(args.result_dir, args.name, 'latent.jpg')
            print('Working on {}...'.format(fname))
            utils.translate_using_latent(nets_ema, args, src.to(self.device), ref.to(self.device), y_ref.to(self.device), fname)

    @torch.no_grad()
    def evaluate(self):
        prompt = self.prompt
        args   = self.args
        nets_ema = self.nets_ema
        infer_mode = args.infer_mode
        self._load_checkpoint(resume_iter)
        calculate_metrics(nets_ema, args, self.prompt_idx, self.prompt, self.base_template, resume_iter, mode=infer_mode)

    @torch.no_grad()
    def FID(self):
        args = self.args
        main_fid(args, self.prompt_idx, mode = 'reference')


def cross_entropy(label, predict):
    """ 0 """
    ce_loss = -1. *(label)*torch.log(F.relu(predict)+ sys.float_info.epsilon)/ torch.sum(label, dim=-1, keepdim=True)\
               + -1.* (1. - label)*torch.log(F.relu(1-predict)+ sys.float_info.epsilon)/ torch.sum(1-label, dim=-1, keepdim=True)

    ce_loss = torch.mean( torch.sum(ce_loss, dim=-1 ) )
    return ce_loss

def domain_c_loss(args, y_real, sim_fake, device="cuda"):
    """ make labels """
    return cross_entropy(y_real, sim_fake)

def compute_p_loss(args, y_real, sim_fake_txt_prop, sim_fake_img_prop, device="cuda"):
    """ when do promptleanring """
    loss = cross_entropy(y_real, sim_fake_txt_prop) + cross_entropy(y_real, sim_fake_img_prop)

    return loss

def gen_ablated_sample(args, nets, z_trg, x_fake, y_trg, detach=True):
    label  = y_trg.clone().detach()
    # at first
    one_count = torch.sum(label, dim=-1)
    
    # at second
    num_zero_list = []
    for i in range(label.shape[0]):
        num_zero_list.append( 1 ) 
    
    # at third
    fir_dim, sec_dim = torch.where(label==1)
    fir_dim_list, sec_dim_list = [], []
    for i in range(label.shape[0]):
        fir_dim_list.extend([i]*num_zero_list[i])
        sec_dim_list.extend( np.random.choice( sec_dim[fir_dim==i].tolist(), num_zero_list[i], replace=False) )

    """ label to 0 """
    label[fir_dim_list, sec_dim_list] = 0

    # sanity check
    for i in range(label.shape[0]):
        if torch.sum(label[i]) == 0.:
            label[i][ np.random.choice( np.arange(args.num_domains), 1, replace=False) ] = 1

    if detach:
        s_trg_del = nets.mapping_network(z_trg, y_trg).detach()
        x_cyc = nets.generator(x_fake, s_trg_del).detach()
    else:
        s_trg_del = nets.mapping_network(z_trg, y_trg)
        x_cyc = nets.generator(x_fake, s_trg_del)

    return x_cyc, sec_dim_list

def cross_entropy_oneside(label, predict, val=1):
    """ 0 """
    if val==1:
        ce_loss = -1. *(label)*torch.log(F.relu(predict)+ sys.float_info.epsilon)/ torch.sum(label, dim=-1, keepdim=True)
        if  0. in torch.sum(label, dim=-1, keepdim=True) < label.shape[0]:
            import pdb; pdb.set_trace()
    elif val==0:
        ce_loss = -1.* (1. - label)*torch.log(F.relu(1-predict)+ sys.float_info.epsilon)/ torch.sum(1-label, dim=-1, keepdim=True)
        if torch.sum(1-label, dim=-1, keepdim=True) < label.shape[0]:
            import pdb; pdb.set_trace()

    ce_loss = torch.mean( torch.sum(ce_loss, dim=-1 ) )
    return ce_loss

def compute_p_loss_cycle(args, y_trg, sim_fake, sim_ablated, ablated_idx, device="cuda"): #y_fake_img = img.detach , y_fake_txt = txt.detach
    N, C = y_trg.shape
    loss = cross_entropy_oneside( y_trg[torch.arange(N), ablated_idx], sim_fake[torch.arange(N), ablated_idx], val=1)\
             + cross_entropy_oneside( (1-y_trg)[torch.arange(N), ablated_idx], sim_ablated[torch.arange(N), ablated_idx], val=0)
    return loss

def compute_p_reg(args, y_real, sim_fake_txt_prop, sim_fake_img_prop, sim_fake_txt_prop_base, sim_fake_img_prop_base, device="cuda"):

    g_backprop = 4+ (sim_fake_img_prop_base.detach()*y_real - sim_fake_img_prop*y_real)\
                 + (sim_fake_img_prop*(1-y_real)-sim_fake_img_prop_base.detach()*(1-y_real))
    t_backprop = 4+(sim_fake_txt_prop_base.detach()*y_real - sim_fake_txt_prop*y_real)\
                 + (sim_fake_txt_prop*(1-y_real)-sim_fake_txt_prop_base.detach()*(1-y_real))
    
    return torch.mean(g_backprop + t_backprop) * 1/2

def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, device="cuda"):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1) 
    loss_reg = r1_reg(out, x_real)

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

    if args.use_base:
        if args.multi_hot:
            """ 0 or sim"""
            zero = torch.FloatTensor([0.]).to(device)
            y_fake = torch.where( (sim_fake-sim_fake_base)<=zero, zero,  torch.tensor(1.).to(device) )

    else: #use_topk
        if args.multi_hot:
            _, y_fake = torch.topk(sim_fake, args.topk, dim=-1)
    y_fake = y_fake.to(device)
    return y_fake

def get_label_from_sim(args, prompt_idx, sim_val_src, sim_val_ref, sim_val_src_base=None, sim_val_ref_base=None, device="cuda"):
    y_val_src = get_label_from_sim_one(args, sim_val_src, sim_val_src_base, device=device) 
    y_val_ref = get_label_from_sim_one(args, sim_val_ref, sim_val_ref_base, device=device)
    return y_val_src, y_val_ref

def get_unsup_labels(args, nets, inputs_val_src, inputs_val_ref, clip_model, prompt, prompt_idx, base_template, text_feat, base_text_feat,  device, input_fake=None, return_base=False, detach=True):
    """ At this function, default   !!detach==True!! """
    if input_fake is not None :
        sim_fake = cal_clip_loss(args, nets, input_fake, clip_model, prompt, text_feat, base_text_feat,  detach=detach, device=device)

        sim_fake_base = None
        if args.use_base:
            sim_fake_base = cal_clip_loss(args, nets, input_fake, clip_model, base_template, text_feat, base_text_feat,  detach=detach, is_prefix=True, device=device)

        y_fake = get_label_from_sim_one(args, sim_fake, sim_fake_base, device=device) 

        sim_fake = sim_fake.to(device)
        y_fake = y_fake.to(device)

        if args.use_base and return_base:
            return sim_fake, y_fake, sim_fake_base
        else:
            return sim_fake, y_fake

    else:
        sim_val_src = cal_clip_loss(args, nets, inputs_val_src, clip_model, prompt, text_feat, base_text_feat,  detach=detach, device=device)
        sim_val_ref = cal_clip_loss(args, nets, inputs_val_ref, clip_model, prompt, text_feat, base_text_feat,  detach=detach,device=device)
        sim_val_src_base = None
        sim_val_ref_base = None

        if args.use_base:
            """ base_prompt is not learnable parameters """
            sim_val_src_base = cal_clip_loss(args, nets, inputs_val_src, clip_model, base_template, text_feat, base_text_feat,  detach=detach, is_prefix=True, device=device)
            sim_val_ref_base = cal_clip_loss(args, nets, inputs_val_ref, clip_model, base_template, text_feat, base_text_feat,  detach=detach, is_prefix=True, device=device)

        y_val_src, y_val_ref = get_label_from_sim(args, prompt_idx, sim_val_src, sim_val_ref, sim_val_src_base, sim_val_ref_base)
        
        if args.use_base and return_base:
            return sim_val_src, y_val_src, sim_val_ref, y_val_ref, sim_val_src_base, sim_val_ref_base
        else:
            return sim_val_src, y_val_src, sim_val_ref, y_val_ref
        
def compute_g_unsup_loss(nets, lpips_loss, args, clip_model, prompt, base_template, text_feat, base_text_feat, prompt_idx, x_real, y_org, y_trg, x_refs= None, sim_ref=None, z_trgs=None, is_aug=False, device="cuda"):
    loss_adv = torch.tensor([0.]).to(device) 
    loss_recon_lpips = torch.tensor([0.]).to(device)
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
    out = nets.discriminator(x_fake, y_trg)
    loss_adv += adv_loss(out, 1)

    """
    grad:
    sim_ref, sim_fake, sim_fake_base: generator backprop.
    sim_fake_detach, sim_fake_base_detach: prompt backprop.
    """
    # G, S/M: backprop, Text:detach
    sim_fake, y_fake, sim_fake_base = get_unsup_labels(args, nets, x_fake, x_fake, clip_model, prompt, prompt_idx, base_template, text_feat, base_text_feat,  device, return_base=True, input_fake=x_fake)
    if args.step2:
        # G, S/M: detach, Text:backprop
        x_fake_detach = nets.generator(x_real, s_trg).detach() 
        sim_fake_detach, y_fake_detach, sim_fake_base_detach = get_unsup_labels(args, nets, x_fake_detach, x_fake_detach, clip_model, prompt, prompt_idx, base_template, text_feat, base_text_feat,  device, return_base=True, input_fake=x_fake_detach, detach=False)
    
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
    if args.dc and not args.dcycle:
        loss_dc += domain_c_loss(args, y_trg, sim_fake,  device)

    if (args.dcycle and z_trgs is not None) or (args.step2 and z_trgs is not None) : 
        N, C = y_trg.shape
        """ when do promptleanring """
        if args.t_update:
            """ update only t """
            # g don't upate
            x_ablated, ablated_idx = gen_ablated_sample(args, nets, z_trg, x_fake_detach, y_trg, detach=True) # type(ablated_idx) = list
            #Text:backprop
            sim_ablated, y_ablated = get_unsup_labels(args, nets, x_ablated, x_ablated, clip_model, prompt, prompt_idx, base_template, text_feat, base_text_feat, device, input_fake=x_ablated, detach=False)
            # g don't upate
            loss_dcycle = compute_p_loss_cycle(args, y_trg, sim_fake_detach, sim_ablated, ablated_idx, device=device)
            cycle_reg = torch.mean( torch.abs(sim_fake_detach[torch.arange(N), ablated_idx] - sim_ablated[torch.arange(N), ablated_idx] ))

            loss_dcycle = loss_dcycle - cycle_reg*args.lambda_dc_reg
        elif args.gt_update:
            """ update g and t simultaneously """
            # g update
            x_ablated, ablated_idx = gen_ablated_sample(args, nets, z_trg, x_fake, y_trg, detach=False) # type(ablated_idx) = list
            #Text:backprop
            sim_ablated, y_ablated = get_unsup_labels(args, nets, x_ablated, x_ablated, clip_model, prompt, prompt_idx, base_template, text_feat, base_text_feat, device, input_fake=x_ablated, detach=False)
            # g update
            loss_dcycle = compute_p_loss_cycle(args, y_trg, sim_fake, sim_ablated, ablated_idx, device=device)
            cycle_reg = torch.mean( torch.abs(sim_fake[torch.arange(N), ablated_idx] - sim_ablated[torch.arange(N), ablated_idx] ))

            loss_dcycle = loss_dcycle - cycle_reg*args.lambda_dc_reg
        elif args.alter_update:
            """ args.g_update not t """
            # g update
            x_ablated, ablated_idx = gen_ablated_sample(args, nets, z_trg, x_fake, y_trg, detach=False) # type(ablated_idx) = list
            #Text:do not backprop
            sim_ablated, y_ablated = get_unsup_labels(args, nets, x_ablated, x_ablated, clip_model, prompt, prompt_idx, base_template, text_feat, base_text_feat, device, input_fake=x_ablated, detach=True)
            # g update
            loss_dcycle = compute_p_loss_cycle(args, y_trg, sim_fake, sim_ablated, ablated_idx, device=device)
            cycle_reg = torch.mean( torch.abs(sim_fake[torch.arange(N), ablated_idx] - sim_ablated[torch.arange(N), ablated_idx] ))

            loss_dcycle = loss_dcycle - cycle_reg*args.lambda_dc_reg
            """ args.t_update not g"""
            # g don't upate
            x_ablated, ablated_idx = gen_ablated_sample(args, nets, z_trg, x_fake_detach, y_trg, detach=True) # type(ablated_idx) = list
            #Text:backprop
            sim_ablated, y_ablated = get_unsup_labels(args, nets, x_ablated, x_ablated, clip_model, prompt, prompt_idx, base_template, text_feat, base_text_feat, device, input_fake=x_ablated, detach=False)
            # g don't upate
            loss_dcycle += compute_p_loss_cycle(args, y_trg, sim_fake_detach, sim_ablated, ablated_idx, device=device) 
            cycle_reg = torch.mean( torch.abs(sim_fake_detach[torch.arange(N), ablated_idx] - sim_ablated[torch.arange(N), ablated_idx] ))
            loss_dcycle = loss_dcycle - cycle_reg*args.lambda_dc_reg

            loss_dcycle = 1/2 * loss_dcycle
        else:
            """ step1 """
            # g update
            x_ablated, ablated_idx = gen_ablated_sample(args, nets, z_trg, x_fake, y_trg, detach=False) # type(ablated_idx) = list
            #Text:do not backprop
            sim_ablated, y_ablated = get_unsup_labels(args, nets, x_ablated, x_ablated, clip_model, prompt, prompt_idx, base_template, text_feat, base_text_feat, device, input_fake=x_ablated, detach=True)
            # g update
            loss_dcycle = compute_p_loss_cycle(args, y_trg, sim_fake, sim_ablated, ablated_idx, device=device)

        if args.step2:
            loss_p += loss_dcycle
        else:
            loss_dc += loss_dcycle


    #cycle_consistency
    s_org = nets.style_encoder(x_real, y_org)
    if args.cycle:
        x_cyc = nets.generator(x_fake, s_org)
        loss_cyc += torch.mean(torch.abs(x_cyc - x_real))  

    if args.recon_lpips:
        s_org_recon = s_org.detach()
        x_recon = nets.generator(x_real, s_org_recon)
        loss_recon_lpips += lpips_loss(x_recon, x_real).mean()


    """ total loss summation"""
    loss = loss_adv + args.lambda_sty*loss_sty

    if args.ds:
        """ diversity loss """
        loss -= args.lambda_ds * loss_ds
    if args.cycle:
        loss += args.lambda_cyc * loss_cyc
    if args.dc:
        # domain consistency loss
        loss += args.lambda_dc * loss_dc
    if args.recon_lpips:
        loss += args.lambda_lpips * loss_recon_lpips
    
    if args.step2:
        loss += args.lambda_p * loss_p
            
    return loss, Munch(adv=loss_adv.item(),
                            cnt=loss_cnt.item(),
                            sty=loss_sty.item(),
                            cyc=loss_cyc.item(),
                            dc = loss_dc.item(),
                            ds=loss_ds.item(),
                            recon = loss_recon_lpips.item(),
                            prompt = loss_p.item())

def moving_average(args, model, model_test, beta=0.999):

    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)

def adv_loss(logits, target):
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

    image_features = clip_model.encode_image( x )
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    tokens = clip.tokenize(prompt).to(device)
    text_feature = clip_model.encode_text(tokens).detach()
    text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

    return image_features @ text_feature.t()

def cal_clip_loss(args, nets, x, clip_model, prompt, text_feat, base_text_feat,  detach=False, is_prefix=False, device="cuda"):
    """ input denormalize """
    x = (x+1.)/2.
    """ resize to 224 """
    x = F.interpolate(x, size=224, mode='bicubic', align_corners=True)
    """ clip noralization """
    x = clip_normalize(x, device)

    image_features = clip_model.encode_image( x )#clip_normalize(x, args.device))
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    if args.step2 or args.text_aug:
        """ use promptLearner """
        if not is_prefix:
            text_feature = text_feat.squeeze()#[0]
        else:
            text_feature = base_text_feat
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        if detach:
            """ detach not to propagat to the text encoder. """
            text_feature = text_feature.detach()
    else:
        tokens = clip.tokenize(prompt).to(device)
        text_feature = clip_model.encode_text(tokens).detach()
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
    
    return image_features @ text_feature.t()

