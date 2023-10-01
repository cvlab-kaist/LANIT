"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
import json
import glob
import os
from os.path import join as ospj
from shutil import copyfile

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
from scipy.stats import truncnorm

def get_prompt_and_att(args):
    if 'lsun_car' in args.dataset:
        init_prompt = 'a photo of the car about {}.'
        base_template = ["a photo of the car about."]
        all_prompt = ['red car', 'orange car', 'gray car', 'blue car', 'truck', 'white car', 'sports car', 'van', 'sedan','compact car']
        
        if args.num_domains == 4:
            prompt = ['red car', 'truck','van', 'white car']
        elif args.num_domains == 7:
            prompt = ['beagle', 'dandie dinmont terrier', 'golden retriever', 'white fox', 'tabby cat', 'snow leopard', 'bengal tiger']
        elif args.num_domains == 10:
            prompt = ['red car', 'orange car', 'gray car', 'blue car', 'truck', 'white car', 'sports car', 'van', 'sedan','compact car']

    if 'anime' in args.dataset:
        init_prompt = 'A photo of anime with {}.'
        base_template = ["A photo of anime with."]
        all_prompt = ['brown hair', 'red hair', 'black hair', 'purple hair', 'blond hair','blue hair', 'pink hair', 'silver hair', 'green hair', 'white hair']
        
        if args.num_domains == 4:
            prompt = ['brown hair', 'red hair', 'black hair', 'purple hair',]
        elif args.num_domains == 7:
            prompt = ['brown hair', 'red hair', 'black hair', 'purple hair', 'blond hair','blue hair', 'pink hair',]
        elif args.num_domains == 10:
            prompt = ['brown hair', 'red hair', 'black hair', 'purple hair', 'blond hair','blue hair', 'pink hair', 'silver hair', 'green hair', 'white hair']
            
    if 'metface' in args.dataset:
        init_prompt = 'a portrait with {}.'
        base_template = ["a portrait with."]
        all_prompt = ['oil painting', 'grayscale', 'black hair', 'wavy hair', 'male', 'mustache', 'smiling', 'gray hair', 'blonde hair','sculpture']
        
        if args.num_domains == 4:
            prompt = ['red car', 'truck','van', 'white car']
        elif args.num_domains == 7:
            prompt = ['beagle', 'dandie dinmont terrier', 'golden retriever', 'white fox', 'tabby cat', 'snow leopard', 'bengal tiger']
        elif args.num_domains == 10:
            prompt = ['oil painting', 'grayscale', 'black hair', 'wavy hair', 'male', 'mustache', 'smiling', 'gray hair', 'blonde hair','sculpture']

    if 'landscape' in args.dataset:
        init_prompt = 'a photo of the scene about {}.'
        base_template = ["a photo of the scene about."]
        all_prompt = ['mountain', 'field', 'lake', 'ocean', 'waterfall', 'summer', 'winter', 'a sunny day', 'a cloudy day', 'sunset']
        
        if args.num_domains == 4:
            prompt = []
        elif args.num_domains == 7:
            prompt = []
        elif args.num_domains == 10:
            prompt = ['mountain', 'field', 'lake', 'ocean', 'waterfall', 'summer', 'winter', 'a sunny day', 'a cloudy day', 'sunset']
    
    if 'animal' in args.dataset:
        init_prompt = 'a photo of the animalface of {}.'
        base_template = ["a photo of the animalface of."]
        all_prompt = ['beagle', 'dandie dinmont terrier', 'golden retriever', 'malinois', 'appenzeller sennenhund', 'white fox', 'tabby cat', 'snow leopard', 'lion', 'bengal tiger']
            
        if args.num_domains == 4:
            prompt = ['beagle', 'golden retriever','tabby cat', 'bengal tiger']
        elif args.num_domains == 7:
            prompt = ['beagle', 'dandie dinmont terrier', 'golden retriever', 'white fox', 'tabby cat', 'snow leopard', 'bengal tiger']
        elif args.num_domains == 10:
            if args.dict:
                prompt = ['dandie dinmont terrier','malinois','appenzeller sennenhund', 'white fox', 'tabby cat', 'snow leopard', 'lion', 'bengal tiger', 'grey fox', 'german shepherd dog']
            else:
                prompt =  ['beagle', 'dandie dinmont terrier', 'golden retriever', 'malinois',\
                        'appenzeller sennenhund', 'white fox', 'tabby cat', 'snow leopard', 'lion', 'bengal tiger']
        elif args.num_domains == 13:
            prompt =  ['beagle', 'dandie dinmont terrier', 'golden retriever', 'malinois',\
                       'appenzeller sennenhund', 'white fox', 'tabby cat', 'snow leopard', 'lion', 'bengal tiger',\
                       'french bulldog', 'mink', 'maned wolf']
        elif args.num_domains == 16:
            prompt =  ['beagle', 'dandie dinmont terrier', 'golden retriever', 'malinois',\
                       'appenzeller sennenhund', 'white fox', 'tabby cat', 'snow leopard', 'lion', 'bengal tiger',\
                       'french bulldog', 'mink', 'maned wolf', 'monkey', 'toy poodle', 'angora rabbit']

    elif 'food' in args.dataset:
        init_prompt = 'a photo of the food of {}.'
        base_template = ["a photo of the food of."]
        all_prompt = [ "baby back ribs", "beef carpaccio", "beignets", "bibimbap", "caesar salad",\
                            "clam chowder", "Chinese dumplings", "edamame", "bolognese", "strawberry shortcake"]

        if args.num_domains == 4:
            prompt = ['baby back ribs', 'beignets', 'dumplings', 'edamame']
        elif args.num_domains == 7:
            prompt = ['baby back ribs','beef carpaccio','beignets','clam chowder','dumplings', 'edamame', 'strawberry shortcake' ]
        elif args.num_domains == 10:
            prompt = [ "baby back ribs", "beef carpaccio", "beignets", "bibimbap", "caesar salad",\
                                "clam chowder", "dumplings", "edamame", "spaghetti bolognese", "strawberry shortcake"]
        elif args.num_domains == 13:
            prompt = [ "baby back ribs", "beef carpaccio", "beignets", "bibimbap", "caesar salad",\
                                "clam chowder", "dumplings", "edamame", "spaghetti bolognese", "strawberry shortcake",\
                                'apple pie', 'chicken wings', 'ice cream']
        elif args.num_domains == 16:
            prompt = [ "baby back ribs", "beef carpaccio", "beignets", "bibimbap", "caesar salad",\
                                "clam chowder", "dumplings", "edamame", "spaghetti bolognese", "strawberry shortcake",\
                                'apple pie', 'chicken wings', 'ice cream', 'oyster', 'pizza', 'tacos']

    elif args.dataset in ['ffhq', 'celeb']:
        init_prompt = 'a face with {}.'
        base_template = ['a face with.']
        all_prompt = ['5 o clock shadow', 'arched eyebrows', 'attractive face', 'bags under eyes', 'bald', 'bangs', 'big lips' ,'big Nose',\
                    'black hair','blond hair', 'blurry', 'brown hair', 'bushy eyebrows', 'cubby', 'double chin', 'eyeglasses', 'goatee', \
                    'gray hair', 'heavy makeup', 'high cheekbones', 'male', 'mouth slightly open', 'mustache', 'narrow eyes', 'no beard', \
                    'oval face', 'pale skin', 'pointy nose', 'receding hairline', 'rosy cheeks', 'sideburns', 'smiling', 'straight hair', \
                    'wavy hair', 'wearing earrings', 'wearing hat', 'wearing lipstick', 'wearing necklace', 'wearing necktie', 'young'] 
        
        if args.num_domains == 2:
            prompt = ['male', 'female']
        if args.num_domains == 4:
            prompt = ['blond hair', 'bangs' , 'smiling', 'wearing lipstick'] 
        elif args.num_domains == 7:
            prompt = ['blond hair', 'black hair' , 'smiling', 'wearing lipstick',  'arched eyebrows', 'bangs','mustache'] 
        elif args.num_domains == 13:
            prompt = ['bangs', 'blond hair', 'black hair' ,'smiling', 'arched eyebrows','heavy makeup','mustache', 'straight hair', 'wearing lipstick', 'male',
                   'eyeglass', 'pale skin', 'young'] 
        elif args.num_domains == 16:
            prompt = ['bangs', 'blond hair', 'black hair' ,'smiling', 'arched eyebrows','heavy makeup','mustache', 'straight hair', 'wearing lipstick', 'male',
                   'eyeglass', 'pale skin', 'young', 'wavy hair', 'bald', 'goatee'] 
        elif args.num_domains == 10:
            prompt =  ['bangs', 'blond hair', 'black hair' ,'smiling', 'arched eyebrows','heavy makeup','mustache', 'straight hair', 'wearing lipstick', 'male']
        elif args.num_domains == 40:
            prompt = ['5 o clock shadow', 'arched eyebrows', 'attractive face', 'bags under eyes', 'bald', 'bangs', 'big lips' ,'big Nose',\
                'black hair','blond hair', 'blurry', 'brown hair', 'bushy eyebrows', 'cubby', 'double chin', 'eyeglasses', 'goatee', \
                'gray hair', 'heavy makeup', 'high cheekbones', 'male', 'mouth slightly open', 'mustache', 'narrow eyes', 'no beard', \
                'oval face', 'pale skin', 'pointy nose', 'receding hairline', 'rosy cheeks', 'sideburns', 'smiling', 'straight hair', \
                'wavy hair', 'wearing earrings', 'wearing hat', 'wearing lipstick', 'wearing necklace', 'wearing necktie', 'young']
        
    prompt_idx = []
    for data in prompt:
        for idx, pt in enumerate(all_prompt):
            if data == pt:
                prompt_idx.append(idx)

    return init_prompt, prompt, prompt_idx, base_template

def clip_normalize(image, device="cuda"):
    image = F.interpolate(image,size=224,mode='bicubic')
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


@torch.no_grad()
def translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    s_ref = nets.style_encoder(x_ref, y_ref)
    x_fake = nets.generator(x_src, s_ref)
    s_src = nets.style_encoder(x_src, y_src)
    x_rec = nets.generator(x_fake, s_src)
    x_concat = [x_src, x_ref, x_fake, x_rec]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)
    del x_concat


def get_truncated_noise(n_samples, z_dim, truncation):
    '''
    Function for creating truncated noise vectors: Given the dimensions (n_samples, z_dim)
    and truncation value, creates a tensor of that shape filled with random
    numbers from the truncated normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        truncation: the truncation value, a non-negative scalar
    '''
    #### START CODE HERE ####
    truncated_noise = truncnorm.rvs(-truncation,truncation, size=(n_samples, z_dim))
    #### END CODE HERE ####
    return torch.Tensor(truncated_noise)

@torch.no_grad()
def translate_using_latent(nets, args, x_src, x_ref, y_ref, filename):
            
    N, C, H, W = x_src.size()
    wb = torch.ones(1, C, H, W).to(x_src.device)
    x_src_with_wb = torch.cat([wb, x_src], dim=0)
    z_trg = torch.randn(N, args.latent_dim).to(x_src.device)

    y_ref_lst = [0 for i in range(args.num_domains)]
    
    for i in args.latent_num:
        y_ref_lst[i] = 1
    y_ref  = torch.LongTensor( [ y_ref_lst ]*N ).to("cuda").view(N,args.num_domains) # view(1,10)=(N,topk), N = val_batch_size
    s_ref = nets.mapping_network(z_trg, y_ref)
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    x_concat = [x_src_with_wb]
    
    #resample_x_fakes = []
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(x_src, s_ref)
        #x_fake = nets.generator(x_fake, s_ref)
        #resample_x_fakes.append(x_fake)
        
        x_fake_with_ref = torch.cat([x_ref[i:i+1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]
    
    # x_src = torch.stack(resample_x_fakes)
    # x_concat = [x_src_with_wb]
    # for i, s_ref in enumerate(s_ref_list):
    #     x_fake = nets.generator(x_src, s_ref)
    #     resample_x_fakes.append(x_fake)
        
    #     x_fake_with_ref = torch.cat([x_ref[i:i+1], x_fake], dim=0)
    #     x_concat += [x_fake_with_ref]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N+1, filename)
    del x_concat

@torch.no_grad()
def translate_using_reference(nets, args, x_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    wb = torch.ones(1, C, H, W).to(x_src.device)
    x_src_with_wb = torch.cat([wb, x_src], dim=0)

    s_ref = nets.style_encoder(x_ref, y_ref)
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    x_concat = [x_src_with_wb]
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(x_src, s_ref)
        x_fake_with_ref = torch.cat([x_ref[i:i+1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N+1, filename)
    del x_concat

@torch.no_grad()
def debug_image(nets, args, inputs_val_src, inputs_val_ref, y_val_src, y_val_ref, step):
    x_src = inputs_val_src   
    x_ref = inputs_val_ref 

    y_src = y_val_src
    y_ref = y_val_ref 

    device = inputs_val_src.device
    N = inputs_val_src.size(0)

    # translate and reconstruct (reference-guided)
    filename = ospj(args.sample_dir, args.name, '%06d_cycle_consistency.jpg' % (step))
    translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename)

    # reference-guided image synthesis
    filename = ospj(args.sample_dir, args.name, '%06d_reference.jpg' % (step))
    translate_using_reference(nets, args, x_src, x_ref, y_ref, filename)
