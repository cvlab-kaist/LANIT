"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from mimetypes import suffix_map
import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms.functional as FF

from core.wing import FAN

import functools
from torch.nn import init
from torch.optim import lr_scheduler

import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from template import * 


from core.utils import *
import copy


class PromptLearner(nn.Module):
    def __init__(self, args, device, K, classes, clip_model, init_prompt='a photo of the {}.', rand_token_len=4):
        super().__init__()
        self.args = args

        prefix, suffix = [x.strip() for x in init_prompt.split('{}')]
        
        self.K = K
        self.rand_token_len = rand_token_len

        """ 1. tokenize """
        prompt_prefix = clip.tokenize(prefix).to(device) # (1, 77),                prefix = "a face with"
        prompt_suffix = clip.tokenize(suffix).to(device) # (1, 77)                 suffix = "."
        class_tokens = clip.tokenize(classes).to(device) # (self.len_classes, 77), classes=['bangs', 'blond hair', ...,]

        self.n_prompt_prefix = self.count_token(prompt_prefix).item()
        self.n_prompt_suffix = self.count_token(prompt_suffix).item()
        self.len_classes = self.count_token(class_tokens) # tensor([1, 2, 2, 1, 2, 2, 2, 2, 2, 1], device='cuda:0'), 1='bangs', 2='blond hair', ...
        self.dict_name = []
        for k in range(len(class_tokens)):
            self.dict_name.append("cls{0}".format(k))
        dum_tokens = {}

        self.max_len = prompt_prefix.shape[-1]

        """ 2. token embedding """
        with torch.no_grad():
            # length = 77, [sos] [sentences] [eos] [pad]...[pad]
            prefix_embedding = clip_model.token_embedding(prompt_prefix).squeeze(0) # (77, 512) # embedder (49040, 512)
            suffix_embedding = clip_model.token_embedding(prompt_suffix).squeeze(0) # (77, 512)
            class_embedding = clip_model.token_embedding(class_tokens)              # (self.len_classes, 77, 512)

            self.sos_token = nn.Parameter(prefix_embedding[0]).to(device)
            self.eos_token = nn.Parameter(prefix_embedding[self.n_prompt_prefix + 1]).to(device)
            self.padding = nn.Parameter(prefix_embedding[-1]).to(device)

        class_embeddings = []
        for i, l in enumerate(self.len_classes):
            class_embeddings.append(nn.Parameter(
                class_embedding[i, 1:l+1] 
            ).to(device))

        rand_tokens = torch.zeros(K - len(classes), rand_token_len, class_embedding.size(-1))#.to(device) # (0, 4, 512)
        nn.init.normal_(rand_tokens, std=0.02)

        """ 3. parameterization of tokens """
        self.prefix_tokens = nn.Parameter(prefix_embedding[1:1 + self.n_prompt_prefix]).to(device) # (len(prefix), 512)
        """ 4. detach tokens not to update """
        with torch.no_grad():
            """ don't update class token and suffix_token, only update prefix. """
            cls_emb_dict = {}
            for k in range(len(class_embeddings)):
                cls_emb_dict[self.dict_name[k]] =class_embeddings[k]
            self.class_embeddings = cls_emb_dict#.to(device)
            self.suffix_tokens = nn.Parameter(suffix_embedding[1:1 + self.n_prompt_suffix]).to(device) # n_prompt, 512
            self.rand_tokens = nn.Parameter(rand_tokens).to(device)

    def count_token(self, x):
        return (x != 0).sum(1) - 2
    
    def get_embedding(self, device="cuda"):
        embeddings = []
        with torch.no_grad():
            self.sos_token = self.sos_token.to(device)
            self.suffix_tokens = self.suffix_tokens.to(device)
            self.eos_token =  self.eos_token.to(device)
            self.padding =  self.padding.to(device)
            for i, cls in enumerate(self.class_embeddings):
                self.class_embeddings[cls] = self.class_embeddings[cls].to(device)

        for i, cls in enumerate(self.class_embeddings):
            embed = torch.cat((
                self.sos_token[None], # self.dum_tokens["sos_token"][None],
                self.prefix_embeddings[cls],#self.prefix_tokens,
                self.class_embeddings[cls],#cls[self.dict_name[i]],
                self.suffix_tokens,
                self.eos_token[None], # self.dum_tokens["eos_token"][None],
            ))
            padding = self.padding[None].repeat(self.max_len - embed.size(0), 1)
            #padding = self.dum_tokens["padding"][None].repeat(self.max_len - embed.size(0), 1)
            embeddings.append(torch.cat((embed, padding), 0))
        
        embeddings = torch.stack(embeddings) # (self.len_classes, 77, 512)
        return embeddings

    def get_base_embedding(self, device="cuda"):
        #self.prefix_tokens = self.prefix_tokens.to(device)
        with torch.no_grad():
            self.sos_token = self.sos_token.to(device)
            self.suffix_tokens = self.suffix_tokens.to(device)
            self.eos_token =  self.eos_token.to(device)
            self.padding =  self.padding.to(device)

        embeddings = []
        for i, cls in enumerate(self.prefix_embeddings):
            embed = torch.cat((
                self.sos_token[None], # self.dum_tokens["sos_token"][None],
                self.prefix_embeddings[cls],#self.prefix_tokens,
                self.suffix_tokens,
                self.eos_token[None], # self.dum_tokens["eos_token"][None],
            ))
            padding = self.padding[None].repeat(self.max_len - embed.size(0), 1)
            embeddings.append(torch.cat((embed, padding), 0))
        
        embeddings = torch.stack(embeddings) # (self.len_classes, 77, 512)
        return embeddings

    def forward_prefix(self, clip_model, device="cuda"):
        x = self.get_base_embedding() # = clip_model.token_embedding(text)
        x = x + clip_model.positional_embedding.to(clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(clip_model.dtype)

        x_cls = x[:, 1 + self.n_prompt_prefix + self.n_prompt_suffix] @ clip_model.text_projection
        return x_cls

    def forward(self, clip_model, device="cuda"):
        x = self.get_embedding(device) # = clip_model.token_embedding(text)
        
        x = x + clip_model.positional_embedding.to(clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(clip_model.dtype)

        x_cls = x[range(len(self.len_classes)), self.len_classes + self.n_prompt_prefix + self.n_prompt_suffix + 1] @ clip_model.text_projection

        return x_cls

class PromptMean(nn.Module):
    def __init__(self, args, K, device, classes, clip_model, templates):
        super().__init__()
        self.args = args
        self.K = K
        prefix = []
        for temp in templates:
            prefix_one, suffix = [x.strip() for x in temp.split('{}')]
            prefix.append(prefix_one)

        
        """ 1. tokenize """
        prompt_prefix = clip.tokenize(prefix).to(device) # (9(self.len_prefix), 77),                prefix = "a face with"
        prompt_suffix = clip.tokenize(suffix).to(device) # (9, 77)                 suffix = "."
        class_tokens = clip.tokenize(classes).to(device) # (self.len_classes, 77), classes=['bangs', 'blond hair', ...,] #self.len_classes

        self.n_prompt_prefix = self.count_token(prompt_prefix) #tensor([3, 4, 4, 5, 4, 3, 3, 4, 5], device='cuda:0')
        self.n_prompt_suffix = self.count_token(prompt_suffix).item() #tensor([1, 1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0')
        self.len_classes = self.count_token(class_tokens) # tensor([1, 2, 2, 1, 2, 2, 2, 2, 2, 1], device='cuda:0'), 1='bangs', 2='blond hair', ...
        self.dict_name = []
        self.prefix_dict_name = []

        for k in range(len(class_tokens)):
            self.dict_name.append("cls{0}".format(k))
        
        for k in range(len(prompt_prefix)):
            self.prefix_dict_name.append("pre{0}".format(k))

        self.max_len = prompt_prefix.shape[-1]

        """ 2. token embedding """
        with torch.no_grad():
            # length = 77, [sos] [sentences] [eos] [pad]...[pad]
            prefix_embedding = clip_model.token_embedding(prompt_prefix) # (9, 77, 512) # embedder (49040, 512)
            suffix_embedding = clip_model.token_embedding(prompt_suffix).squeeze(0) # (77, 512)
            class_embedding = clip_model.token_embedding(class_tokens)              # (self.len_classes, 77, 512)

            self.sos_token = nn.Parameter(prefix_embedding[0][0]).to(device)
            self.padding = nn.Parameter(prefix_embedding[0][-1]).to(device)

            class_embeddings = []
            for i, l in enumerate(self.len_classes):
                class_embeddings.append(class_embedding[i, 1:l+1] .to(device))

            prefix_token_all = []
            eos_token_all = []
            for j, n in enumerate(self.n_prompt_prefix):
                eos_token_all.append(nn.Parameter(prefix_embedding[j][self.n_prompt_prefix[j] + 1]).to(device))
                prefix_token_all.append(nn.Parameter(prefix_embedding[j,1:n+1].to(device))) # (len(prefix), 512)

        """ 4. detach tokens not to update """
        if args.step2:
            pre_emb_dict = {}
            for k in range(len(prefix_token_all)):
                pre_emb_dict[self.prefix_dict_name[k]] =prefix_token_all[k].requires_grad_(True)
            self.prefix_tokens = nn.ParameterDict(pre_emb_dict)
        else:
            with torch.no_grad():
                pre_emb_dict = {}
                for k in range(len(prefix_token_all)):
                    pre_emb_dict[self.prefix_dict_name[k]] = prefix_token_all[k].requires_grad_(False)
                self.prefix_tokens = nn.ParameterDict(pre_emb_dict)

        with torch.no_grad():
            """ don't update class token and suffix_token, only update prefix. """
            self.suffix_tokens = suffix_embedding[1:1 + self.n_prompt_suffix].to(device)

            cls_emb_dict = {}
            for k in range(len(class_embeddings)):
                cls_emb_dict[self.dict_name[k]] =class_embeddings[k]
            self.class_embeddings = cls_emb_dict

            eos_emb_dict = {}
            for k in range(len(eos_token_all)):
                eos_emb_dict[self.prefix_dict_name[k]] =eos_token_all[k]
            self.eos_token = eos_emb_dict


    def count_token(self, x):
        return (x != 0).sum(1) - 2
    
    def get_embedding(self, device="cuda"):
        embeddings = []

        with torch.no_grad():
            self.sos_token = self.sos_token.to(device)
            self.padding =  self.padding.to(device)
            self.suffix_tokens =  self.suffix_tokens.to(device)

            for i, cls in enumerate(self.class_embeddings):
                self.class_embeddings[cls] = self.class_embeddings[cls].to(device)
            for i, pre in enumerate(self.prefix_tokens):
                self.prefix_tokens[pre] = self.prefix_tokens[pre].to(device)
            for i, eos in enumerate(self.eos_token):
                self.eos_token[eos] = self.eos_token[eos].to(device)

        for j, pre in enumerate(self.prefix_tokens):
            cls_embedding = []
            for i, cls in enumerate(self.class_embeddings):
            #for i, cls in enumerate(self.class_tokens):
                embed = torch.cat((
                    self.sos_token[None], # self.dum_tokens["sos_token"][None],
                    self.prefix_tokens[pre],#self.prefix_tokens,
                    self.class_embeddings[cls],#cls[self.dict_name[i]],
                    self.suffix_tokens,
                    self.eos_token[pre][None], # self.dum_tokens["eos_token"][None],
                ))
                padding = self.padding[None].repeat(self.max_len - embed.size(0), 1)
                cls_embedding.append(torch.cat((embed, padding), 0))

            cls_embedding = torch.stack(cls_embedding) # (self.len_classes, 77, 512)
            embeddings.append(cls_embedding)

        embeddings = torch.stack(embeddings)
        return embeddings

    def get_base_embedding(self, device="cuda"):
        with torch.no_grad():
            self.sos_token = self.sos_token.to(device)
            self.suffix_tokens = self.suffix_tokens.to(device)
            self.padding =  self.padding.to(device)

            for i, pre in enumerate(self.prefix_tokens):
                self.prefix_tokens[pre] = self.prefix_tokens[pre].to(device)
            for i, eos in enumerate(self.eos_token):
                self.eos_token[eos] = self.eos_token[eos].to(device)

        embeddings = []
        for i, pre in enumerate(self.prefix_tokens):
            embed = torch.cat((
                self.sos_token[None], # self.dum_tokens["sos_token"][None],
                self.prefix_tokens[pre],#self.prefix_tokens,
                self.suffix_tokens,
                self.eos_token[pre][None], # self.dum_tokens["eos_token"][None],
            ))
            padding = self.padding[None].repeat(self.max_len - embed.size(0), 1)
            embeddings.append(torch.cat((embed, padding), 0))
        
        embeddings = torch.stack(embeddings) # (self.len_classes, 77, 512)
        return embeddings

    def init_mean_embed(self, clip_model, device="cuda"):
        cls_mean = []
        emb = self.get_embedding(device) # = clip_model.token_embedding(text)
        for i in range(len(self.n_prompt_prefix)):
            x = emb[i]
            x = x + clip_model.positional_embedding.to(clip_model.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = clip_model.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = clip_model.ln_final(x).type(clip_model.dtype)
            cls_mean.append(x[range(len(self.len_classes)), self.len_classes + self.n_prompt_prefix[i] + self.n_prompt_suffix + 1])
            
        cls_mean = torch.stack(cls_mean)
        return torch.mean(cls_mean @ clip_model.text_projection, dim=0, keepdim=True)


    def init_base_mean_embed(self, clip_model, device="cuda"):
        x = self.get_base_embedding() # = clip_model.token_embedding(text)
        x = x + clip_model.positional_embedding.to(clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(clip_model.dtype)
        x_cls = x[ torch.arange(9), 1 + self.n_prompt_prefix + self.n_prompt_suffix]
        return torch.mean(x_cls @ clip_model.text_projection, dim=0, keepdim=True)
        
    def forward_prefix(self, clip_model, device="cuda"):
        return self.init_base_mean_embed(clip_model, device=device)

    def forward(self, clip_model, device="cuda"):
        return self.init_mean_embed(clip_model, device=device)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


class Generator(nn.Module):
    def __init__(self, args, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2**14 // img_size
        self.args = args
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        if w_hpf > 0:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)
            
    def forward(self, x, s):
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)
        for block in self.decode:
            x = block(x, s)
        return self.to_rgb(x)


class MappingNetwork(nn.Module):
    def __init__(self, args, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        self.args = args
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]

    def forward(self, z, y):
        h = self.shared(z) #(8,512)
        out = []
        for layer in self.unshared:
            out += [layer(h)] #(8,64)
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim) #(8,10,64)
        if self.args.multi_hot: #y = idx
            if self.args.use_base and self.args.zero_cut:
                sty = out * y.unsqueeze(-1) 
                sty = torch.sum(sty, dim=1)/torch.sum(y, dim=-1, keepdim=True) #sty = torch.mean(sty, dim=1)
            else:
                idx = torch.LongTensor( [ [i] for i in range(y.size(0)) ] )
                sty = out[idx, y]
                sty = torch.mean(sty, dim=1)
        return sty


class StyleEncoder(nn.Module):
    def __init__(self, args, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        self.args = args
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        #import pdb;pdb.set_trace()
        h = self.shared(x) #8,512,4,4
        h = h.view(h.size(0), -1) #8,8192
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)

        if self.args.multi_hot:
            if self.args.use_base and self.args.zero_cut: 
                sty = out * y.unsqueeze(-1)
                sty = torch.sum(sty, dim=1)/torch.sum(y, dim=-1, keepdim=True) #sty = torch.mean(sty, dim=1)
            else:
                idx = torch.LongTensor( [ [i] for i in range(y.size(0)) ] )
                sty = out[idx, y]
                sty = torch.mean(sty, dim=1)

        return sty

class Discriminator(nn.Module):
    def __init__(self, args, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        self.args = args
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x) #(8,10,4,4)
        out = out.view(out.size(0), -1)  # (batch, num_domains) #(8,160)
        
        
        output = [] 

        if self.args.multi_hot:
            if self.args.use_base and self.args.zero_cut:
                output = [] #[[] for i in range(y.size(0))]
                for i in range(y.size(0)):
                    for j in range(y.size(1)):
                        if y[i][j] == 1:
                            output.append(out[i][j][None])
                        else:
                            continue
                return torch.cat(output, dim=0)#.type(torch.cuda.FloatTensor).to("cuda")            
            else:
                idx = torch.LongTensor( [ [i] for i in range(y.size(0)) ] )
                out = out[idx, y] # (batch, num_topk)
                return out
               
        elif self.args.thresholding:
            output = [] #[[] for i in range(y.size(0))]
            for i in range(y.size(0)):
                for j in range(y.size(1)):
                    if y[i][j] == 1:
                        output.append(out[i][j][None])
                    else:
                        continue
            return torch.cat(output, dim=0)


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


def build_model(args):
    """ get attr and prompt to use. """
    template, prompt, prompt_idx, base_template = get_prompt_and_att(args)
    args.num_domains = len(prompt)

    """ model definition & initialization """
    generator = nn.DataParallel(Generator(args, args.img_size, args.style_dim, w_hpf=args.w_hpf))
    mapping_network = nn.DataParallel(MappingNetwork(args, args.latent_dim, args.style_dim, args.num_domains))
    style_encoder = nn.DataParallel(StyleEncoder(args, args.img_size, args.style_dim, args.num_domains))
    discriminator = nn.DataParallel(Discriminator(args, args.img_size, args.num_domains))
    if args.text_aug or args.step2:
        clip_model, preprocess = clip.load('ViT-B/32', device="cpu", jit=False) 
        """ freeze the network parameters """
        for clip_param in clip_model.parameters():
            clip_param.requires_grad = False

        if not args.text_aug:
            promptLearner = nn.DataParallel(PromptLearner(args, device="cuda", K=len(prompt), init_prompt=template, classes=prompt, clip_model=clip_model))
        else:
            imagenet_templates, base_imagenet_templates = get_templates(args.dataset)
            promptLearner = nn.DataParallel(PromptMean(args, device="cuda", K=len(prompt), templates=imagenet_templates, classes=prompt, clip_model=clip_model.to('cuda')))
        del clip_model


    """ make ema models """
    mapping_network_ema = copy.deepcopy(mapping_network)
    generator_ema = copy.deepcopy(generator)
    style_encoder_ema = copy.deepcopy(style_encoder)
    if args.step2 or args.text_aug:
        promptLearner_ema = copy.deepcopy(promptLearner)
    

    """ make munch """
    if args.text_aug or args.step2:
        nets = Munch(generator=generator,
                    mapping_network=mapping_network,
                    style_encoder=style_encoder,
                    promptLearner = promptLearner,
                    discriminator=discriminator)
        nets_ema = Munch(generator=generator_ema,
                        mapping_network=mapping_network_ema,
                        promptLearner = promptLearner_ema,
                        style_encoder=style_encoder_ema,
                        )        
    else:
        nets = Munch(generator=generator,
                    mapping_network=mapping_network,
                    style_encoder=style_encoder,
                    discriminator=discriminator)
        nets_ema = Munch(generator=generator_ema,
                        mapping_network=mapping_network_ema,
                        style_encoder=style_encoder_ema,
                        )

    return nets, nets_ema
