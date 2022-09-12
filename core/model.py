"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

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

from core.utils import *
import copy


class PromptLearner(nn.Module):
    def __init__(self, args, device, K, classes, clip_model, init_prompt='a photo of the {}.', rand_token_len=4):
        super().__init__()
        self.args = args

        prefix, suffix = [x.strip() for x in init_prompt.split('{}')]
        
        self.K = K
        self.rand_token_len = rand_token_len
     
        prompt_prefix = clip.tokenize(prefix).to(device)
        prompt_suffix = clip.tokenize(suffix).to(device)
        class_tokens = clip.tokenize(classes).to(device)

        self.n_prompt_prefix = self.count_token(prompt_prefix).item()
        self.n_prompt_suffix = self.count_token(prompt_suffix).item()
        self.len_classes = self.count_token(class_tokens)
        
        self.max_len = prompt_prefix.shape[-1]

        with torch.no_grad():
            prefix_embedding = clip_model.token_embedding(prompt_prefix).squeeze(0)
            suffix_embedding = clip_model.token_embedding(prompt_suffix).squeeze(0)
            class_embedding = clip_model.token_embedding(class_tokens)

            sos_token = prefix_embedding[0] 
            eos_token = prefix_embedding[self.n_prompt_prefix + 1]
            padding = prefix_embedding[-1] 

        class_embeddings = []
        for i, l in enumerate(self.len_classes):
            class_embeddings.append(nn.Parameter(
                class_embedding[i, 1:l+1] # 클래스도 딱 ctx부분만. 
            ))

        rand_tokens = torch.zeros(K - len(classes), rand_token_len, class_embedding.size(-1)).to(device)
        nn.init.normal_(rand_tokens, std=0.02)

        self.rand_tokens = nn.Parameter(rand_tokens) # K - len(classes), 1, 512

        self.prefix_tokens = nn.Parameter(prefix_embedding[1:1 + self.n_prompt_prefix]) # (length_prefix, 512)
        self.class_tokens = nn.ParameterList(class_embeddings) # List of l, 512
        with torch.no_grad():

            self.suffix_tokens = nn.Parameter(suffix_embedding[1:1 + self.n_prompt_suffix]) # n_prompt, 512
            """ don't update class token and suffix_token, only update prefix. """

            self.suffix_tokens.requires_grad = False
            self.rand_tokens.requires_grad = False
            
        self.register_buffer('sos_token', sos_token)
        self.register_buffer('eos_token', eos_token)
        self.register_buffer('padding', padding)

    def count_token(self, x):
        return (x != 0).sum(1) - 2
    
    def get_embedding(self):
        embeddings = []
        for i, cls in enumerate(self.class_tokens):
            embed = torch.cat((
                self.sos_token[None],
                self.prefix_tokens,
                cls,
                self.suffix_tokens,
                self.eos_token[None]
            ))
            padding = self.padding[None].repeat(self.max_len - embed.size(0), 1)
            embeddings.append(torch.cat((embed, padding), 0))
        embeddings = torch.stack(embeddings)
        
        rand_len = self.rand_tokens.size(0)

        rand_embeddings = torch.cat((
            self.sos_token[None, None].repeat(rand_len, 1, 1),
            self.prefix_tokens[None].repeat(rand_len, 1, 1),
            self.rand_tokens,
            self.suffix_tokens[None].repeat(rand_len, 1, 1),
            self.eos_token[None, None].repeat(rand_len, 1, 1),
        ), dim=1)
        rand_embeddings = torch.cat((
            rand_embeddings,
            self.padding[None, None].repeat(rand_len, self.max_len - rand_embeddings.size(1), 1)
        ), dim=1)
        
        return torch.cat((embeddings, rand_embeddings), 0)
    
    def forward(self, clip_model):
        x = self.get_embedding()

        x = x + clip_model.positional_embedding.to(clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(clip_model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x_cls = x[range(len(self.len_classes)), self.len_classes + self.n_prompt_prefix + self.n_prompt_suffix + 1] @ clip_model.text_projection
        x_rand = x[len(self.len_classes):, self.rand_token_len + self.n_prompt_prefix + self.n_prompt_suffix + 1] @ clip_model.text_projection

        return torch.cat((x_cls, x_rand), 0)


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
                sty = torch.mean(sty, dim=1)
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
                sty = torch.mean(sty, dim=1)
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
        #import pdb;pdb.set_trace()
        out = self.main(x) #(8,10,4,4)
        out = out.view(out.size(0), -1)  # (batch, num_domains) #(8,160)

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
            # network의 출력값이 딱 0이 나오는 값은 없을거라 가정하고, 해당 코드 작성.
            # return out[ torch.where(out*y>0) ] 

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
        #import pdb;pdb.set_trace()
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
    #att_to_use, init_prompt, prompt, prompt_idx = get_prompt_and_att(args)
    args.num_domains = len(prompt) #+ len(prompt_out)

    """ model definition & initialization """
    generator = nn.DataParallel(Generator(args, args.img_size, args.style_dim, w_hpf=args.w_hpf))
    mapping_network = nn.DataParallel(MappingNetwork(args, args.latent_dim, args.style_dim, args.num_domains))
    style_encoder = nn.DataParallel(StyleEncoder(args, args.img_size, args.style_dim, args.num_domains))
    discriminator = nn.DataParallel(Discriminator(args, args.img_size, args.num_domains))
    if args.use_prompt:
        # """ 사용할 clip model load. """
        clip_model, preprocess = clip.load('ViT-B/32', device="cuda", jit=False) 
        """ freeze the network parameters """
        for clip_param in clip_model.parameters():
            clip_param.requires_grad = False
        promptLearner = nn.DataParallel(PromptLearner(args, device="cuda", K=len(prompt), init_prompt=template, classes=prompt, clip_model=clip_model))
        del clip_model

    """ make ema models """
    mapping_network_ema = copy.deepcopy(mapping_network)
    generator_ema = copy.deepcopy(generator)
    style_encoder_ema = copy.deepcopy(style_encoder)
    if args.use_prompt:
        promptLearner_ema = copy.deepcopy(promptLearner)
    

    """ make munch """
    if args.use_prompt:
        nets = Munch(generator=generator,
                    mapping_network=mapping_network,
                    style_encoder=style_encoder,
                    promptLearner = promptLearner,
                    discriminator=discriminator)
        nets_ema = Munch(generator=generator_ema,
                        mapping_network=mapping_network_ema,
                        style_encoder=style_encoder_ema,
                        promptLearner = promptLearner_ema,
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
