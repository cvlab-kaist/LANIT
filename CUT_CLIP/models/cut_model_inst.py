import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from clip_loss import CLIPLoss
from template import *
import clip
from torchvision import transforms

class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--clip_patch',action='store_true')
        parser.add_argument('--clip_dir',action='store_true')
        parser.add_argument('--ganloss',action='store_true')
        parser.add_argument('--nce_y',action='store_true')
        parser.add_argument('--clip_crop',action='store_true', help='instance-aware clip loss')
        parser.add_argument('--reg',action='store_true')

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        # device='cuda'
        self.clip_model, self.preprocess = clip.load('ViT-B/32', self.device, jit=False)
        if opt.nce_idt and self.isTrain:
            # import pdb;pdb.set_trace()
            if self.opt.nce_y:
                self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']
        if opt.clip_patch:
            self.loss_names += ['clip']
        if opt.clip_dir:
            self.loss_names += ['dirclip']
        if opt.clip_crop:
            self.loss_names += ['bboxclip']
        if opt.reg:
            self.loss_names += ['var_l2']
        if opt.ganloss:
            self.loss_names += ['G_GAN', 'D_real', 'D_fake']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        self.w = 256
        self.h = 256

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            # if self.opt.ganloss:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            # clip_loss = CLIPLoss().to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            if self.opt.ganloss:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_D)
            
            
        self.num_crops=8
        self.crop_size = 64

        self.cropper = transforms.Compose([
            transforms.RandomCrop(self.crop_size)
        ])
        self.augment = transforms.Compose([
            transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
            transforms.Resize(256)
        ])

        self.resize = transforms.Compose([
            transforms.Resize(256)
        ])

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            if self.opt.ganloss:
                self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def prepare_clip_input(self, source, prompt, instance_prompt):
        self.source = source
        self.prompt = prompt
        self.inst_prompt = instance_prompt
        # return super().prepare_clip_input(source, prompt)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D

        
        
        if self.opt.ganloss:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.loss_D = self.compute_D_loss()
            self.loss_D.backward()
            self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        #JB_EDIT
        if self.opt.use_box:
            self.box_info_A = input['A_box'].to(self.device)
            
            # self.box_info_B = input['B_box'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        #JB_EDIT
        if self.opt.use_box:
            # self.box_info = torch.cat((self.box_info_A, self.box_info_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.box_info_A 
            # self.style_dim = 8
            # self.s_b = torch.randn(self.real_B.size(0), self.style_dim, 1, 1).cuda() #4 or 9 - 210827 Edit #rand_style
            self.box_info = self.box_info_A
            # import pdb;pdb.set_trace()
            # self.fake_B, self.fake_B_box, self.fake_B_box_list = self.netG(self.real, self.box_info, rand_style=self.s_b,i2i=True) #networks에서 else의 output
            # self.fake_B_box = self.fake_B_box.flatten(0, 1)
            # self.fake_B_box_list = self.fake_B_box_list.tolist()


        self.fake = self.netG(self.real)

        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

        self.fake_bbox_images = []
        # import pdb;pdb.set_trace()
        for i in range(self.opt.num_box):
            for bbox in self.box_info[:,i]: # bbox --> [1.000, 0.1,0.1,0.2,0.2]
                if bbox[0] == -1:
                    continue
                # tmp = self.fake_B
                bbox[1] = int(bbox[1]*self.w)
                bbox[2] = int(bbox[2]*self.h)
                bbox[3] = int(bbox[3]*self.w)
                bbox[4] = int(bbox[4]*self.h)
                crop = self.fake_B.clone()[:,:,int(bbox[2]):int(bbox[4]),int(bbox[1]):int(bbox[3])]

                self.fake_bbox_images.append(crop)#(원본 이미지에서 자른 이미지.clone()) # clone() 꼭 사용하고

        # self.fake_bbox_images = [img tensor1, img tensor2 ....]
        loss_patch=0
        with torch.no_grad():
            
            template_text = compose_text_with_templates(self.prompt, imagenet_templates) #a bad photo of a rainy day
            tokens = clip.tokenize(template_text).to(self.device)

            text_features = self.clip_model.encode_text(tokens).detach()
            text_features = text_features.mean(axis=0, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            inst_template_text = compose_text_with_templates(self.inst_prompt, imagenet_templates) #a bad photo of a rainy day
            inst_tokens = clip.tokenize(inst_template_text).to(self.device)
            inst_text_features = self.clip_model.encode_text(inst_tokens).detach()
            inst_text_features = inst_text_features.mean(axis=0, keepdim=True)
            inst_text_features /= inst_text_features.norm(dim=-1, keepdim=True)
            
            template_source = compose_text_with_templates(self.source, imagenet_templates) # a bad photo of a sunny day
            # import pdb;pdb.set_trace()
            tokens_source = clip.tokenize(template_source).to(self.device)
            text_source = self.clip_model.encode_text(tokens_source).detach()
            text_source = text_source.mean(axis=0, keepdim=True)
            text_source /= text_source.norm(dim=-1, keepdim=True)
            source_features = self.clip_model.encode_image(clip_normalize(self.real_A.to(self.device),self.device))
            source_features /= (source_features.clone().norm(dim=-1, keepdim=True))
        img_proc =[]

        for n in range(self.num_crops):
            target_crop = self.cropper(self.fake_B)
            target_crop = self.augment(target_crop)
            img_proc.append(target_crop)
        img_proc = torch.cat(img_proc,dim=0)
        img_aug = img_proc
        image_features = self.clip_model.encode_image(clip_normalize(img_aug,self.device))
        image_features /= (image_features.clone().norm(dim=-1, keepdim=True))

        self.img_direction = (image_features-source_features) #image_features : 64,512, source_features : 1,512
        self.img_direction /= self.img_direction.clone().norm(dim=-1, keepdim=True) #64,512
        
        #global
        self.text_direction = (text_features-text_source).repeat(image_features.size(0),1)
        self.text_direction /= self.text_direction.norm(dim=-1, keepdim=True) #64,512

        #instance
        self.inst_text_direction = (inst_text_features-text_source).repeat(image_features.size(0),1)
        self.inst_text_direction /= self.inst_text_direction.norm(dim=-1, keepdim=True) #64,512
        
        self.glob_features = self.clip_model.encode_image(clip_normalize(self.fake_B,self.device))
        self.glob_features /= (self.glob_features.clone().norm(dim=-1, keepdim=True))
        
        self.glob_direction = (self.glob_features-source_features)
        self.glob_direction /= self.glob_direction.clone().norm(dim=-1, keepdim=True)

        self.crop_features = []
        for fake_bbox_img in self.fake_bbox_images: # 혹시나 비어있어도 돌아가면 if 문 사용 
            tmp_features = self.clip_model.encode_image(clip_normalize(self.resize(fake_bbox_img),self.device))
            tmp_features /= (tmp_features.clone().norm(dim=-1, keepdim=True))
            self.crop_features.append(tmp_features)


    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        # clip_loss = CLIPLoss()
        if self.opt.ganloss:
            if self.opt.lambda_GAN > 0.0:
                pred_fake = self.netD(fake)
                self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
            else:
                self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)*6

            # self.loss_clip =  self.clip_loss(self.fake_B, text_inputs)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            loss_NCE_both = ( self.loss_NCE ) * 0.5
            if self.opt.nce_y:
                self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
                loss_NCE_both += (self.loss_NCE_Y) * 0.5
            
        else:
            loss_NCE_both = self.loss_NCE
        if self.opt.ganloss:
            self.loss_G = self.loss_G_GAN + loss_NCE_both 
        else:
            self.loss_G = loss_NCE_both 
        if self.opt.clip_patch:
            self.loss_clip = self.calculate_patchclip_loss()*10
            self.loss_G += self.loss_clip
        if self.opt.clip_dir:
            self.loss_dirclip = self.calculate_dirclip_loss()*5
            self.loss_G += self.loss_dirclip
        if self.opt.clip_crop: #옵션추가하기
            self.loss_bboxclip = self.calculate_bboxclip_loss()
            self.loss_G += self.loss_bboxclip
        if self.opt.reg:
            # R_prior losses
            # import pdb;pdb.set_trace()
            self.loss_var_l2 = self.get_image_prior_losses(self.fake_B)*0.002
            self.loss_G += self.loss_var_l2
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
    
    def calculate_patchclip_loss(self):
        loss_patch=0
        loss_temp = (1- torch.cosine_similarity(self.img_direction, self.text_direction, dim=1))
        # loss_temp[loss_temp<args.thresh] =0
        
        loss_patch+=loss_temp.mean()
        return loss_patch

    def calculate_dirclip_loss(self):
        loss_glob = (1- torch.cosine_similarity(self.glob_direction, self.text_direction, dim=1)).mean()
        return loss_glob


    def calculate_bboxclip_loss(self):
        if len(self.crop_features) == 0:
            return 0
        loss_bbox = 0
        for crop_features in self.crop_features:
            loss_bbox += (1 - torch.cosine_similarity(crop_features, self.inst_text_direction, dim=1)).mean()
        return loss_bbox/len(self.crop_features)

    def get_image_prior_losses(self,inputs_jit):
        # COMPUTE total variation regularization loss
        # import pdb;pdb.set_trace()
        diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
        diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
        diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
        diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

        loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
        # loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
        #         diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
        # loss_var_l1 = loss_var_l1 * 255.0
        return loss_var_l2
