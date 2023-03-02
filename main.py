"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse

from munch import Munch
from torch.backends import cudnn
import torch

from core.data_loader import get_train_loader
from core.data_loader import get_test_loader
from core.solver import Solver
import util as util


def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


def main(args):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    solver = Solver(args)

    if args.mode == 'train':
        loaders = Munch(src=get_train_loader(root=args.train_img_dir,
                                             which='source',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             shuffle = True,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers),
                        ref=get_train_loader(root=args.train_img_dir,
                                             which='reference',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             shuffle = True,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers),
                        val_src=get_test_loader(root=args.val_img_dir,
                                            which='source',
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers),
                        val_ref=get_test_loader(root=args.val_img_dir,
                                            which='reference',
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers)
                                            )

        solver.train(loaders)

    elif args.mode == 'sample':
        loaders = Munch(src=get_test_loader(root=args.src_dir,
                                            which='source',
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers),
                        ref=get_test_loader(root=args.ref_dir,
                                            which='reference',
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers))
        solver.sample(loaders)

    elif args.mode == 'eval':
        solver.evaluate()

    elif args.mode == "fid":
       solver.FID()

    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=10,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')

    # weight for objective functions
    parser.add_argument('--lambda_cyc', type=float, default=1,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_lpips', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_dc', type=float, default=0.5,
                        help='Weight for clip contrastive loss')
    parser.add_argument('--lambda_ds', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--lambda_dc_reg', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=130000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_p', type=float, default=0.5,
                        help='Weight for prompt learning')
    parser.add_argument('--w_hpf', type=float, default=0,
                        help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=150000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=7,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=8,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--m_lr', type=float, default=1e-4,
                        help='Learning rate for mapping network')
    parser.add_argument('--p_lr', type=float, default=2e-5,
                        help='Learning rate for promptLearner')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=1,
                        help='Number of generated images per domain during sampling')

    # misc
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'sample', 'eval', 'align', 'fid'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='data/celeba_hq/train',
                        help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='data/celeba_hq/val',
                        help='Directory containing validation images')
    parser.add_argument('--src_img_dir', type=str, default='data/celeba_hq/val',
                        help='Directory containing validation images')
    parser.add_argument('--ref_img_dir', type=str, default='data/celeba_hq/val',
                        help='Directory containing validation images')
    parser.add_argument('--sample_dir', type=str, default='expr/samples',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='~/dataset1/smoothing/food/checkpoints',
                        help='Directory for saving network checkpoints')
    parser.add_argument('--lpips_type', default='alex', type=str, help='LPIPS backbone')

    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='expr/eval',
                        help='Directory for saving metrics, i.e., FID and LPIPS')
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--infer_mode', type=str, default='reference', help='mode to inference model')
    parser.add_argument('--latent_num', nargs="+", type=int, help='number to use for attribute')

    # directory for testing
    parser.add_argument('--result_dir', type=str, default='expr/results',
                        help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default='root/Data/data/animal_faces',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref',
                        help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                        help='output directory when aligning faces')

    # face alignment
    parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='expr/checkpoints/celeba_lm_mean.npz')
    parser.add_argument('--input', type=str, default='assets/real_A', help='input image name')
    parser.add_argument('--test_mode', type=str, default='reference', help='[latent | reference]')

    # step size
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=100000)

    parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
    parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
    parser.add_argument('--display_id', type=int, default=None, help='window id of the web display. Default is random window id')
    parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
    parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
    parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
    parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
    parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
    parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
    parser.add_argument('--name', type=str, default='animal_faces', help='name of the experiment. It decides where to store samples and models')

    """ LANIT additional parameters: start """
    parser.add_argument('--dataset', default='animal_faces', help='Dataset name to use',
                choices=['celeb', 'ffhq', 'food', 'food10', 'animal_faces', 'animal_faces_10', 'lsun_church', 'lsun_car', 'anime', 'metface', 'anime', 'landscape'])
    
    parser.add_argument('--step1',action='store_true', help='step1')
    parser.add_argument('--step2',action='store_true', help='step2')

    parser.add_argument('--dc',action='store_true', help='use domain consistency loss')
    parser.add_argument('--dcycle',action='store_true', help='use domain regularization loss')
    parser.add_argument('--ds',action='store_true', help='use style diversification loss')
    parser.add_argument('--recon_lpips',action='store_true', help='use lpips consistency loss')
    parser.add_argument('--cycle',action='store_true', help='use cycle consistency loss')

    parser.add_argument('--zero_cut',action='store_true', help='use cut under zero if top k > 1')

    parser.add_argument('--multi_hot',action='store_true', help='multi-hot encoding of logits of style E & Discriminator ')
    parser.add_argument('--topk', action='store_true', help='top_k value')

    parser.add_argument('--use_base', action='store_true', help='filtering out of class to be used')
    parser.add_argument('--cal_fid', action='store_true', help='calculate fid when evaluate')
    parser.add_argument('--dict', action='store_true', help='calculate fid when evaluate')

    """ add this arguemnt in step2. """
    parser.add_argument('--text_aug',    action='store_true', help='text augmentation')
    parser.add_argument('--base_fix',    action='store_true', help='fix base prompt')
    parser.add_argument('--g_update',    action='store_true', help='use augmentation with mapping network')
    parser.add_argument('--t_update',    action='store_true', help='use augmentation with mapping network')
    parser.add_argument('--gt_update',    action='store_true', help='use augmentation with mapping network')
    parser.add_argument('--alter_update',    action='store_true', help='use augmentation with mapping network')

    parser.add_argument('--use_scheduler',    action='store_true', help='use augmentation with mapping network')

    
    """ end """

    args = parser.parse_args()
    main(args)
