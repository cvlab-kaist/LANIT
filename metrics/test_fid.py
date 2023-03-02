"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
from prdc import compute_prdc

import torch
import argparse
import os

import pdb
import cv2
import numpy as np

import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from pytorch_fid.inception import InceptionV3

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)
    #import pdb;pdb.set_trace()
    height, width = 299, 299
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = TF.Compose([
        TF.Resize([256, 256]),
        TF.Resize([height, width]),
        TF.ToTensor(),
        TF.Normalize(mean=mean, std=std)
    ])
    dataset = ImagePathDataset(files, transforms=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0] #(8,2048,1,1)

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """

    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma, act


def compute_statistics_of_path(path, model, batch_size, dims, device,
                               num_workers=1):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        m, s, act = calculate_activation_statistics(files, model, batch_size,
                                               dims, device, num_workers)

    return m, s, act


def calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)
    #import pdb;pdb.set_trace()
    m1, s1, act1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device, num_workers)
    m2, s2, act2 = compute_statistics_of_path(paths[1], model, batch_size,
                                        dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value, act1, act2


def main_fid(args, prompt_idx, mode): 
    
    if args.num_domains == 10:
        domains_list = os.listdir(args.val_img_dir)
        domains_list.sort() 
        for idx, i in enumerate(prompt_idx):
            domains.append(domains_list[i])
    else:
        domains_list = os.listdir(args.val_img_dir)
        domains_list.sort() 
        domains = domains_list
    num_domains = len(domains)
    print('Number of domains: %d' % num_domains)

    weights = []
    best_FID = 150.0
    best_FID_model = []

    for w in os.listdir(os.path.join(args.eval_dir, args.name, 'reference')):
        resume_iter = w
        fid_total = []
        density_total = []
        convergence_total = []
        for trg_idx, trg_domain in enumerate(domains):
            device = 'cuda'

            task = '%s' % (trg_domain)
            if mode == 'latent':
                dir = os.path.join(args.eval_dir, args.name, 'latent', str(resume_iter), task)
                save_dir = os.path.join(args.eval_dir, args.name,'latent', str(resume_iter))
            else:
                dir = os.path.join(args.eval_dir, args.name,'reference',str(resume_iter), task)
                save_dir = os.path.join(args.eval_dir, args.name,  'reference', str(resume_iter))
            path = [dir , os.path.join(args.val_img_dir, trg_domain)]

            fid_value, fake_feat, real_feat = calculate_fid_given_paths(path,
                                                                        args.batch_size,
                                                                        device,
                                                                        2048,
                                                                        args.num_workers)
            
            fid_total.append(fid_value)
            with open(os.path.join(save_dir,'./FID_all.txt'),'a') as f:
                f.write(f'FID_score:{fid_value}\n')

            """ for d&c """
            nearest_k = 5
            feature_dim = 2048
            metrics = compute_prdc(real_features=real_feat,
                                    fake_features=fake_feat,
                                    nearest_k=nearest_k)

            density_total.append(metrics["density"])
            convergence_total.append(metrics["coverage"])
            with open(os.path.join(save_dir, './DC_all.txt'),'a') as f:
                f.write(f'diversity_score:{metrics["density"]}, convergence_score:{metrics["coverage"]}\n')

            print("FID: {0}, density: {1}, convergence: {2}".format(fid_value, metrics["density"], metrics["coverage"]) )
            
        total_fid = sum(fid_total)/len(fid_total)
        total_density = sum(density_total)/len(density_total)
        total_convergence = sum(convergence_total)/len(convergence_total)
        with open(os.path.join(save_dir, './FID_all.txt'),'a') as f:
            f.write(f'total_FID_score:{total_fid}\n')
        with open(os.path.join(save_dir, './DC_all.txt'),'a') as f:
            f.write(f'total_diversity_score:{total_density}, total_convergence_score:{total_convergence}\n')
            
        print("total_FID: {0}, total_density: {1}, total_convergence: {2}".format(total_fid, total_density, total_convergence) )
        if total_fid <= best_FID:
            if total_fid == best_FID:
                best_FID_model.append(w)
            else:
                best_FID_model = []
                best_FID_model.append(w)
                best_FID = total_fid
                
    with open(os.path.join(args.eval_dir, args.name, './best_FID.txt'), 'a') as f:
        f.write(f'best_weight(FID_score:{best_FID}):{best_FID_model}\n')

