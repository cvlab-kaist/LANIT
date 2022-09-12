"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from core.custom_dataset import ImageFolerRemap, CrossdomainFolder


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG','webp']]))
    return fnames


"""
DefaultDataset: ony get image
FilePathDefaultDataset: get image, filepath 
ReferenceDataset: get ref1, ref2

get_train_loader: trainloader
get_eval_loader: evaloader
get_filePathEval_loader: evaloader with filepath
get_test_loader: testloader
"""
class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)

class FilePathDefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort() # 정렬!
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        domains = os.listdir(root)
        domains = sorted(domains) # 정렬!

        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(domains):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)

        return img,img2, label

    def __len__(self):
        return len(self.targets)


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, prob=0.5, shuffle = True, num_workers=4):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    crop = transforms.RandomResizedCrop(
        img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < prob else x)

    transform = transforms.Compose([
        rand_crop,
        transforms.Resize([img_size, img_size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    if which == 'source':
        dataset = ImageFolder(root, transform)
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform) #ReferenceDataset(root, transform)
    else:
        raise NotImplementedError 
    
    if  ('animal' in root) or ('af' in root) or ('food' in root):
        class_to_use = [0,1,2,3,4,5,6,7,8,9]
    
        min_data = 99999999
        max_data = 0

        # 각 image 들의 index list.
        tot_targets = torch.tensor(dataset.targets)

        train_idx = None
        
        train_class_idx = []

        #import pdb; pdb.set_trace()
        for k in class_to_use:
            #import pdb; pdb.set_trace()
            train_tmp_idx = torch.nonzero(tot_targets == k)#.nonzero()
            train_tmp_idx = train_tmp_idx[:-50]
            if k == class_to_use[0]:
                train_idx = train_tmp_idx.clone()
            else:
                train_idx = torch.cat((train_idx, train_tmp_idx))

            #import pdb; pdb.set_trace()
            train_class_idx = train_class_idx + (tot_targets[tot_targets==k].tolist()[:-50])
            if min_data > len(train_tmp_idx):
                min_data = len(train_tmp_idx)
            if max_data < len(train_tmp_idx):
                max_data = len(train_tmp_idx)

        #import pdb; pdb.set_trace()
        dataset = torch.utils.data.Subset(dataset, train_idx.squeeze())
        sampler = _make_balanced_sampler(train_class_idx)
    else:
        dataset = dataset
        sampler = _make_balanced_sampler(dataset.targets)

    """ shuffle 제외. 대신 sampler """
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           #sampler = sampler,
                           shuffle = shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)

def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
    #print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = DefaultDataset(root, transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_filePathEval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = FilePathDefaultDataset(root, transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)

def get_test_loader(root, which='source', img_size=224, batch_size=32,
                    shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
        
    if which == 'source':
        dataset = ImageFolder(root, transform)
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform) #ReferenceDataset(root, transform)
    
    #root = root.split("/")
    if ('animal' in root) or ('af' in root) or ('food' in root):
        """ animalface10, food10 을 기준으로 잡음. """
        class_to_use = [0,1,2,3,4,5,6,7,8,9]

        min_data = 99999999
        max_data = 0

        tot_targets = torch.tensor(dataset.targets)
        val_idx = None
        
        for k in class_to_use:
            val_tmp_idx = torch.nonzero(tot_targets == k)#.nonzero()
            val_tmp_idx = val_tmp_idx[-50:]
            if k == class_to_use[0]:
                val_idx = val_tmp_idx.clone()
            else:
                val_idx = torch.cat((val_idx, val_tmp_idx))
            if min_data > len(val_tmp_idx):
                min_data = len(val_tmp_idx)
            if max_data < len(val_tmp_idx):
                max_data = len(val_tmp_idx)
        dataset = torch.utils.data.Subset(dataset, val_idx.squeeze())
    else:
        dataset = dataset

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           )