import os.path
from data.base_dataset import BaseDataset, get_transform
#from data.image_folder import make_init_dataset
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import numpy as np
import torch
import pdb

import os


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseDataset.__init__(self, opt)
        #self.dir_A = os.path.join(opt.dataroot, opt.phase + 'night')  # create a path '/path/to/data/trainA'
        #self.dir_B = os.path.join(opt.dataroot, opt.phase + 'sunny')  # create a path '/path/to/data/trainB'
        
        # self.dir_A = '/home/hamacojr/lunit/af_10' #'/home/cvlab04/project/i2i/CUT_CLIP/results/cloudy'
        #self.dir_A = '/home/cvlab06/project/i2i/food-10'
        """ 여기 데이터셋 부분 바꿔주면 됨 """
        self.dataset = "ffhq"

        if ("celeb" in self.dataset) or ("ffhq" in self.dataset):
            self.dir_A = opt.dir_A#"/root/project/2022_clip/CelebAMask-HQ/CelebA-HQ-img/"
            self.ffhq_anno_path = opt.anno_path #"/root/project/2022_clip/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt"
            self.annotations, self.selected_attrs = self.load_annotations()
            self.img_list = sorted(self.annotations.keys())
            #import pdb; pdb.set_trace()
        else:
            # animal, food ...
            self.dir_A = opt.dir_A#'/home/cvlab06/project/i2i/food-10'
        
        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'

        self.A_size = len(self.A_paths)  # get the size of dataset A


    def load_annotations(self, selected_attrs=None):
        file = self.ffhq_anno_path
        lines = open(file).readlines()
        '''
        202599
        Attribute names
        000001.jpg -1  1  1 -1 -1 -1 -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1 -1  1  1 -1  1 -1 -1  1 -1 -1  1 -1 -1 -1  1  1 -1  1 -1  1 -1 -1  1
        ...
        
        selected_attrs = ['5_o_Clock_Shadow', 'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Male', 'Mustache', 'Pale_Skin', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat', 'Young']

        selected_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                            'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                            'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
                            'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
                            'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
                            'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
        '''
        
        # selected_attrs =  ['black hair', 'blond hair', 'brwon hair', 'eyeglasses', 'mustache',\
               #   'smiling', 'young', 'old', 'male', 'female'\
                #] 
        selected_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                            'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                            'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
                            'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
                            'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
                            'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
        attrs = lines[1].split()
        if selected_attrs is None:
            selected_attrs = attrs
        selected_attrs_idx = [attrs.index(a) for a in selected_attrs]
        #import pdb; pdb.set_trace()

        annotations = {}
        #annotations = []
        for line in lines[2:]:
            #import pdb;pdb.set_trace()
            tokens = line.split()
            file = tokens[0]
            anno = [(int(t)+1)/2 for t in tokens[1:]]
            anno = [anno[idx] for idx in selected_attrs_idx]
            annotations[file] = anno
            #annotations.append(anno)
        return annotations, selected_attrs

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        if ("celeb" in self.dataset) or ("ffhq" in self.dataset):
            idx = index % self.A_size
            multi_label = self.annotations[self.img_list[idx]] #(0,0,0,1,0,1,0,1)
            #print(multi_label)
            multi_arg = torch.LongTensor(np.where( np.array(multi_label)==1. ))
            #print(multi_arg)
            A_path = self.A_paths[idx]
        else:
            A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
            multi_arg = None


        A_img = Image.open(A_path).convert('RGB')

        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        #print("modified_opt: ", modified_opt)
        
        transform = get_transform(modified_opt)
        A = transform(A_img)
        #B = transform(B_img)
        
        return {'A': A, 'A_paths': A_path, 'arg':multi_arg, 'label':multi_label}


    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.A_size #, self.B_size)
