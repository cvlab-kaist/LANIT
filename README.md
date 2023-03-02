## LANIT: Language-Driven Image-to-Image Translation for Unlabeled Data (CVPR 2023)
<a href="https://arxiv.org/abs/2208.14889"><img src="https://img.shields.io/badge/arXiv-2208.14889-%23B31B1B"></a>
<a href="https://KU-CVLAB.github.io/LANIT/"><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>

<!--ECCV'22 camera ready version can be found here : [[arXiv](https://arxiv.org/abs/2207.10866)].-->

![alt text](./images/teaser_lanit.png)

> **LANIT: Language-Driven Image-to-Image Translation for Unlabeled Data**
>
> Abstract: Existing techniques for image-to-image translation commonly have suffered from two critical problems: heavy reliance on per-sample domain annotation and/or inability of handling multiple attributes per image. Recent methods adopt clustering approaches to easily provide per-sample annotations in an unsupervised manner. However, they cannot account for the real-world setting; one sample may have multiple attributes. In addition, the semantics of the clusters are not easily coupled to human understanding. To overcome these, we present a LANguage-driven Image-to-image Translation model, dubbed LANIT. We leverage easy-to-obtain candidate domain annotations given in texts for a dataset and jointly optimize them during training. The target style is specified by aggregating multi-domain style vectors according to the multi-hot domain assignments. As the initial candidate domain texts might be inaccurate, we set the candidate domain texts to be learnable and jointly fine-tune them during training. Furthermore, we introduce a slack domain to cover samples that are not covered by the candidate domains. Experiments on several standard benchmarks demonstrate that LANIT achieves comparable or superior performance to existing models.

# Level of Supervision

For unpaired image-to-image translation, (a) conventional methods (CycleGAN, MUNIT, FUNIT, StarGAN, SEMIT) require at least per-sample-level domain supervision, which is often hard to collect. To overcome this, (b) unsupervised learning methods (TUNIT, Style aware discriminator) learn image translation model using a dataset itself without any supervision, but it shows limited performance and lacks the semantic understanding of each cluster, limiting its applicability. Unlike them, (c) we present a novel framework for image translation that requires a dataset with possible textual domain descriptions (i.e., dataset-level annotation), which achieves comparable or even better performance than previous methods.

![alt text](./images/level_sup.png)

# Network Configuration

Our model LANIT is illustrated below:

![alt text](./images/network_config_lanit.png)

# Environment Settings
```
git clone https://github.com/KU-CVLAB/LANIT
cd LANIT
conda env create -f environment.yaml
conda activate lanit
```
# Preparing datasets
* Download : [CelebA-HQ (PGGAN)](https://github.com/tkarras/progressive_growing_of_gans) / [FFHQ (StyleGAN)](https://github.com/NVlabs/ffhq-dataset) / [AnimalFaces (FUNIT)](https://github.com/NVlabs/FUNIT) / [Food](https://www.kaggle.com/datasets/dansbecker/food-101) / [Lsun-Car](https://github.com/Tin-Kramberger/LSUN-Stanford-dataset) / [Lsun-Church](https://www.yf.io/p/lsun) / [LHQ (ALIS)](https://github.com/universome/alis) / [MetFace](https://github.com/NVlabs/metfaces-dataset) / [Anime](https://github.com/bchao1/Anime-Face-Dataset)
* Example directory hierarchy (CelebA-HQ, AnimalFaces, and other datasets): 
```
Project
|--- LANIT
|          |--- main.py
|          |--- core    
|                 |--- solver.py
|                 |--- data_loader.py
|          |--- shell
|
|          |--- datasets
|                 |--- CelebA-HQ
|                         |--- train
|                             |--- images
|                                   |--- 000001.jpg
|                                   |--- ...
|                         |--- test
|                             |--- images
|                                   |--- 000001.jpg
|                                   |--- ...
|                 |--- animal_faces
|                         |--- n02085620
|                         |--- n02085782
|                         |--- ...
|                 |--- ffhq, lsun-car, lsun-church, LHQ, metface, anime
|                         |--- images
|                                |--- 000001.jpg
|                                |--- ...

Then, call --train_img_dir='./datasets/CelebA-HQ/train' or './datasets/ffhq' etc.
```

# Training

## Set prompts and domains to utilize.
Please refer to **a function get_rpompt-and_att in ./core/utils.py**

If you want to use other datasets, you shold set prompt and domain to utilize.

For example, in the case of animalfaces, the code is written as below:
```
    if 'animal' in args.dataset:
        init_prompt = 'a photo of the {}.'
        base_template = ["a photo of the animal face."]
        all_prompt = ['beagle', 'dandie dinmont terrier', 'golden retriever', 'malinois', 'appenzeller sennenhund', 'white fox', 'tabby cat', 'snow leopard', 'lion', 'bengal tiger']
             
        if args.num_domains == 4:
            prompt = ['beagle', 'golden retriever','tabby cat', 'bengal tiger']
        elif args.num_domains == 7:
            prompt = ['beagle', 'dandie dinmont terrier', 'golden retriever', 'white fox', 'tabby cat', 'snow leopard', 'bengal tiger']
        elif args.num_domains == 10:
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
```

* init_prompt: prompt format to be shared by class prompts  defined by users.
* base_template: prompt that includes all the class prompts defined by users.
* prompt: the list that include class prompts defined by users.

## Training code (example of CelebA-HQ):

The example shell script is located in "./shell"

* step1 setting
```
python main.py \
--name $name \
--dataset celeb \
--mode train \
--train_img_dir ../datasets/CelebAMask-HQ/celeba \
--val_img_dir ../datasets/CelebAMask-HQ/celeba \
--checkpoint_dir ./expr/checkpoint/lanit_celeb_weight/ \
--step1 \
--num_domains 10 \
--cycle \
--ds \
--multi_hot \
--use_base \
--zero_cut \
--w_hpf 0 \
--text_aug \
--dcycle \
--base_fix \

```

* step2 setting

```
python main.py \
--name $name \
--dataset celeb \
--mode train \
--train_img_dir ../datasets/CelebAMask-HQ/celeba \
--val_img_dir ../datasets/CelebAMask-HQ/celeba \
--checkpoint_dir ./expr/checkpoint/lanit_celeb_weight/ \
--step2 \
--num_domains 10 \
--cycle \
--ds \
--multi_hot \
--use_base \
--zero_cut \
--w_hpf 0 \
--text_aug \
--dcycle \
--resume_iter 99000 \
--p_lr 1e-6 \
--gt_update \
--lambda_dc_reg 0. \

```
 
# Inference
Results are saved in the path: ./expr/results/args.name/latent(or reference).jpg

The example shell script is located in "./shell"

* Reference-guided inference code (CelebA-HQ):
```
python main.py \
--name $name \
--dataset celeb \
--mode sample \
--train_img_dir ../datasets/CelebAMask-HQ/celeba \
--val_img_dir ../datasets/CelebAMask-HQ/celeba \
--src_dir ../datasets/CelebAMask-HQ/celeba \
--ref_dir ../datasets/CelebAMask-HQ/celeba \
--checkpoint_dir [./checkpoints/args.dataset/args.name/, ex) ./checkpoints/celeb/celeb-10/] \
--step1 \
--num_domains 10 \
--cycle \
--ds \
--multi_hot \
--use_base \
--zero_cut \
--w_hpf 0 \
--text_aug \
--dcycle \
--base_fix \
--val_batch_size 8 \
--resume_iter [iteration at which the checkpoint is saved, ex) 98000] \
--infer_mode reference \
```

* Latent-guided inference code (CelebA-HQ):
```
python main.py \
--name $name \
--dataset celeb \
--mode sample \
--train_img_dir ../datasets/CelebAMask-HQ/celeba \
--val_img_dir ../datasets/CelebAMask-HQ/celeba \
--src_dir ../datasets/CelebAMask-HQ/celeba \
--ref_dir ../datasets/CelebAMask-HQ/celeba \
--checkpoint_dir [./checkpoints/args.dataset/args.name/, ex) ./checkpoints/celeb/celeb-10/] \
--step1 \
--num_domains 10 \
--cycle \
--ds \
--multi_hot \
--use_base \
--zero_cut \
--w_hpf 0 \
--text_aug \
--dcycle \
--base_fix \
--val_batch_size 8 \
--resume_iter [iteration at which the checkpoint is saved, ex) 98000] \
--infer_mode latent\
--latent_num 0 1 2\ # latent_num is the list that includes the attributes
```

### BibTeX
If you find this research useful, please consider citing:
````BibTeX
@article{park2022lanit,
  title = {LANIT: Language-Driven Image-to-Image Translation for Unlabeled Data},
  author = {Park, Jihye and Kim, Sunwoo and Kim, Soohyun and Cho, seokju and Yoo, Jaejun and Uh, Youngjung and Kim, Seungryong},
  journal = {arXiv preprint arXiv:2208.14889},
  year = {2022},
}
````
