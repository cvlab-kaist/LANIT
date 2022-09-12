## LANIT: Language-Driven Image-to-Image Translation for Unlabeled Data
Check out project [[Project Page](https://KU-CVLAB.github.io/LANIT/)] and the paper on [[arXiv](https://arxiv.org/abs/2208.14889)].
We will update codes and pretrained weights soon.

<!--ECCV'22 camera ready version can be found here : [[arXiv](https://arxiv.org/abs/2207.10866)].-->

![alt text](./images/teaser_lanit.png)

> **LANIT: Language-Driven Image-to-Image Translation for Unlabeled Data**
>
> Abstract: Existing techniques for image-to-image translation commonly have suffered from two critical problems: heavy reliance on per-sample domain annotation and/or inability of handling multiple attributes per image. Recent methods adopt clustering approaches to easily provide per-sample annotations in an unsupervised manner. However, they cannot account for the real-world setting; one sample may have multiple attributes. In addition, the semantics of the clusters are not easily coupled to human understanding. To overcome these, we present a LANguage-driven Image-to-image Translation model, dubbed LANIT. We leverage easy-to-obtain candidate domain annotations given in texts for a dataset and jointly optimize them during training. The target style is specified by aggregating multi-domain style vectors according to the multi-hot domain assignments. As the initial candidate domain texts might be inaccurate, we set the candidate domain texts to be learnable and jointly fine-tune them during training. Furthermore, we introduce a slack domain to cover samples that are not covered by the candidate domains. Experiments on several standard benchmarks demonstrate that LANIT achieves comparable or superior performance to existing models.

# Network Configuration

Our model LANIT is illustrated below:

![alt text](./images/network_config_lanit.png)

# Environment Settings
```
git clone https://github.com/KU-CVLAB/LANIT

cd LANIT

conda env create -f environment.yaml
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

Training code(example of CelebA-HQ):
```
CUDA_VISIBLE_DEVICES=0 python main.py\
--name celeb-10\
--dataset celeb\
--mode train\
--train_img_dir ./dataset/CelebA-HQ/train\
--val_img_dir ./dataset/CelebA-HQ/test\
--checkpoint_dir ./checkpoints/lanit_celeb_weight/\
--step1\
--num_domains 10\
--cycle\
--dc\
--ds\
--multi_hot\
--topk 3\
--use_base\
--zero_cut\
```

* In step1(don't do prompt learning): num_domain=10, topk=3, use_all_losses(cycle,dc,ds), use_phi_domain(use_base, zero_cut)
* In step2: change --step1 to --step2, add --use_prompt
```
--num_domain: number of domain that we want to consider  
 --topk: number of multi-attribute that we want to consider in an image. (In this paper, CelebA-HQ=3, AnimalFaces,Food=1)  
 --use_prompt: get PromptLearner from ./core/model/
 ```
 
# Inference

* Reference-guided inference code(CelebA-HQ):
```
CUDA_VISIBLE_DEVICES=0 python main.py\
--name celeb-10\
--dataset celeb\
--mode sample\
--infer_mode reference\
--src_dir ./dataset/CelebA-HQ/train\
--ref_dir ./dataset/CelebA-HQ/test\
--checkpoint_dir ./checkpoints/lanit_celeb_weight/\
--step1\
--num_domains 10\
--multi_hot\
--topk 3\
--use_base\
--zero_cut\
```

* Latent-guided inference code(CelebA-HQ):
```
CUDA_VISIBLE_DEVICES=0 python main.py\
--name celeb-10\
--dataset celeb\
--mode sample\
--infer_mode latent\
--latent_num [0,1,2]\
--src_dir ./dataset/CelebA-HQ/train\
--ref_dir ./dataset/CelebA-HQ/test\
--checkpoint_dir ./checkpoints/lanit_celeb_weight/\
--step1\
--num_domains 10\
--multi_hot\
--topk 3\
--use_base\
--zero_cut\
```

### BibTeX
If you find this research useful, please consider citing:
````BibTeX
@article{park2022lanit,
  title = {LANIT: Language-Driven Image-to-Image Translation for Unlabeled Data},
  author = {Park, Jihye and Kim, Soohyun and Kim, Sunwoo and Yoo, Jaejun and Uh, Youngjung and Kim, Seungryong},
  journal = {arXiv preprint arXiv:2208.14889},
  year = {2022},
}
````
