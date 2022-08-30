## LANIT: Language-Driven Image-to-Image Translation for Unlabeled Data
Check out project [[Project Page](https://KU-CVLAB.github.io/LANIT/)] and the paper on [[arXiv](https://arxiv.org/list/cs.CV/recent)].
We will update codes and pretrained weights will be ready before the conference. 

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
git clone https://github.com/sunwoo76/LANIT

cd LANIT

conda env create -f environment.yaml
```

# Training

Training code:

      to be realised soon... #python train.py 

# Inference

Inference code:

      to be realised soon... #python test.py 


# Acknowledgement <a name="Acknowledgement"></a>
.

### BibTeX
If you find this research useful, please consider citing:
````BibTeX
.
````
