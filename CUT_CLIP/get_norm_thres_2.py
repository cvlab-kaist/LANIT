import numpy as np
import os
import cv2
from tqdm import tqdm
import torch

use_base = False

dataset = "celeb" # animal, food, celeb : 데이터셋 이름.
n = 10# n= 4, 7, 10 (domain 개수.)
topk=300

sim_path = "./clip_{0}_{1}_PL8000".format(dataset,n)#_step2_84000" # 해당 npy 파일들이 저장될 경로.
save_root = "./topk300" # topk들의 이미지가 저장될 경로.
sim_perclass         = np.load( os.path.join(sim_path, "topk_sim_perclass.npy"), allow_pickle=True)
rel_sim_perclass     = np.load( os.path.join(sim_path, "topk_rel_sim_perclass.npy"), allow_pickle=True)
sim_perclass_imgpath = np.load( os.path.join(sim_path, "topk_sim_perclass_imgpath.npy"), allow_pickle=True)


if "animal" in dataset:
    template = 'a photo of the {}'
    base_prompt = ["a photo of the animal."]
    if n == 4:
        # af4
        class_prompt = ['beagle', 'golden retriever','tabby cat', 'bengal tiger']
    elif n==7:
        # af7
        class_prompt = ['beagle', 'dandie dinmont terrier', 'golden retriever', 'white fox', 'tabby cat', 'snow leopard', 'bengal tiger']
    elif n == 10:
        # af10
        class_prompt = ['beagle', 'dandie dinmont terrier', 'golden retriever', 'malinois',\
                        'appenzeller sennenhund', 'white fox', 'tabby cat', 'snow leopard', 'lion', 'bengal tiger']

elif "food" in dataset:
    template = 'a photo of the {}'
    base_prompt = ["a photo of the food."]
    if n == 4:
        # food 4
        class_prompt = ['baby back ribs', 'beignets', 'dumplings','edamame']
    elif n == 7:
        # food 7
        class_prompt = ['baby back ribs','beef carpaccio','beignets','clam chowder','dumplings','edamame', 'strawberry shortcake' ]
    elif n == 10:
        # food 10
        class_prompt =  [ "baby back ribs", "beef carpaccio", "French beignets", "Korean bibimbap", "caesar salad",\
                            "clam chowder", "Chinese dumplings", "edamame", "spaghetti bolognese", "strawberry shortcake"]
    
elif "celeb" or "ffhq" in dataset:
    template = 'a face with {}'
    base_prompt = ["a face."]
    if n == 4:
        # celeb4
        class_prompt = ['blond hair', 'black hair' , 'smiling', 'eyeglasses',] 
        print()
    elif n == 7:
        # celeb7
        class_prompt = ['blond hair', 'wavy hair', 'black hair' , 'smiling', 'eyeglasses', 'goatee', 'bangs',]
        print()
    elif n == 10:
        # celeb10
        class_prompt = ['blond hair', 'bald', 'wavy hair', 'black hair' ,\
                        'smiling', 'straight hair', 'eyeglasses', 'goatee', 'bangs', 'arched eyebrows']


for k in range(len(class_prompt)):
     os.makedirs( os.path.join(save_root, "{0}".format(class_prompt[k])), exist_ok=True)

topk_mean_list = []
overzero_mean_list = []

topk_std_list = []
for k in tqdm(range(len(class_prompt))):
    
    if use_base:
        top_all = rel_sim_perclass[k][0]
        top_all_path = sim_perclass_imgpath[k][0]
        for i in range(len(class_prompt)-1):
            top_all += rel_sim_perclass[k][i+1] #+ sim_perclass[k][1]
            top_all_path += sim_perclass_imgpath[k][i+1] #+ sim_perclass_imgpath[k][1]
    else:
        top_all = sim_perclass[k][0]
        top_all_path = sim_perclass_imgpath[k][0]
        for i in range(len(class_prompt)-1):
            top_all += sim_perclass[k][i+1] #+ sim_perclass[k][1]
            top_all_path += sim_perclass_imgpath[k][i+1] #+ sim_perclass_imgpath[k][1]

    top_all = torch.Tensor(top_all)
    """ topk의 경우 """
    topk_idx = torch.topk(top_all, k=topk)[1] # torch.where(top_all>0)

    top_all_topk = top_all[topk_idx]
    top100_paths_topk = np.array(top_all_path)[topk_idx]

    top_all_topk, topk_idx = torch.sort(top_all_topk, descending=True)
    top100_paths_topk = top100_paths_topk[topk_idx]

    """ over_zero """
    over_zero_idx = torch.where(top_all>0)

    top_all_overzero = top_all[over_zero_idx]
    top100_paths_overzero = np.array(top_all_path)[over_zero_idx]

    top_all_overzero, overzero_idx = torch.sort(top_all_overzero, descending=True)

    top100_paths_overzero = top100_paths_overzero[overzero_idx]

    """ top300 similarity mean """
    mean_val_topk = torch.mean(top_all_topk)
    mean_val_overzero = torch.mean(top_all_overzero)

    topk_mean_list.append(mean_val_topk)
    topk_std_list.append( torch.sqrt(torch.var(top_all_topk)) )
    overzero_mean_list.append(mean_val_overzero)

# """ 이미지 저장 """
# for tm in tqdm(range( len(topk_idx) )):
#     #import pdb; pdb.set_trace()
#     filepath = top100_paths_topk[tm][0]
#     filename = os.path.basename(filepath)
#     img = cv2.imread(filepath)
#     img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
#     #import pdb; pdb.set_trace()
#     cv2.imwrite( os.path.join(save_root, "./{0}/{1}_{2:.3f}_{3}".format(class_dir[k], tm, top_all_topk[tm], filename)), img  )

""" norm, threshold 출력 """
topk_mean_tensor = torch.Tensor(topk_mean_list)
topk_std_tensor = torch.Tensor(topk_std_list)
overzero_mean_tensor = torch.Tensor(overzero_mean_list)
print("norm=torch.{}".format( max(topk_mean_tensor)/topk_mean_tensor ))
print("thres=torch.{}".format(overzero_mean_tensor) )
print("mean=torch.{}".format(topk_mean_tensor) )
print("std=torch.{}".format(topk_std_tensor) )