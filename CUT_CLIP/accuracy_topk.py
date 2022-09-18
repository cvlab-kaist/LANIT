import os
import torch
from glob import glob
import numpy as np
from tqdm import tqdm
from copy import deepcopy

# use_base = True
# use_zerocut = True # use_base/use_zerocut: using pi domian.
# topk=1 # topk
# dataset = "animal" # food, celeb, ffhq, lsun ...
# dataset_root = "/root/project/2022_clip/af_10" # path of dataset
# sim_path = "./sw_clip_animal_10_2" # path of result directory made by running "cluster_1.py"
# n = 10 # n= 4, 7, 10 # number of domains

use_base = True
use_zerocut = True
topk=1
n = 10
dataset = "animal"
dataset_root = "/root/project/2022_clip/af_10"
sim_path = "./clip_af_{0}".format(n)

sim_perclass         = np.load( os.path.join(sim_path, "topk_sim_perclass.npy"), allow_pickle=True)
rel_sim_perclass     = np.load( os.path.join(sim_path, "topk_rel_sim_perclass.npy"), allow_pickle=True)
sim_perclass_imgpath = np.load( os.path.join(sim_path, "topk_sim_perclass_imgpath.npy"), allow_pickle=True)

rel_sim_vec = np.load( os.path.join(sim_path, "topk_rel_vector_perclass.npy"), allow_pickle=True)
sim_vec = np.load( os.path.join(sim_path, "topk_vector_perclass.npy"), allow_pickle=True)
num_data = sum( [ len(sim_perclass[0][i]) for i in range(len(sim_perclass[0])) ] )

def get_over_under_paths(num_data, norm, mean, std, prompts, sim_perclass, base_sim_perclass, sim_perclass_imgpath):
    class_prompt = prompts

    if use_base:
        top_all_vec = rel_sim_vec[0][0]
        top_all_path_vec = sim_perclass_imgpath[0][0]
        for i in range(len(class_prompt)-1):
            top_all_vec += rel_sim_vec[0][i+1] #+ sim_perclass[k][1]
            top_all_path_vec += sim_perclass_imgpath[0][i+1] #+ sim_perclass_imgpath[k][1]
    else:
        top_all_vec = sim_vec[0][0]
        top_all_path_vec = sim_perclass_imgpath[0][0]
        for i in range(len(class_prompt)-1):
            top_all_vec += sim_vec[0][i+1] #+ sim_perclass[k][1]
            top_all_path_vec += sim_perclass_imgpath[0][i+1] #+ sim_perclass_imgpath[k][1]
    
    top_all_vec = torch.Tensor(top_all_vec)
    if use_base:
        zero = torch.FloatTensor([0.])
        topk_val, topk_idx = torch.topk(top_all_vec, k=topk, dim=1) # (30000, 3)
        zerocut_idx = (topk_val <= 0) # False True
        num_data = num_data - sum(torch.sum(zerocut_idx, dim=-1)==topk)
    
    if use_zerocut:
        if use_base:
            for k in range(topk):
                topk_idx[zerocut_idx[:,k], k] = -999

    topk_clip_paths = [ [] for i in range(topk) ]
    for k in range(topk):
        for domain_idx in range(len(class_prompt)):
            topk_clip_paths[k].append( np.array(top_all_path_vec)[topk_idx[:,k] == domain_idx] )

    return topk_clip_paths,  num_data

if "animal" in dataset:
    template = 'a photo of the {}'
    base_prompt = ["a photo of the animal."]
    class_dir = [ "n02088364", "n02096437", "n02099601", "n02105162", "n02107908", "n02120079", "n02123045", "n02128757", "n02129165", "n02129604", ]
    # all_prompt = ['beagle', 'dandie dinmont terrier', 'golden retriever', 'malinois',\
    #      'appenzeller sennenhund', 'white fox', 'tabby cat', 'snow leopard', 'lion', 'bengal tiger']
    if n == 4:
        prompts_root = [ class_dir[i] for i in [0, 2, 6, 9] ]
        prompts = ['beagle', 'golden retriever','tabby cat', 'bengal tiger']
    elif n==7:
        prompts_root = [ class_dir[i] for i in [0, 1, 2, 5, 6, 7, 9] ]
        prompts = ['beagle', 'dandie dinmont terrier', 'golden retriever', 'white fox', 'tabby cat', 'snow leopard', 'bengal tiger']
    elif n == 10:
        prompts_root = [ class_dir[i] for i in range(10) ]
        prompts = ['beagle', 'dandie dinmont terrier', 'golden retriever', 'malinois',\
                        'appenzeller sennenhund', 'white fox', 'tabby cat', 'snow leopard', 'lion', 'bengal tiger']

elif "food" in dataset:
    template = 'a photo of the {}'
    base_prompt = ["a photo of the food."]

    class_dir = [ "baby_back_ribs", "beef_carpaccio", "beignets", "bibimbap", "caesar_salad",\
                            "clam_chowder", "dumplings", "edamame", "spaghetti_bolognese", "strawberry_shortcake"]
    if n == 4:
        prompts_root = [ class_dir[i] for i in [0, 2, 6, 7] ]
        prompts = ['baby back ribs', 'beignets', 'dumplings','edamame']
    elif n == 7:
        prompts_root = [ class_dir[i] for i in [0, 1, 2, 5, 6, 7, 9] ]
        prompts = ['baby back ribs','beef carpaccio','beignets','clam chowder','dumplings','edamame', 'strawberry shortcake' ]
    elif n == 10:
        prompts_root = [ class_dir[i] for i in range(10) ]
        prompts =  [ "baby back ribs", "beef carpaccio", "French beignets", "Korean bibimbap", "caesar salad",\
                            "clam chowder", "Chinese dumplings", "edamame", "spaghetti bolognese", "strawberry shortcake"]
    
elif "celeb" or "ffhq" in dataset:
    template = 'a face with {}'
    base_prompt = ["a face."]
    all_prompt = ['5 o clock shadow', 'arched eyebrows', 'attractive face', 'bags under eyes', 'bald', 'bangs', 'big lips' ,'big Nose',\
                'black hair','blond hair', 'blurry', 'brown hair', 'bushy eyebrows', 'cubby', 'double chin', 'eyeglasses', 'goatee', \
                'gray hair', 'heavy makeup', 'high cheekbones', 'male', 'mouth slightly open', 'mustache', 'narrow eyes', 'no beard', \
                'oval face', 'pale skin', 'pointy nose', 'receding hairline', 'rosy cheeks', 'sideburns', 'smiling', 'straight hair', \
                'wavy hair', 'wearing earrings', 'wearing hat', 'wearing lipstick', 'wearing necklace', 'wearing necktie', 'young'] 
    class_dir = sorted(os.listdir(dataset_root))
    if n == 4:
        prompts_root = [ class_dir[i] for i in [9, 8, 31, 15] ]
        prompts = ['blond hair', 'black hair' , 'smiling', 'eyeglasses',] 
        print()
    elif n == 7:
        prompts_root = [ class_dir[i] for i in [9, 33, 8, 31, 15, 16, 5] ]
        prompts = ['blond hair', 'wavy hair', 'black hair' , 'smiling', 'eyeglasses', 'goatee', 'bangs',]
        print()
    elif n == 10:
        prompts_root = [ class_dir[i] for i in [9, 4, 33, 8, 31, 32, 15, 16, 5, 1] ]
        prompts = ['blond hair', 'bald', 'wavy hair', 'black hair' ,\
                        'smiling', 'straight hair', 'eyeglasses', 'goatee', 'bangs', 'arched eyebrows']

prompts_root_dir = []
prompts_over_dir, num_data = get_over_under_paths(num_data, norm, mean, std, prompts, sim_perclass, rel_sim_perclass, sim_perclass_imgpath)
for i in range(len(prompts)):
    prompts_root_dir.append( os.path.join(dataset_root, prompts_root[i]) )


total_over_count_list = np.array(  [0]*len(prompts))
total_under_count_list = np.array( [0]*len(prompts) )
num_over_list = np.array(  [0]*len(prompts))
num_under_list = np.array( [0]*len(prompts))
num_gt_list = np.array(    [0]*len(prompts))

total_over_paths = [ [] for i in range(len(prompts)) ]
for k_idx in range(topk):
    for k in range(len(prompts)):
        total_over_paths[k].extend( prompts_over_dir[k_idx][k][:,0] )

for k in range(len(prompts)):
    total_over_paths[k] = list(set(total_over_paths[k]))

TP_list = []
FP_list = []
confusion_list = []
P_mean_list = []
N_mean_list = []
total_gt  = 0

for k in range(len(prompts)):
    #glob( os.path.join( prompts_root_dir[k],  "*.jpg") )
    #root_paths = [ root_paths[i].split("_")[-1] for i in range(len(root_paths)) ]
    root_paths  = os.listdir(prompts_root_dir[k])
    over_paths  = total_over_paths[k]

    num_gt = len(root_paths)
    total_gt += num_gt
    num_over = len(over_paths)

    over_count = 0

    for o in range(num_over):
        #import pdb; pdb.set_trace()
        if "animal" in dataset:
            if over_paths[o].split("/")[-1] in root_paths:
                over_count += 1
        else:
            if over_paths[o].split("/")[-1].split("_")[-1] in root_paths:
                over_count += 1

    total_over_count_list[k] = over_count
    num_over_list[k] = num_over
    num_gt_list[k] = num_gt
    

print("------------------------------Total f1 score -------------------------")
total_c_f1, total_n_f1, total_c_acc, total_n_acc = 0., 0., 0., 0.
# for k_idx in range(topk):
for p in range(len(prompts)):
    c_P, c_N = num_over_list[p], num_data - num_over_list[p]
    c_TP =  total_over_count_list[p]
    c_FP=  (c_P-c_TP)
    c_FN = (num_gt_list[p]-c_TP)
    c_TN=  c_N-c_FN

    c_pre, c_recall = c_TP/(c_TP+c_FP), c_TP/(c_TP+c_FN)
    c_f1= (2*c_pre*c_recall)/(c_pre+c_recall)

    c_acc = (c_TP+c_TN)/(c_TP+c_FP+c_TN+c_FN)

    total_c_f1 += c_f1
    total_c_acc += c_acc
    # print("TP: {0}: clip:{1}/{2}".format(prompts[p], c_TP, num_gt_list[p]))
    # print("FP: {0}: clip:{1}/{2}".format(prompts[p], c_P-c_TP, c_P))

print("Total: F1:{0}, acc:{1}".format(total_c_f1/(len(prompts)), total_c_acc/(len(prompts)) ))
