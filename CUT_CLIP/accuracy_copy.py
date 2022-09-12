import os
import torch
from glob import glob
import numpy as np
from tqdm import tqdm

# dataset = "animal" # food, celeb, ffhq, lsun ...
# dataset_root = "/root/project/2022_clip/af_10" # "/root/project/2022_clip/celeba_per"
# sim_path = "./sw_clip_animal_10_2"
# n = 10 # n= 4, 7, 10
# threshold = 5.6#7.5411/1.8190

dataset = "celeb" # dataset종류:  food, celeb, ffhq, lsun ...
dataset_root = "/root/project/2022_clip/celeba_per" # dataset 경로
sim_path = "./sw_clip_celeb_10_2" # 저장된 경로.
n = 10 # n= 4, 7, 10
#threshold = 0. #5.4741-0.5862#5.6#7.5411/1.8190 # 3

sim_perclass         = np.load( os.path.join(sim_path, "topk_sim_perclass.npy"), allow_pickle=True)
rel_sim_perclass     = np.load( os.path.join(sim_path, "topk_rel_sim_perclass.npy"), allow_pickle=True)
sim_perclass_imgpath = np.load( os.path.join(sim_path, "topk_sim_perclass_imgpath.npy"), allow_pickle=True)


def get_over_under_paths(threshold, norm, prompts, sim_perclass, base_sim_perclass, sim_perclass_imgpath):
    class_prompt = prompts

    over_zero_path = []
    under_zero_path = []

    for k in tqdm(range(len(prompts))):
        norm_val = norm[k]
        top_all = rel_sim_perclass[k][0]
        top_all_path = sim_perclass_imgpath[k][0]
        for i in range(len(class_prompt)-1):
            top_all += rel_sim_perclass[k][i+1] #+ sim_perclass[k][1]
            top_all_path += sim_perclass_imgpath[k][i+1] #+ sim_perclass_imgpath[k][1]

        #try:
        top_all = torch.Tensor(top_all)

        """ over zero """
        over_zero_idx = torch.where( (top_all)> norm_val )[0]
        top_all_over = top_all[over_zero_idx]

        top100_paths_over = np.array(top_all_path)[over_zero_idx]
        top_all_over, topk_idx = torch.sort(top_all_over, descending=True)
        over_zero_path.append( top100_paths_over[topk_idx] )

        """ under zero """
        under_zero_idx = torch.where((top_all)< norm_val )[0]
        top_all_under = top_all[under_zero_idx]

        top100_paths_under = np.array(top_all_path)[under_zero_idx]
        top_all_under, topk_idx = torch.sort(top_all_under, descending=True)
        under_zero_path.append( top100_paths_under[topk_idx] )

    return over_zero_path, under_zero_path


if "animal" in dataset:
    template = 'a photo of the {}'
    base_prompt = ["a photo of the animal."]
    class_dir = [ "n02088364", "n02096437", "n02099601", "n02105162", "n02107908",
                            "n02120079", "n02123045", "n02128757", "n02129165", "n02129604", ]
    # all_prompt = ['beagle', 'dandie dinmont terrier', 'golden retriever', 'malinois',\
    #      'appenzeller sennenhund', 'white fox', 'tabby cat', 'snow leopard', 'lion', 'bengal tiger']
    if n == 4:
        # af4
        prompts_root = [ class_dir[i] for i in [0, 2, 6, 9] ]
        prompts = ['beagle', 'golden retriever','tabby cat', 'bengal tiger']
    elif n==7:
        # af7
        prompts_root = [ class_dir[i] for i in [0, 1, 2, 5, 6, 7, 9] ]
        prompts = ['beagle', 'dandie dinmont terrier', 'golden retriever', 'white fox', 'tabby cat', 'snow leopard', 'bengal tiger']
    elif n == 10:
        # af10
        prompts_root = [ class_dir[i] for i in range(10) ]
        #norm = np.array([0,0,0,0,0,0,0,0,0,0])
        # np.array( [1.6287, 1.1542, 0.9306, 0.0000, 1.5250, 0.0878, 3.2473, 2.1078, 3.9066,2.8960] )
        #norm = np.array([1.1266, 1.0973, 1.1138, 1.0441, 1.1251, 1.0000, 1.1451, 1.0403, 1.1221, 1.0833])
        #mean = np.array([7.2861, 7.7606, 7.9842, 8.9148, 7.3898, 8.8269, 5.6674, 6.8070, 5.0082,6.0188])
        #var = np.array([[0.5038, 1.0706, 0.7352, 0.9962, 0.7936, 0.5598, 0.3845, 0.6442, 0.3875,0.5180]])

        prompts = ['beagle', 'dandie dinmont terrier', 'golden retriever', 'malinois',\
                        'appenzeller sennenhund', 'white fox', 'tabby cat', 'snow leopard', 'lion', 'bengal tiger']

elif "food" in dataset:
    template = 'a photo of the {}'
    base_prompt = ["a photo of the food."]

    class_dir = [ "baby_back_ribs", "beef_carpaccio", "beignets", "bibimbap", "caesar_salad",\
                            "clam_chowder", "dumplings", "edamame", "spaghetti_bolognese", "strawberry_shortcake"]
    if n == 4:
        # food 
        prompts_root = [ class_dir[i] for i in [0, 2, 6, 7] ]
        prompts = ['baby back ribs', 'beignets', 'dumplings','edamame']
    elif n == 7:
        # food 7
        prompts_root = [ class_dir[i] for i in [0, 1, 2, 5, 6, 7, 9] ]
        norm = np.array([3.6262, 5.0826, 5.6407, 5.5700, 2.9986, 5.7678, 3.5004])
        prompts = ['baby back ribs','beef carpaccio','beignets','clam chowder','dumplings','edamame', 'strawberry shortcake' ]
    elif n == 10:
        # food 10
        prompts_root = [ class_dir[i] for i in range(10) ]
        #norm = np.array([3.5901, 5.0876, 5.5116, 5.1071, 3.8063, 4.5887, 3.1261, 5.8397, 4.2568, 3.4980])
        norm = np.array([0.8292, 0.6253, 0.5845, 0.5356, 1.0000, 0.5777, 0.9225, 0.5565, 0.7241, 0.8462]) * np.array([4.5901, 6.0876, 6.5116, 7.1071, 3.8063, 6.5887, 4.1261, 6.8397, 5.2568,4.4980])
        prompts =  [ "baby back ribs", "beef carpaccio", "French beignets", "Korean bibimbap", "caesar salad",\
                            "clam chowder", "Chinese dumplings", "edamame", "spaghetti bolognese", "strawberry shortcake"]
    
elif "celeb" or "ffhq" in dataset:
    template = 'a face with {}'
    base_prompt = ["a face."]
    # all_prompt = ['5 o clock shadow', 'arched eyebrows', 'attractive face', 'bags under eyes', 'bald', 'bangs', 'big lips' ,'big Nose',\
    #             'black hair','blond hair', 'blurry', 'brown hair', 'bushy eyebrows', 'cubby', 'double chin', 'eyeglasses', 'goatee', \
    #             'gray hair', 'heavy makeup', 'high cheekbones', 'male', 'mouth slightly open', 'mustache', 'narrow eyes', 'no beard', \
    #             'oval face', 'pale skin', 'pointy nose', 'receding hairline', 'rosy cheeks', 'sideburns', 'smiling', 'straight hair', \
    #             'wavy hair', 'wearing earrings', 'wearing hat', 'wearing lipstick', 'wearing necklace', 'wearing necktie', 'young'] 
    class_dir = sorted(os.listdir(dataset_root))
    if n == 4:
        # celeb4
        # class_prompt = awfawfawfawawfawf
        print()
    elif n == 7:
        # celeb7
        # class_prompt =
        print()
    elif n == 10:
        # celeb10
        #norm = np.array([1,1,1,1,1,1,1,1,1,1])
        #norm = np.array([0,0,0,0,0,0,0,0,0,0])
        
        #norm = np.array([22.0829, 19.9571, 21.9861, 20.5855, 22.7017, 21.2946, 20.2298, 20.1588, 19.4741, 23.4727])
        norm = np.array( [2.7094, 1.2937, 1.8249, 1.6485, 2.0895, 1.5358, 1.2121, 1.7531, 1.9388, 2.3101] )-0.5
        #norm = np.array( [2.7094, 1.2937, 1.8249, 1.6485, 2.0895, 1.5358, 1.2121, 1.7531, 1.9388, 2.3101] )* np.array([0.4474, 0.9369, 0.6642, 0.7353, 0.5801, 0.7892, 1.0000, 0.6914, 0.6252, 0.5247])
        #norm = np.array( [2.7094, 1.2937, 1.8249, 1.6485, 2.0895, 1.5358, 1.2121, 1.7531, 1.9388, 2.3101] )*np.array([1.0000, 2.0943, 1.4847, 1.6436, 1.2967, 1.7642, 2.2353, 1.5455, 1.3975, 1.1728])
        prompts_root = [ class_dir[i] for i in [9, 4, 33, 8, 31, 32, 15, 16, 5, 1] ]
        prompts = ['blond hair', 'bald', 'wavy hair', 'black hair' ,\
                        'smiling', 'straight hair', 'eyeglasses', 'goatee', 'bangs', 'arched eyebrows']


prompts_root_dir = []
prompts_over_dir, prompts_under_dir = get_over_under_paths(threshold, norm, prompts, sim_perclass, rel_sim_perclass, sim_perclass_imgpath)
for i in range(len(prompts)):
    prompts_root_dir.append( os.path.join(dataset_root, prompts_root[i]) )
    # prompts_over_dir.append( os.path.join(over_zero_dir, prompts[i]) )
    # prompts_under_dir.append( os.path.join(under_zero_dir, prompts[i]) )
print(prompts_root_dir)

TP_list = []
FP_list = []
P_mean_list = []
N_mean_list = []
for k in range(len(prompts)):
    root_paths  = os.listdir(prompts_root_dir[k])  #glob( os.path.join( prompts_root_dir[k],  "*.jpg") )
    #root_paths = [ root_paths[i].split("_")[-1] for i in range(len(root_paths)) ]
    over_paths  = prompts_over_dir[k]  #glob( os.path.join( prompts_over_dir[k],  "*.jpg") )
    under_paths = prompts_under_dir[k] #glob( os.path.join( prompts_under_dir[k], "*.jpg") )

    num_gt = len(root_paths)
    num_over = len(over_paths)
    num_under = len(under_paths)

    over_count = 0
    under_count = 0
    #import pdb; pdb.set_trace()
    for o in range(num_over):
        #import pdb; pdb.set_trace()
        if "animal" in dataset:
            if over_paths[o][0].split("/")[-1] in root_paths:
                over_count += 1
        else:
            if over_paths[o][0].split("/")[-1].split("_")[-1] in root_paths:
                over_count += 1
    for u in range(num_under):
        if "animal" in dataset:
            if under_paths[u][0].split("/")[-1] in root_paths:
                under_count += 1
        else:
            if under_paths[u][0].split("/")[-1].split("_")[-1] in root_paths:
                under_count += 1
    
    TP_list.append("{0}: over:{1}/{2}, under:{3}/{4}".format(prompts[k], over_count, num_gt, under_count, num_gt))
    FP_list.append("{0}: over:{1}/{2}".format(prompts[k], num_over-over_count, num_over))

print("------------------------------ TP&FN -------------------------")
for k in range(len(prompts)):
    print(TP_list[k])
print("------------------------------ FP ---------------------------")
for k in range(len(prompts)):
    print(FP_list[k])

    