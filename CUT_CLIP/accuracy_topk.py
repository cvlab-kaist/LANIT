import os
import torch
from glob import glob
import numpy as np
from tqdm import tqdm
from copy import deepcopy
# dataset = "animal" # food, celeb, ffhq, lsun ...
# dataset_root = "/root/project/2022_clip/af_10" # "/root/project/2022_clip/celeba_per"
# sim_path = "./sw_clip_animal_10_2"
# n = 10 # n= 4, 7, 10
# threshold = 5.6#7.5411/1.8190


# dataset = "animal" # dataset종류:  food, celeb, ffhq, lsun ...
# dataset_root = "/root/project/2022_clip/af_10" # dataset 경로
# sim_path = "./clip_af_{0}".format(n) # 저장된 경로.

# dataset = "food" # dataset종류:  food, celeb, ffhq, lsun ...
# dataset_root = "/root/project/2022_clip/food" # dataset 경로
# sim_path = "./clip_food_{0}".format(n) # 저장된 경로.

# use_base = True
# use_zerocut = True
# topk=1
# n = 10 # n= 4, 7, 10
# threshold = 0. #5.4741-0.5862#5.6#7.5411/1.8190 # 3
# dataset = "animal" # dataset종류:  food, celeb, ffhq, lsun ...
# dataset_root = "/root/project/datasets/af_10" # dataset 경로
# sim_path = "./clip_af_{0}_PL44000".format(n) # 저장된 경로.

# use_base = True
# use_zerocut = True
# topk=1
# n = 10 # n= 4, 7, 10
# threshold = 0. #5.4741-0.5862#5.6#7.5411/1.8190 # 3
# dataset = "food" # dataset종류:  food, celeb, ffhq, lsun ...
# dataset_root = "/root/project/datasets/food" # dataset 경로
# sim_path = "./clip_food_{0}_PL74000".format(n) # 저장된 경로.

use_base = True
use_zerocut = True
topk=3
n = 7 # n= 4, 7, 10
threshold = 0. #5.4741-0.5862#5.6#7.5411/1.8190 # 3
dataset = "celeb" # dataset종류:  food, celeb, ffhq, lsun ...
dataset_root = "/root/project/datasets/celeba_per" # dataset 경로
sim_path = "./clip_celeb_{0}_PL091000".format(n) # 저장된 경로.

sim_perclass         = np.load( os.path.join(sim_path, "topk_sim_perclass.npy"), allow_pickle=True)
rel_sim_perclass     = np.load( os.path.join(sim_path, "topk_rel_sim_perclass.npy"), allow_pickle=True)
sim_perclass_imgpath = np.load( os.path.join(sim_path, "topk_sim_perclass_imgpath.npy"), allow_pickle=True)

rel_sim_vec = np.load( os.path.join(sim_path, "topk_rel_vector_perclass.npy"), allow_pickle=True)
sim_vec = np.load( os.path.join(sim_path, "topk_vector_perclass.npy"), allow_pickle=True)
num_data = sum( [ len(sim_perclass[0][i]) for i in range(len(sim_perclass[0])) ] )

def get_over_under_paths(num_data, norm, mean, std, prompts, sim_perclass, base_sim_perclass, sim_perclass_imgpath):
    class_prompt = prompts

    """ similarity vector를 모두 가져온다. """
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
   
    
    top_all_path_vec_norm = deepcopy(top_all_path_vec)
    

    """ 기존의 similarity vector들 """
    top_all_vec = torch.Tensor(top_all_vec)
    top_all_vec_norm = deepcopy(top_all_vec)
    

    """ usebase에서 0으로 만드는부분 까지 추가한다면? """
    if use_base:
        zero = torch.FloatTensor([0.])
        #minus = torch.FloatTensor([-999.])
        #top_all_vec_norm = torch.where(top_all_vec_norm >= zero, top_all_vec_norm, zero ) 

        """ zerocut용 top1을 구해보자. """
        topk_val_norm_zerocut, topk_idx_norm_zerocut = torch.topk(top_all_vec_norm, k=topk, dim=1) # (30000, 3)
        zerocut_idx = (topk_val_norm_zerocut <= 0) # False True
        num_data = num_data - sum(torch.sum(zerocut_idx, dim=-1)==topk)
        print("num_data: ", num_data)
    #import pdb; pdb.set_trace()

    """ norm을 곱한후 similarit vector들 """
    top_all_vec_norm = top_all_vec_norm * norm.unsqueeze(dim=0)

    """ 여기서 다시 topk을 구해보자. """
    topk_val, topk_idx = torch.topk(top_all_vec, k=topk, dim=1)
    topk_val_norm, topk_idx_norm = torch.topk(top_all_vec_norm, k=topk, dim=1) # (30000,3)
    
    #import pdb; pdb.set_trace()
    """ zerocut """
    if use_zerocut:
        if use_base:
            for k in range(topk):
                topk_idx[zerocut_idx[:,k], k] = -999
                topk_idx_norm[zerocut_idx[:,k], k] = -999

    #import pdb; pdb.set_trace()
    topk_clip_paths = [ [] for i in range(topk) ]
    topk_norm_paths = [ [] for i in range(topk) ]
    for k in range(topk):
        #clip_paths = list()
        #norm_paths = list()
        #import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()
        for domain_idx in range(len(class_prompt)):
            topk_clip_paths[k].append( np.array(top_all_path_vec)[topk_idx[:,k] == domain_idx] )
            topk_norm_paths[k].append( np.array(top_all_path_vec_norm)[topk_idx_norm[:,k] == domain_idx] )
        #topk_clip_paths.append(clip_paths)
        #topk_norm_paths.append(norm_paths)
    return topk_clip_paths, topk_norm_paths, num_data

if "animal" in dataset:
    template = 'a photo of the {}'
    base_prompt = ["a photo of the animal."]
    class_dir = [ "n02088364", "n02096437", "n02099601", "n02105162", "n02107908", "n02120079", "n02123045", "n02128757", "n02129165", "n02129604", ]
    # all_prompt = ['beagle', 'dandie dinmont terrier', 'golden retriever', 'malinois',\
    #      'appenzeller sennenhund', 'white fox', 'tabby cat', 'snow leopard', 'lion', 'bengal tiger']
    if n == 4:
        # af4
        if use_base:
            norm=torch.Tensor([1.0938, 1.0000, 1.4070, 1.3208])
            thres=torch.Tensor([3.3959, 3.4321, 2.9552, 3.4636])
            mean=torch.Tensor([7.2840, 7.9668, 5.6624, 6.0317])
            std=torch.Tensor([0.7347, 0.8803, 0.6532, 0.7080])
        else:
            norm=torch.Tensor([1.0434, 1.0299, 1.0598, 1.0000])
            thres=torch.Tensor([22.1315, 22.9216, 23.2281, 22.6498])
            mean=torch.Tensor([31.1761, 31.5837, 30.6950, 32.5295])
            std=torch.Tensor([0.5785, 0.5690, 0.5541, 0.4711])
        prompts_root = [ class_dir[i] for i in [0, 2, 6, 9] ]
        prompts = ['beagle', 'golden retriever','tabby cat', 'bengal tiger']
    elif n==7:
        # af7
        if use_base:
            norm=torch.Tensor([1.2102, 1.1394, 1.1047, 1.0000, 1.5533, 1.2966, 1.4583])
            thres=torch.Tensor([3.4056, 4.8593, 3.4502, 4.3200, 2.9419, 4.6056, 3.4650])
            mean=torch.Tensor([7.2788, 7.7314, 7.9737, 8.8088, 5.6710, 6.7935, 6.0403])
            std=torch.Tensor([0.7322, 1.0144, 0.8770, 0.7602, 0.6296, 0.8184, 0.7117])
        else:
            norm=torch.Tensor([1.1248, 1.0994, 1.1113, 1.0000, 1.1431, 1.0384, 1.0786])
            thres=torch.Tensor([22.1342, 18.4878, 22.9184, 22.8871, 23.2362, 20.3108, 22.6570])
            mean=torch.Tensor([31.1968, 31.9179, 31.5747, 35.0904, 30.6985, 33.7915, 32.5335])
            std=torch.Tensor([0.5757, 0.8883, 0.5832, 0.7672, 0.5660, 0.6549, 0.4543])
        prompts_root = [ class_dir[i] for i in [0, 1, 2, 5, 6, 7, 9] ]
        prompts = ['beagle', 'dandie dinmont terrier', 'golden retriever', 'white fox', 'tabby cat', 'snow leopard', 'bengal tiger']
    elif n == 10:
        # af10
        if use_base:
            norm=torch.Tensor([1.2241, 1.1400, 1.1170, 1.0000, 1.2093, 1.0075, 1.5652, 1.3098, 1.7823,1.4773])
            thres=torch.Tensor([3.3952, 4.8875, 3.4263, 5.3992, 3.0236, 4.3157, 2.9382, 4.6172, 2.6932,3.4536])
            mean=torch.Tensor([7.2757, 7.8124, 7.9732, 8.9063, 7.3648, 8.8397, 5.6900, 6.7995, 4.9972,6.0287])
            std=torch.Tensor([0.7079, 1.0205, 0.8534, 0.9863, 0.8658, 0.7466, 0.6241, 0.7879, 0.6264,0.6983])
        else:
            norm=torch.Tensor([1.1252, 1.1003, 1.1109, 1.0454, 1.1272, 1.0000, 1.1428, 1.0394, 1.1185,1.0788])
            thres=torch.Tensor([22.1389, 18.4968, 22.9203, 21.1564, 23.6689, 22.8917, 23.2350, 20.3092,23.6181, 22.6553])
            mean=torch.Tensor([31.1951, 31.9007, 31.5974, 33.5756, 31.1408, 35.1015, 30.7148, 33.7696,31.3834, 32.5366])
            std=torch.Tensor([0.5869, 0.8923, 0.5809, 0.8500, 0.8497, 0.7624, 0.5611, 0.6317, 0.5027,0.4530])
        prompts_root = [ class_dir[i] for i in range(10) ]
        prompts = ['beagle', 'dandie dinmont terrier', 'golden retriever', 'malinois',\
                        'appenzeller sennenhund', 'white fox', 'tabby cat', 'snow leopard', 'lion', 'bengal tiger']

elif "food" in dataset:
    template = 'a photo of the {}'
    base_prompt = ["a photo of the food."]

    class_dir = [ "baby_back_ribs", "beef_carpaccio", "beignets", "bibimbap", "caesar_salad",\
                            "clam_chowder", "dumplings", "edamame", "spaghetti_bolognese", "strawberry_shortcake"]
    if n == 4:
        # food 
        if use_base:
            norm=torch.Tensor([1.5637, 1.0000, 1.3215, 1.2469])
            thres=torch.Tensor([4.5680, 6.6849, 3.9575, 6.8483])
            mean=torch.Tensor([ 7.1738, 11.2174,  8.4882,  8.9965])
            std=torch.Tensor([1.0241, 0.7693, 0.4985, 0.5476])
        else:
            norm=torch.Tensor([1.1180, 1.0000, 1.0506, 1.0313])
            thres=torch.Tensor([20.8280, 21.6058, 23.9672, 22.0352])
            mean=torch.Tensor([32.2527, 36.0574, 34.3191, 34.9638])
            std=torch.Tensor([0.7133, 0.8058, 0.8056, 0.6978])
        prompts_root = [ class_dir[i] for i in [0, 2, 6, 7] ]
        prompts = ['baby back ribs', 'beignets', 'dumplings','edamame']
    elif n == 7:
        # food 7
        if use_base:
            norm=torch.Tensor([1.5669, 1.0000, 1.0004, 1.1397, 1.3173, 1.2419, 1.3999])
            thres=torch.Tensor([4.5743, 6.0909, 6.6673, 6.6015, 3.9851, 6.7936, 4.5143])
            mean=torch.Tensor([ 7.1411, 11.1895, 11.1850,  9.8182,  8.4940,  9.0103,  7.9929])
            std=torch.Tensor([1.0446, 1.0050, 0.7800, 0.7831, 0.5173, 0.5251, 0.8261])
        else:
            norm=torch.Tensor([1.1338, 1.0000, 1.0144, 1.0410, 1.0656, 1.0454, 1.1227])
            thres=torch.Tensor([20.8295, 21.2649, 21.6002, 21.6064, 23.9687, 22.0481, 20.7577])
            mean=torch.Tensor([32.2456, 36.5586, 36.0405, 35.1176, 34.3067, 34.9725, 32.5625])
            std=torch.Tensor([0.7292, 1.0437, 0.8213, 0.7660, 0.8108, 0.6904, 0.7228])
        prompts_root = [ class_dir[i] for i in [0, 1, 2, 5, 6, 7, 9] ]
        prompts = ['baby back ribs','beef carpaccio','beignets','clam chowder','dumplings','edamame', 'strawberry shortcake' ]
    elif n == 10:
        # food 10
        if use_base:
            norm=torch.Tensor([1.5586, 1.0000, 1.1088, 1.0646, 1.2460, 1.1371, 1.2663, 1.2399, 1.4251,1.3958])
            thres=torch.Tensor([4.5673, 6.0816, 6.5011, 7.0163, 3.8244, 6.6153, 4.1326, 6.8310, 5.2493,4.4851])
            mean=torch.Tensor([ 7.1688, 11.1731, 10.0769, 10.4954,  8.9672,  9.8259,  8.8231,  9.0115,7.8402,  8.0047])
            std=torch.Tensor([1.0464, 0.9919, 0.6352, 0.8694, 0.8487, 0.7978, 0.5394, 0.5173, 0.6520, 0.8442])
        else:
            norm=torch.Tensor([1.1503, 1.0149, 1.0614, 1.0000, 1.0551, 1.0552, 1.0740, 1.0592, 1.1027,1.1388])
            thres=torch.Tensor([20.8303, 21.2689, 21.3105, 20.4814, 23.1095, 21.6097, 23.7660, 22.0443,21.4264, 20.7624])
            mean=torch.Tensor([32.2322, 36.5302, 34.9305, 37.0755, 35.1406, 35.1365, 34.5199, 35.0018,33.6225, 32.5564])
            std=torch.Tensor([0.7608, 1.0583, 0.8089, 0.9087, 0.9750, 0.7772, 0.8271, 0.6961, 0.6737,0.7092])
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
        # celeb4
        if use_base: 
            norm=torch.Tensor([1.0000, 1.4675, 1.4140, 1.4801])
            thres=torch.Tensor([2.7025, 1.6486, 2.0882, 1.2115])
            mean=torch.Tensor([7.2923, 4.9692, 5.1572, 4.9268])
            std=torch.Tensor([0.4397, 0.4864, 0.3087, 0.8990])
        else:
            norm=torch.Tensor([1.0000, 1.0708, 1.0745, 1.1001])
            thres=torch.Tensor([22.0828, 20.5836, 22.7025, 20.2290])
            mean=torch.Tensor([28.7448, 26.8432, 26.7516, 26.1285])
            std=torch.Tensor([0.5649, 0.5855, 0.4096, 0.8572])
        prompts_root = [ class_dir[i] for i in [9, 8, 31, 15] ]
        prompts = ['blond hair', 'black hair' , 'smiling', 'eyeglasses',] 
        print()
    elif n == 7:
        # celeb7
        if use_base: 
            norm=torch.Tensor([1.0000, 1.3099, 1.4618, 1.4123, 1.4723, 1.2772, 1.2826])
            thres=torch.Tensor([2.7096, 1.8316, 1.6461, 2.0885, 1.2063, 1.7403, 1.9382])
            mean=torch.Tensor([7.2680, 5.5487, 4.9720, 5.1461, 4.9365, 5.6907, 5.6666])
            std=torch.Tensor([0.4122, 0.4675, 0.4942, 0.3083, 0.8916, 0.7225, 0.4984])
        else:
            norm=torch.Tensor([1.0000, 1.0710, 1.0704, 1.0741, 1.0991, 1.0609, 1.0505])
            thres=torch.Tensor([22.0843, 21.9871, 20.5867, 22.7026, 20.2304, 20.1576, 19.4745])
            mean=torch.Tensor([28.7308, 26.8255, 26.8412, 26.7493, 26.1407, 27.0826, 27.3507])
            std=torch.Tensor([0.5563, 0.4715, 0.6158, 0.3837, 0.8553, 0.6982, 0.6306])
        prompts_root = [ class_dir[i] for i in [9, 33, 8, 31, 15, 16, 5] ]
        prompts = ['blond hair', 'wavy hair', 'black hair' , 'smiling', 'eyeglasses', 'goatee', 'bangs',]
        print()
    elif n == 10:
        # celeb10
        if use_base:
            # [1.0000, 1.1075, 1.0495, 1.0686, 1.0921, 1.1100, 1.1958, 1.0824, 1.2675, 1.0159]
            # [1.0000, 1.3064, 1.3130, 1.4598, 1.4078, 1.6798, 1.4666, 1.2764, 1.2868,1.2979]
            norm=torch.Tensor([1.0000, 1.1075, 1.0495, 1.0686, 1.0921, 1.1100, 1.1958, 1.0824, 1.2675, 1.0159])
            #norm=torch.Tensor([1.0000, 1.3064, 1.3130, 1.4598, 1.4078, 1.6798, 1.4666, 1.2764, 1.2868,1.2979])
            thres=torch.Tensor([2.7089, 1.2969, 1.8257, 1.6521, 2.0859, 1.5384, 1.2118, 1.7422, 1.9401, 2.3117])
            mean=torch.Tensor([7.2653, 5.5615, 5.5332, 4.9769, 5.1608, 4.3251, 4.9537, 5.6920, 5.6458, 5.5975])
            std=torch.Tensor([0.4272, 1.0211, 0.4847, 0.4980, 0.3043, 0.4211, 0.8901, 0.7345, 0.4882,0.3603])
        else:
            norm=torch.Tensor([1.0000, 1.0645, 1.0703, 1.0689, 1.0734, 1.0969, 1.0988, 1.0609, 1.0501,1.0482])
            thres=torch.Tensor([22.0837, 19.9591, 21.9849, 20.5851, 22.7019, 21.2948, 20.2298, 20.1563,19.4749, 23.4730])
            mean=torch.Tensor([28.7175, 26.9770, 26.8314, 26.8655, 26.7543, 26.1803, 26.1357, 27.0699,27.3483, 27.3972])
            std=torch.Tensor([0.5606, 0.9889, 0.4633, 0.6014, 0.3848, 0.4524, 0.8516, 0.6948, 0.6289,0.4254])
        prompts_root = [ class_dir[i] for i in [9, 4, 33, 8, 31, 32, 15, 16, 5, 1] ]
        prompts = ['blond hair', 'bald', 'wavy hair', 'black hair' ,\
                        'smiling', 'straight hair', 'eyeglasses', 'goatee', 'bangs', 'arched eyebrows']


prompts_root_dir = []
prompts_over_dir, prompts_under_dir, num_data = get_over_under_paths(num_data, norm, mean, std, prompts, sim_perclass, rel_sim_perclass, sim_perclass_imgpath)
for i in range(len(prompts)):
    prompts_root_dir.append( os.path.join(dataset_root, prompts_root[i]) )

#print(prompts_root_dir)


# TP_list.append("{0}: clip:{1}/{2}, norm:{3}/{4}".format(prompts[k], over_count, num_gt, under_count, num_gt))
# FP_list.append("{0}: clip:{1}/{2}, norm:{3}/{4}".format(prompts[k], num_over-over_count, num_over, num_under-under_count, num_under))

# total_over_count_list = [0]*len(prompts)
# total_under_count_list = [0]*len(prompts)
# num_over_list = [0]*len(prompts)
# num_under_list = [0]*len(prompts)
# num_gt_list = [0]*len(prompts)

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

    #import pdb; pdb.set_trace()
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
    
# print("------------------------------Total True Positive -------------------------")
# for p in range(len(prompts)):
#     #import pdb; pdb.set_trace()
#     print("{0}: clip:{1}/{2}".format(prompts[p], total_over_count_list[p], num_gt_list[p]))

# print("------------------------------Total False Positive -------------------------")
# for p in range(len(prompts)):
#     print("{0}: clip:{1}/{2}".format(prompts[p], num_over_list[p]-total_over_count_list[p], num_over_list[p]))
#import pdb; pdb.set_trace()

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