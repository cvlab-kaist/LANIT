CUDA_VISIBLE_DEVICES=5 python main.py\
    --name animalface-10\
    --dataset animal_faces\
    --mode train\
    --train_img_dir ./dataset/animal_faces\
    --val_img_dir ./dataset/animal_faces\
    --checkpoint_dir ./checkpoints/lanit_af_weight/\
    --num_domains 10\
    --cycle\
    --dc\
    --ds\
    --multi_hot\
    --topk 1\
    --use_base\
    --zero_cut\
    --w_hpf 0\


    
