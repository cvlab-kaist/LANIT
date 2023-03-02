dir="./expr/samples/"
name="celeb-10"
mkdir "$dir$name/"
cat $0 > "$dir$name/${0##*/}"
chmod -R 777 ./

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
