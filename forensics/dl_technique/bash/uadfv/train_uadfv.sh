python train.py --train_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/test" --batch_size 32 --lr 3e-4 --n_epochs 25 --image_size 128 --workers 8 --checkpoint "uadfv_checkpoint/dual_efficient_vit" --resume "" --gpu_id 0 --es_patience 5 --es_metric "val_acc" --loss "focalloss" --gamma 0.5 dual_efficient_vit --version "cross_attention-freq-add" --weight 1 --patch_size 2
python train.py --train_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/test" --batch_size 32 --lr 3e-4 --n_epochs 25 --image_size 128 --workers 8 --checkpoint "uadfv_checkpoint/dual_efficient_vit" --resume "" --gpu_id 0 --es_patience 5 --es_metric "val_acc" --loss "wbce" dual_efficient_vit --version "cross_attention-freq-add" --weight 1 --patch_size 2
python train.py --train_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/test" --batch_size 32 --lr 3e-4 --n_epochs 25 --image_size 128 --workers 8 --checkpoint "uadfv_checkpoint/dual_efficient_vit" --resume "" --gpu_id 0 --es_patience 5 --es_metric "val_acc" --loss "bce" dual_efficient_vit --version "cross_attention-freq-add" --weight 1 --patch_size 2
#
python train.py --train_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/test" --batch_size 32 --lr 3e-4 --n_epochs 25 --image_size 128 --workers 8 --checkpoint "uadfv_checkpoint/dual_efficient_vit" --resume "" --gpu_id 0 --es_patience 5 --es_metric "val_acc" dual_efficient_vit --version "cross_attention-freq-add" --weight 1 --patch_size 1
python train.py --train_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/test" --batch_size 32 --lr 3e-4 --n_epochs 25 --image_size 128 --workers 8 --checkpoint "uadfv_checkpoint/dual_efficient_vit" --resume "" --gpu_id 0 --es_patience 5 --es_metric "val_acc" dual_efficient_vit --version "cross_attention-freq-cat" --weight 1 --patch_size 2
python train.py --train_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/test" --batch_size 32 --lr 3e-4 --n_epochs 25 --image_size 128 --workers 8 --checkpoint "uadfv_checkpoint/dual_efficient_vit" --resume "" --gpu_id 0 --es_patience 5 --es_metric "val_acc" dual_efficient_vit --version "cross_attention-freq-cat" --weight 1 --patch_size 1
# 
python train.py --train_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/test" --batch_size 64 --lr 3e-4 --n_epochs 25 --image_size 128 --workers 8 --checkpoint "uadfv_checkpoint/dual_efficient_vit" --resume "" --gpu_id 0 --es_patience 5 --es_metric "val_acc" dual_efficient_vit --version "cross_attention-freq-add" --weight 0.5 --patch_size 2
python train.py --train_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/test" --batch_size 64 --lr 3e-4 --n_epochs 25 --image_size 128 --workers 8 --checkpoint "uadfv_checkpoint/dual_efficient_vit" --resume "" --gpu_id 0 --es_patience 5 --es_metric "val_acc" dual_efficient_vit --version "cross_attention-freq-add" --weight 0.5 --patch_size 1
python train.py --train_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/test" --batch_size 64 --lr 3e-4 --n_epochs 25 --image_size 128 --workers 8 --checkpoint "uadfv_checkpoint/dual_efficient_vit" --resume "" --gpu_id 0 --es_patience 5 --es_metric "val_acc" dual_efficient_vit --version "cross_attention-freq-cat" --weight 0.5 --patch_size 2
python train.py --train_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/test" --batch_size 64 --lr 3e-4 --n_epochs 25 --image_size 128 --workers 8 --checkpoint "uadfv_checkpoint/dual_efficient_vit" --resume "" --gpu_id 0 --es_patience 5 --es_metric "val_acc" dual_efficient_vit --version "cross_attention-freq-cat" --weight 0.5 --patch_size 1
# 
python train.py --train_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/test" --batch_size 32 --lr 3e-4 --n_epochs 25 --image_size 128 --workers 8 --checkpoint "uadfv_checkpoint/dual_efficient_vit" --resume "" --gpu_id 0 --es_patience 5 --es_metric "val_acc" dual_efficient_vit --version "cross_attention-spatial-add" --weight 1 --patch_size 2
python train.py --train_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/test" --batch_size 32 --lr 3e-4 --n_epochs 25 --image_size 128 --workers 8 --checkpoint "uadfv_checkpoint/dual_efficient_vit" --resume "" --gpu_id 0 --es_patience 5 --es_metric "val_acc" dual_efficient_vit --version "cross_attention-spatial-add" --weight 1 --patch_size 1
python train.py --train_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/test" --batch_size 32 --lr 3e-4 --n_epochs 25 --image_size 128 --workers 8 --checkpoint "uadfv_checkpoint/dual_efficient_vit" --resume "" --gpu_id 0 --es_patience 5 --es_metric "val_acc" dual_efficient_vit --version "cross_attention-spatial-cat" --weight 1 --patch_size 2
python train.py --train_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/test" --batch_size 32 --lr 3e-4 --n_epochs 25 --image_size 128 --workers 8 --checkpoint "uadfv_checkpoint/dual_efficient_vit" --resume "" --gpu_id 0 --es_patience 5 --es_metric "val_acc" dual_efficient_vit --version "cross_attention-spatial-cat" --weight 1 --patch_size 1
# 
python train.py --train_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/test" --batch_size 32 --lr 3e-4 --n_epochs 25 --image_size 128 --workers 8 --checkpoint "uadfv_checkpoint/dual_efficient_vit" --resume "" --gpu_id 0 --es_patience 5 --es_metric "val_acc" dual_efficient_vit --version "merge-add" --weight 0.8 --patch_size 2
python train.py --train_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/test" --batch_size 32 --lr 3e-4 --n_epochs 25 --image_size 128 --workers 8 --checkpoint "uadfv_checkpoint/dual_efficient_vit" --resume "" --gpu_id 0 --es_patience 5 --es_metric "val_acc" dual_efficient_vit --version "merge-add" --weight 0.8 --patch_size 1
python train.py --train_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/test" --batch_size 32 --lr 3e-4 --n_epochs 25 --image_size 128 --workers 8 --checkpoint "uadfv_checkpoint/dual_efficient_vit" --resume "" --gpu_id 0 --es_patience 5 --es_metric "val_acc" dual_efficient_vit --version "merge-cat" --weight 0.8 --patch_size 2
python train.py --train_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/test" --batch_size 32 --lr 3e-4 --n_epochs 25 --image_size 128 --workers 8 --checkpoint "uadfv_checkpoint/dual_efficient_vit" --resume "" --gpu_id 0 --es_patience 5 --es_metric "val_acc" dual_efficient_vit --version "merge-cat" --weight 0.8 --patch_size 1
# Efficientvit
python train.py --train_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/test" --batch_size 64 --lr 3e-4 --n_epochs 25 --image_size 128 --workers 8 --checkpoint "uadfv_checkpoint/efficient_vit" --resume "" --gpu_id 0 --es_patience 5 --es_metric "val_acc" efficient_vit --patch_size 1
python train.py --train_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/UADFV/image/test" --batch_size 64 --lr 3e-4 --n_epochs 25 --image_size 128 --workers 8 --checkpoint "uadfv_checkpoint/efficient_vit" --resume "" --gpu_id 0 --es_patience 5 --es_metric "val_acc" efficient_vit --patch_size 2