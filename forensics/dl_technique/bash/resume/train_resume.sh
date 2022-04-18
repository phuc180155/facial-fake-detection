python train.py --train_dir "/mnt/disk1/phucnp/Dataset/dfdc/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/dfdc/image/test" --batch_size 64 --lr 3e-4 --n_epochs 30 --image_size 128 --workers 8 --checkpoint "checkpoint/dfdc/dual_efficient_vit/ver_cross_attention-freq-add_weight_0.8_imgsize_128_lr_0.0003_patchsize_2/ver_cross_attention-freq-add_weight_0.8_imgsize_128_lr_0.0003_patchsize_2_es_val_acc_loss_bce_freeze_0" --resume "epoch_17_0.5462956806617764_0.8890678231363494.pt" --gpu_id 2 --es_patience -1 --es_metric "none" --loss "bce" dual_efficient_vit --freeze 0 --version "cross_attention-freq-add" --weight 0.8 --patch_size 2
python train.py --train_dir "/mnt/disk1/phucnp/Dataset/df_in_the_wild/image/train" --val_dir "/mnt/disk1/phucnp/Dataset/df_in_the_wild/image/test" --batch_size 64 --lr 3e-4 --n_epochs 30 --image_size 128 --workers 8 --checkpoint "checkpoint/df_in_the_wild/dual_efficient_vit/ver_cross_attention-freq-add_weight_0.8_imgsize_128_lr_0.0003_patchsize_2_es_val_acc_loss_wbce_freeze_0/ver_cross_attention-freq-add_weight_0.8_imgsize_128_lr_0.0003_patchsize_2_es_val_acc_loss_wbce_freeze_0" --resume "epoch_12_1.5428371882291316_0.7880262220666175.pt" --gpu_id 2 --es_patience -1 --es_metric "none" --loss "wbce" dual_efficient_vit --freeze 0 --version "cross_attention-freq-add" --weight 0.8 --patch_size 2
# 