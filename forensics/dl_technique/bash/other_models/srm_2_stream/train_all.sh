export lr=3e-4
export batch_size=16
#
export image_size=256
export workers=4
export TRAIN_DIR="/mnt/disk1/phucnp/Dataset/UADFV/image/train"
export VAL_DIR="/mnt/disk1/phucnp/Dataset/UADFV/image/val"
export TEST_DIR="/mnt/disk1/phucnp/Dataset/UADFV/image/test"
#
export n_epochs=30
export es_patience=5
export es_metric="none"
export eval_per_iters=200  # 24000
#
export gpu_id=2
export loss="cbce"
export gamma=0.0

python args_train.py --train_dir $TRAIN_DIR --val_dir $VAL_DIR --test_dir $TEST_DIR --batch_size $batch_size --lr $lr --n_epochs $n_epochs --image_size $image_size --workers $workers --gpu_id $gpu_id --es_metric $es_metric --es_patience $es_patience --loss $loss --gamma $gamma --eval_per_iters $eval_per_iters \
                    --checkpoint "checkpoint/uadfv/srm_2_stream" --resume "" \
                    srm_2_stream
#