export lr=1e-3
export batch_size=32
#
export image_size=128
export workers=4
export TRAIN_DIR="/mnt/disk1/phucnp/Dataset/ff/image/all/train"
export VAL_DIR="/mnt/disk1/phucnp/Dataset/ff/image/all/val"
export TEST_DIR="/mnt/disk1/phucnp/Dataset/ff/image/all/test"
#
export n_epochs=30
export es_patience=5
export es_metric="none"
export eval_per_iters=4000
#
export gpu_id=0
export loss="bce"
export gamma=0.0
# 
python args_train.py --train_dir $TRAIN_DIR --val_dir $VAL_DIR --test_dir $TEST_DIR --batch_size $batch_size --lr $lr --n_epochs $n_epochs --image_size $image_size --workers $workers --gpu_id $gpu_id --es_metric $es_metric --es_patience $es_patience --loss $loss --gamma $gamma --eval_per_iters $eval_per_iters \
                    --checkpoint "checkpoint/ff_all/dual_efficient" --resume "" \
                    dual_efficient
python args_train.py --train_dir $TRAIN_DIR --val_dir $VAL_DIR --test_dir $TEST_DIR --batch_size $batch_size --lr $lr --n_epochs $n_epochs --image_size $image_size --workers $workers --gpu_id $gpu_id --es_metric $es_metric --es_patience $es_patience --loss $loss --gamma $gamma --eval_per_iters $eval_per_iters \
                    --checkpoint "checkpoint/ff_all/xception" --resume "" \
                    xception
python args_train.py --train_dir $TRAIN_DIR --val_dir $VAL_DIR --test_dir $TEST_DIR --batch_size $batch_size --lr $lr --n_epochs $n_epochs --image_size $image_size --workers $workers --gpu_id $gpu_id --es_metric $es_metric --es_patience $es_patience --loss $loss --gamma $gamma --eval_per_iters $eval_per_iters \
                    --checkpoint "checkpoint/ff_all/meso4" --resume "" \
                    meso4
# # 
# 
