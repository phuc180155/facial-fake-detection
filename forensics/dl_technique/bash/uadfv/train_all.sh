export lr=3e-4
export batch_size=32
#
export image_size=128
export workers=4
export TRAIN_DIR="/mnt/disk1/phucnp/Dataset/UADFV/image/train"
export VAL_DIR="/mnt/disk1/phucnp/Dataset/UADFV/image/val"
export TEST_DIR="/mnt/disk1/phucnp/Dataset/UADFV/image/test"
#
export n_epochs=30
export es_patience=5
export es_metric="none"
export eval_per_iters=200
#
export gpu_id=0
export loss="bce"
export gamma=0.0
#
python args_train.py --train_dir $TRAIN_DIR --val_dir $VAL_DIR --test_dir $TEST_DIR --batch_size $batch_size --lr $lr --n_epochs $n_epochs --image_size $image_size --workers $workers --gpu_id $gpu_id --es_metric $es_metric --es_patience $es_patience --loss $loss --gamma $gamma --eval_per_iters $eval_per_iters \
                    --checkpoint "checkpoint/uadfv/dual_efficient_vit" --resume "" \
                    --depth 4 --heads 3 \
                    dual_efficient_vit --version "cross_attention-freq-add" --weight 0.8 --patch_size 2
#
python args_train.py --train_dir $TRAIN_DIR --val_dir $VAL_DIR --test_dir $TEST_DIR --batch_size $batch_size --lr $lr --n_epochs $n_epochs --image_size $image_size --workers $workers --gpu_id $gpu_id --es_metric $es_metric --es_patience $es_patience --loss $loss --gamma $gamma --eval_per_iters $eval_per_iters \
                    --checkpoint "checkpoint/uadfv/dual_efficient_vit" --resume "" \
                    dual_efficient_vit --version "cross_attention-freq-add" --weight 0.8 --patch_size 1
#
python args_train.py --train_dir $TRAIN_DIR --val_dir $VAL_DIR --test_dir $TEST_DIR --batch_size $batch_size --lr $lr --n_epochs $n_epochs --image_size $image_size --workers $workers --gpu_id $gpu_id --es_metric $es_metric --es_patience $es_patience --loss $loss --gamma $gamma --eval_per_iters $eval_per_iters \
                    --checkpoint "checkpoint/uadfv/dual_efficient_vit" --resume "" \
                    dual_efficient_vit --version "cross_attention-freq-add" --weight 0.8 --patch_size 2
#
python args_train.py --train_dir $TRAIN_DIR --val_dir $VAL_DIR --test_dir $TEST_DIR --batch_size $batch_size --lr $lr --n_epochs $n_epochs --image_size $image_size --workers $workers --gpu_id $gpu_id --es_metric $es_metric --es_patience $es_patience --loss $loss --gamma $gamma --eval_per_iters $eval_per_iters \
                    --checkpoint "checkpoint/uadfv/dual_efficient_vit" --resume "" \
                    dual_efficient_vit --version "cross_attention-freq-cat" --weight 0.8 --patch_size 1
#
python args_train.py --train_dir $TRAIN_DIR --val_dir $VAL_DIR --test_dir $TEST_DIR --batch_size $batch_size --lr $lr --n_epochs $n_epochs --image_size $image_size --workers $workers --gpu_id $gpu_id --es_metric $es_metric --es_patience $es_patience --loss $loss --gamma $gamma --eval_per_iters $eval_per_iters \
                    --checkpoint "checkpoint/uadfv/dual_efficient_vit" --resume "" \
                    dual_efficient_vit --version "cross_attention-freq-cat" --weight 0.8 --patch_size 2
#
python args_train.py --train_dir $TRAIN_DIR --val_dir $VAL_DIR --test_dir $TEST_DIR --batch_size $batch_size --lr $lr --n_epochs $n_epochs --image_size $image_size --workers $workers --gpu_id $gpu_id --es_metric $es_metric --es_patience $es_patience --loss $loss --gamma $gamma --eval_per_iters $eval_per_iters \
                    --checkpoint "checkpoint/uadfv/meso4" --resume "" \
                    meso4
#
python args_train.py --train_dir $TRAIN_DIR --val_dir $VAL_DIR --test_dir $TEST_DIR --batch_size $batch_size --lr $lr --n_epochs $n_epochs --image_size $image_size --workers $workers --gpu_id $gpu_id --es_metric $es_metric --es_patience $es_patience --loss $loss --gamma $gamma --eval_per_iters $eval_per_iters \
                    --checkpoint "checkpoint/uadfv/xception" --resume "" \
                    xception
#
python args_train.py --train_dir $TRAIN_DIR --val_dir $VAL_DIR --test_dir $TEST_DIR --batch_size $batch_size --lr $lr --n_epochs $n_epochs --image_size $image_size --workers $workers --gpu_id $gpu_id --es_metric $es_metric --es_patience $es_patience --loss $loss --gamma $gamma --eval_per_iters $eval_per_iters \
                    --checkpoint "checkpoint/uadfv/dual_efficient" --resume "" \
                    dual_efficient
#
