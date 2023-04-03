for dataset in  b70b15brest
do
for ver in B-TWM TWM TWM-ReLU B D D-ReLU
do
for model_name in Linear NLinear DLinear DLinear_T DLinear_S
do
CUDA_VISIBLE_DEVICES=0 python -u run_longExp.py \
  --dataset $dataset \
  --seed 2021 \
  --is_training 1 \
  --model_id Linear_2021 \
  --model $model_name \
  --time_emb 0 \
  --data mirae \
  --des 'Exp' \
  --ver $ver \
  --weight linear \
  --time_channel 4 \
  --time conv \
  --loss mse \
  --seq_len 72 \
  --pred_len 12 \
  --train_epochs 30 \
  --itr 1 --batch_size 32 --learning_rate 0.005 >./logs/mirae-2021/final'_'$model_name'_'mirae'_'$ver.log
done
done
done
