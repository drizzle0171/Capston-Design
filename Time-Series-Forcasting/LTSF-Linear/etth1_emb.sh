if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ETT-2021" ]; then
    mkdir ./logs/ETT-2021
fi

for model_name in NLinear #NLinear DLinear DLinear_T DLinear_S
do
seq_len=336
ver=TWM-ReLU
weight=linear

CUDA_VISIBLE_DECVIES=0 python -u run_longExp.py \
  --dataset none \
  --seed 2021 \
  --time_emb 1 \
  --time_channel 16 \
  --emb_dim 8 \
  --is_training 1 \
  --root_path /nas/datahub/ETT/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'96'_'emb \
  --model $model_name \
  --data ETTh1 \
  --ver $ver \
  --weight $weight \
  --features M \
  --seq_len 336 \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.00005
  # >logs/ETT-2021/final'_'$model_name'_'Etth1'_'336'_'$ver'_'96'_'time_sigmoid.log
printf "Done %s 96\n" ${model_name}


# CUDA_VISIBLE_DECVIES=0 python -u run_longExp.py \
#   --dataset none \
#   --seed 2021 \
#   --time_channel 32 \
#   --emb_dim 8 \
#   --time_emb 1 \
#   --is_training 1 \
#   --root_path /nas/datahub/ETT/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_$seq_len'_'192'_'emb \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --ver $ver \
#   --weight $weight \
#   --embed none \
#   --seq_len 336 \
#   --pred_len 192 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/ETT-2021/final'_'$model_name'_'Etth1'_'336'_'$ver'_'192'_'time_sigmoid.log
# printf "Done %s 192\n" ${model_name}


# CUDA_VISIBLE_DECVIES=0 python -u run_longExp.py \
#   --dataset none \
#   --seed 2021 \
#   --time_emb 1 \
#   --is_training 1 \
#   --time_channel 32 \
#   --emb_dim 8 \
#   --root_path /nas/datahub/ETT/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_$seq_len'_'336'_'emb \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --ver $ver \
#   --weight $weight \
#   --embed none \
#   --seq_len 336 \
#   --pred_len 336 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/ETT-2021/final'_'$model_name'_'Etth1'_'336'_'$ver'_'336'_'time_sigmoid.log
# printf "Done %s 336\n" ${model_name}


# CUDA_VISIBLE_DECVIES=0 python -u run_longExp.py \
#   --dataset none \
#   --seed 2021 \
#   --time_emb 1 \
#   --is_training 1 \
#   --time_channel 32 \
#   --emb_dim 8 \
#   --root_path /nas/datahub/ETT/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_$seq_len'_'720'_'emb \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --ver $ver \
#   --weight $weight \
#   --embed none \
#   --seq_len 336 \
#   --pred_len 720 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/ETT-2021/final'_'$model_name'_'Etth1'_'336'_'$ver'_'720'_'time_sigmoid.log
# printf "Done %s 720\n" ${model_name}


done