if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ETT-2021" ]; then
    mkdir ./logs/ETT-2021
fi

for model_name in Linear NLinear DLinear DLinear_T DLinear_S
do
seq_len=336
ver=TWM-2
weight=linear

CUDA_VISIBLE_DECVIES=0 python -u run_longExp.py \
  --is_training 1 \
  --root_path /nas/datahub/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --ver $ver \
  --weight $weight \
  --time_channel 4 \
  --features S \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.05 >logs/ETT-2021/$model_name'_'ETTh2_$seq_len'_'96'_'time_sigmoid.log

CUDA_VISIBLE_DECVIES=0 python -u run_longExp.py \
  --is_training 1 \
  --root_path /nas/datahub/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --ver $ver \
  --weight $weight \
  --features S \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.05 >logs/ETT-2021/$model_name'_'ETTh2_$seq_len'_'192'_'time_sigmoid.log


CUDA_VISIBLE_DECVIES=0 python -u run_longExp.py \
  --is_training 1 \
  --root_path /nas/datahub/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --ver $ver \
  --weight $weight \
  --features S \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.05 >logs/ETT-2021/$model_name'_'ETTh2_$seq_len'_'336'_'time_sigmoid.log

CUDA_VISIBLE_DECVIES=0 python -u run_longExp.py \
  --is_training 1 \
  --root_path /nas/datahub/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --ver $ver \
  --weight $weight \
  --features S \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.05 >logs/ETT-2021/$model_name'_'ETTh2_$seq_len'_'720'_'time_sigmoid.log
done