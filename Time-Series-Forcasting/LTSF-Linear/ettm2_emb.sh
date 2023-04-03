if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ETT-2021" ]; then
    mkdir ./logs/ETT-2021
fi

for model_name in Linear NLinear DLinear DLinear_T DLinear_S
do
seq_len=336
ver=B
weight=linear

CUDA_VISIBLE_DEVICES=3 python -u run_longExp.py \
  --dataset none \
  --seed 2021 \
  --time_emb 1 \
  --time_channel 40 \
  --emb_dim 8 \
  --is_training 1 \
  --root_path /nas/datahub/ETT/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'96'_'emb \
  --model $model_name \
  --data ETTm2 \
  --ver $ver \
  --weight $weight \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.0001   >logs/ETT-2021/final'_'$model_name'_'ETTm2_$seq_len'_'96'_'time_sigmoid.log
printf "Done %s 96" ${model_name}

CUDA_VISIBLE_DEVICES=3 python -u run_longExp.py \
  --dataset none \
  --seed 2021 \
  --time_emb 1 \
  --time_channel 40 \
  --emb_dim 8 \
  --is_training 1 \
  --root_path /nas/datahub/ETT/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'192'_'emb \
  --model $model_name \
  --data ETTm2 \
  --ver $ver \
  --weight $weight \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/ETT-2021/final'_'$model_name'_'ETTm2_$seq_len'_'192'_'time_sigmoid.log
printf "Done %s 192" ${model_name}


CUDA_VISIBLE_DEVICES=3 python -u run_longExp.py \
  --dataset none \
  --seed 2021 \
  --time_emb 1 \
  --time_channel 40 \
  --emb_dim 8 \
  --is_training 1 \
  --root_path /nas/datahub/ETT/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'336'_'emb \
  --model $model_name \
  --data ETTm2 \
  --ver $ver \
  --weight $weight \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/ETT-2021/final'_'$model_name'_'ETTm2_$seq_len'_'336'_'time_sigmoid.log
printf "Done %s 336" ${model_name}


CUDA_VISIBLE_DEVICES=3 python -u run_longExp.py \
  --dataset none \
  --seed 2021 \
  --time_emb 1 \
  --time_channel 40 \
  --emb_dim 8 \
  --is_training 1 \
  --root_path /nas/datahub/ETT/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'720'_'emb \
  --model $model_name \
  --data ETTm2 \
  --ver $ver \
  --weight $weight \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/ETT-2021/final'_'$model_name'_'ETTm2_$seq_len'_'720'_'time_sigmoid.log
printf "Done %s 720" ${model_name}


done