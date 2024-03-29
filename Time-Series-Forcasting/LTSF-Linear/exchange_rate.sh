
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/exchange_rate-2021" ]; then
    mkdir ./logs/exchange_rate-2021
fi

for model_name in Linear NLinear DLinear DLinear_T DLinear_S
do
seq_len=336
ver=TWM-ReLU

CUDA_VISIBLE_DEVICES=6 python -u run_longExp.py \
  --time_emb 0 \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange'_'$ver'_'$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --ver $ver \
  --time_channel 3 \
  --time conv \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 --batch_size 8 --learning_rate 0.0005 >logs/exchange_rate-2021/$model_name'_'Exchange'_'$ver'_'$seq_len'_'96'_'time_sigmoid.log 
printf "Done %s 96" ${model_name}

CUDA_VISIBLE_DEVICES=6 python -u run_longExp.py \
  --time_emb 0 \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange'_'$ver'_'$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --ver $ver \
  --time_channel 3 \
  --time conv \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 --batch_size 8 --learning_rate 0.0005 >logs/exchange_rate-2021/$model_name'_'Exchange'_'$ver'_'$seq_len'_'192'_'time_sigmoid.log 
printf "Done %s 192" ${model_name}

CUDA_VISIBLE_DEVICES=6 python -u run_longExp.py \
  --time_emb 0 \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange'_'$ver'_'$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --ver $ver \
  --time_channel 3\
  --time conv \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 --batch_size 32  --learning_rate 0.0005 >logs/exchange_rate-2021/$model_name'_'Exchange'_'$ver'_'$seq_len'_'336'_'time_sigmoid.log 
printf "Done %s 336" ${model_name}

CUDA_VISIBLE_DEVICES=6 python -u run_longExp.py \
  --time_emb 0 \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange'_'$ver'_'$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --ver $ver \
  --time_channel 3 \
  --time conv \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.0005 >logs/exchange_rate-2021/$model_name'_'Exchange'_'$ver'_'$seq_len'_'720'_'time_sigmoid.log
printf "Done %s 720" ${model_name}

done