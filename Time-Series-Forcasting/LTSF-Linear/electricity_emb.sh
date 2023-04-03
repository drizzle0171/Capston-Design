if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/electricity-2021" ]; then
    mkdir ./logs/electricity-2021
fi

for model_name in Linear NLinear DLinear DLinear_T DLinear_S
do
seq_len=336
ver=TWM-ReLU
weight=linear

CUDA_VISIBLE_DEVICES=2 python -u run_longExp.py \
  --electricity 1 \
  --dataset none \
  --seed 2021 \
  --time_emb 1 \
  --time_channel 32 \
  --emb_dim 8 \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'96'_'emb \
  --model $model_name \
  --data custom \
  --ver $ver \
  --weight $weight \
  --time conv \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate 0.0005 >logs/electricity-2021/final'_'$model_name'_'electricity'_'$ver'_'$seq_len'_'96'_'time_sigmoid.log 
printf "Done %s 96" ${model_name}


CUDA_VISIBLE_DEVICES=2 python -u run_longExp.py \
  --electricity 1 \
  --dataset none \
  --seed 2021 \
  --time_emb 1 \
  --time_channel 32 \
  --emb_dim 8 \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'192'_'emb \
  --model $model_name \
  --data custom \
  --ver $ver \
  --weight $weight \
  --time conv \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate 0.0005 >logs/electricity-2021/final'_'$model_name'_'electricity'_'$ver'_'$seq_len'_'192'_'time_sigmoid.log  
printf "Done %s 192" ${model_name}


CUDA_VISIBLE_DEVICES=2 python -u run_longExp.py \
  --electricity 1 \
  --dataset none \
  --seed 2021 \
  --time_emb 1 \
  --time_channel 32 \
  --emb_dim 8 \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'336'_'emb \
  --model $model_name \
  --data custom \
  --ver $ver \
  --weight $weight \
  --time conv \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate 0.0005  >logs/electricity-2021/final'_'$model_name'_'electricity'_'$ver'_'$seq_len'_'336'_'time_sigmoid.log  
printf "Done %s 336" ${model_name}


CUDA_VISIBLE_DEVICES=2 python -u run_longExp.py \
  --electricity 1 \
  --dataset none \
  --seed 2021 \
  --time_emb 1 \
  --time_channel 32 \
  --emb_dim 8 \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'720'_'emb \
  --model $model_name \
  --data custom \
  --ver $ver \
  --weight $weight \
  --time conv \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate 0.0005  >logs/electricity-2021/final'_'$model_name'_'electricity'_'$ver'_'$seq_len'_'720'_'time_sigmoid.log  
printf "Done %s 720" ${model_name}


done