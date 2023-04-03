
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/weather-2021" ]; then
    mkdir ./logs/weather-2021
fi

for model_name in Linear NLinear DLinear DLinear_T DLinear_S
do
seq_len=336
ver=TWM-ReLU
weight=linear

CUDA_VISIBLE_DEVICES=2 python -u run_longExp.py \
  --weather 1 \
  --dataset none \
  --seed 2021 \
  --time_emb 1 \
  --time_channel 40 \
  --emb_dim 8 \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'96'_'emb \
  --model $model_name \
  --data custom \
  --ver $ver \
  --weight $weight \
  --time conv \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 21 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/weather-2021/final'_'$model_name'_'Weather'_'$ver'_'$seq_len'_'96'_'time_sigmoid.log
printf "Done %s 96" ${model_name}


CUDA_VISIBLE_DEVICES=2 python -u run_longExp.py \
  --weather 1 \
  --dataset none \
  --seed 2021 \
  --time_emb 1 \
  --time_channel 40 \
  --emb_dim 8 \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'19'_'emb2 \
  --model $model_name \
  --data custom \
  --ver $ver \
  --weight $weight \
  --time conv \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 21 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate 0.0001 >logs/weather-2021/final'_'$model_name'_'Weather'_'$ver'_'$seq_len'_'192'_'time_sigmoid.log
printf "Done %s 192" ${model_name}


CUDA_VISIBLE_DEVICES=2 python -u run_longExp.py \
  --weather 1 \
  --dataset none \
  --seed 2021 \
  --time_emb 1 \
  --time_channel 40 \
  --emb_dim 8 \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'33'_'emb6 \
  --model $model_name \
  --data custom \
  --ver $ver \
  --weight $weight \
  --time conv \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 21 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/weather-2021/final'_'$model_name'_'Weather'_'$ver'_'$seq_len'_'336'_'time_sigmoid.log
printf "Done %s 336" ${model_name}


CUDA_VISIBLE_DEVICES=2 python -u run_longExp.py \
  --weather 1 \
  --dataset none \
  --seed 2021 \
  --time_emb 1 \
  --time_channel 40 \
  --emb_dim 8 \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'72'_'emb0 \
  --model $model_name \
  --data custom \
  --ver $ver \
  --weight $weight \
  --time conv \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 21 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate 0.0001 >logs/weather-2021/final'_'$model_name'_'Weather'_'$ver'_'$seq_len'_'720'_'time_sigmoid.log
printf "Done %s 720" ${model_name}
done

