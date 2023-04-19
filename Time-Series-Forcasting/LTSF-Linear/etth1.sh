if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ETT-2021" ]; then
    mkdir ./logs/ETT-2021
fi

for model_name in Linear NLinear DLinear #DLinear_T DLinear_S
do
for ver in B
do
seq_len=336

CUDA_VISIBLE_DEVICES=4 python -u run_longExp.py \
  --dataset none \
  --seed 2021 \
  --time_emb 0 \
  --is_training 1 \
  --root_path /nas/datahub/ETT/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'96 \
  --model $model_name \
  --data ETTh1 \
  --ver $ver \
  --time_channel 4 \
  --features S \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/ETT-2021/$model_name'_'Etth1'_'336'_'$ver'_'96'_'time_sigmoid.log
printf "Done %s %s 96" ${model_name} ${ver}

CUDA_VISIBLE_DEVICES=4 python -u run_longExp.py \
  --dataset none \
  --seed 2021 \
  --time_emb 0 \
  --is_training 1 \
  --root_path /nas/datahub/ETT/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'192 \
  --model $model_name \
  --data ETTh1 \
  --features S \
  --time_channel 4 \
  --ver $ver \
  --embed none \
  --seq_len 336 \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/ETT-2021/$model_name'_'Etth1'_'336'_'$ver'_'192'_'time_sigmoid.log
printf "Done %s %s 96" ${model_name} ${ver}

CUDA_VISIBLE_DEVICES=4 python -u run_longExp.py \
  --dataset none \
  --seed 2021 \
  --time_emb 0 \
  --is_training 1 \
  --root_path /nas/datahub/ETT/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'336 \
  --model $model_name \
  --data ETTh1 \
  --features S \
  --ver $ver \
  --time_channel 4 \
  --embed none \
  --seq_len 336 \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/ETT-2021/$model_name'_'Etth1'_'336'_'$ver'_'336'_'time_sigmoid.log
printf "Done %s %s 96" ${model_name} ${ver}

CUDA_VISIBLE_DEVICES=4 python -u run_longExp.py \
  --dataset none \
  --seed 2021 \
  --time_emb 0 \
  --is_training 1 \
  --root_path /nas/datahub/ETT/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'720 \
  --model $model_name \
  --data ETTh1 \
  --features S \
  --ver $ver \
  --time_channel 4 \
  --embed none \
  --seq_len 336 \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/ETT-2021/$model_name'_'Etth1'_'336'_'$ver'_'720'_'time_sigmoid.log
printf "Done %s %s 96" ${model_name} ${ver}
done
done