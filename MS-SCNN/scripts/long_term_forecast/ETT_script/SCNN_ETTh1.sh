export CUDA_VISIBLE_DEVICES=0

model_name=MS-SCNN

# ETTh1 预测长度 3 的最优配置
python -u my_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_168_3_optimal \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 3 \
  --cycle_len 24 \
  --short_period_len 8 \
  --kernel_size 2 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 8 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --lradj 'type3' \
  --itr 1 \
  --train_epoch 200 \
  --patience 5 \
  --reduction_ratio 4 \
  --season_dilations "[1,4,7]" \
  --long_dilations "[2,5,8,11]" \
  --short_dilations "[1,2]" \
  --long_kernel_size 5 \
  --season_kernel_size 3 \
  --short_kernel_size 2

# ETTh1 预测长度 24 的最优配置
python -u my_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_168_24_optimal \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 24 \
  --cycle_len 24 \
  --short_period_len 8 \
  --kernel_size 2 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 8 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --lradj 'type3' \
  --itr 1 \
  --train_epoch 200 \
  --patience 5 \
  --reduction_ratio 4 \
  --season_dilations "[1,3,5]" \
  --long_dilations "[1,3,5,7]" \
  --short_dilations "[2,4]" \
  --long_kernel_size 5 \
  --season_kernel_size 5 \
  --short_kernel_size 3

# ETTh1 预测长度 96 的最优配置
python -u my_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_168_96_optimal \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 96 \
  --cycle_len 24 \
  --short_period_len 8 \
  --kernel_size 2 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 8 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --lradj 'type3' \
  --itr 1 \
  --train_epoch 200 \
  --patience 5 \
  --reduction_ratio 2 \
  --season_dilations "[1,3,5]" \
  --long_dilations "[1,4,7,10]" \
  --short_dilations "[1,3]" \
  --long_kernel_size 5 \
  --season_kernel_size 5 \
  --short_kernel_size 3

# ETTh1 预测长度 192 的最优配置
python -u my_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_168_192_optimal \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 192 \
  --cycle_len 24 \
  --short_period_len 8 \
  --kernel_size 2 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 8 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --lradj 'type3' \
  --itr 1 \
  --train_epoch 200 \
  --patience 5 \
  --reduction_ratio 2 \
  --season_dilations "[1,3,5]" \
  --long_dilations "[2,5,8,11]" \
  --short_dilations "[2,4]" \
  --long_kernel_size 5 \
  --season_kernel_size 5 \
  --short_kernel_size 2

