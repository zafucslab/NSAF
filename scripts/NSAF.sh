export CUDA_VISIBLE_DEVICES=0

model_name=NSAF

for target in WS_1.5m WD_1.5m Ta_1.5m Ta_2.8m RH_1.5m RH_2.8m Vapor_1.5m Vapor_2.8m P RainSnow PAR Rn Ts_0cm Ts_20cm Ts_50cm Ts_100cm Ts_200cm Swc_0cm Swc_20cm Swc_50cm Swc_100cm Swc_200cm H LE Fc; do
  echo target: $target >>result_long_term_forecast.txt

  for pred_len in 96 192 336 720; do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path dataset.csv \
      --model_id NSAF_96_$pred_len_$target \
      --model $model_name \
      --data custom \
      --features MS \
      --seq_len 96 \
      --pred_len $pred_len \
      --enc_in 25 \
      --cycle 24 \
      --model_type linear \
      --train_epochs 10 \
      --patience 5 \
      --batch_size 256 \
      --learning_rate 0.001 --target $target \
      --itr 1
  done
done

