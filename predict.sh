i=0
for checkpoint in $(ls ./work_dirs/speed_final/checkpoints/best_mse*.ckpt | sort); do
    python3 src/test_pl.py \
        --data_config configs/preprocess_final.yaml \
        --train_config_speed configs/speed_final.yaml \
        --speed_checkpoint "$checkpoint" \
        --output_path "speed_${i}" \
        --run_test True
    i=$((i+1))
done

i=0
for checkpoint in $(ls ./work_dirs/yaw_final/checkpoints/best_mse*.ckpt | sort); do
    python3 src/test_pl.py \
        --data_config configs/preprocess_final.yaml \
        --train_config_yaw configs/yaw_final.yaml \
        --yaw_checkpoint "$checkpoint" \
        --output_path "yaw_${i}" \
        --run_test True
    i=$((i+1))
done

python3 src/blend.py --data_config configs/preprocess_final.yaml
