if [ "$1" = "only_test" ]; then
    python3 src/prepare_data.py --config configs/preprocess_final.yaml --only_test True
else
    python3 src/prepare_data.py --config configs/preprocess_final.yaml
fi