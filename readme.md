Solution for Yandex Cup 24 Self-Driving Cars.

**Hardware**:

- Python 3.10.12 [GCC 11.4.0] on linux
- RAM: 32GB
- GPU: 2080ti
- nvidia-driver: 535.161.07
- CUDA: 12.2

time for train ~4 hours

**Run**:

change data folder in `configs/preprocess_final.yaml` ( `input_data.root_data_folder` )

```
    python3 -m venv drive_env
    source drive_env/bin/activate
    pip install -r requirements.txt

    sh prepare_data.sh

    sh train.sh

    # here its better to run just one checkpoint, but my last sub was blend
    # so lets wait a bit more and run all of them ( why did i even decide to mean yaw ((  )
    sh predict.sh

    # U r amazing
```

**Inference:**

for inference only place ckpts to `work_dirs/yaw_final/checkpoints` and `work_dirs/speed_final/checkpoints` accordingly [Ckpts weights - git release](https://github.com/ktverdov/yacup-24-selfdriving/releases/tag/submission)

```
    sh prepare_data.sh only_test
    sh predict.sh
```

```
wget https://github.com/ktverdov/yacup-24-selfdriving/releases/download/submission/speed_final.zip
mkdir -p ./work_dirs/speed_final/checkpoints/
unzip speed_final.zip -d ./work_dirs/speed_final/checkpoints/

wget https://github.com/ktverdov/yacup-24-selfdriving/releases/download/submission/yaw_final.zip
mkdir -p ./work_dirs/yaw_final/checkpoints/
unzip yaw_final.zip -d ./work_dirs/yaw_final/checkpoints/
```