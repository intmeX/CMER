# CMER
Context-Aware Emotion Recognition via Caption-Enhanced Network with Adaptive Multi-Stream Feature Attention


This code is partially referenced from https://github.com/Tandon-A/emotic

please check requirements before running
```shell
pip install -r requirements.txt
```

Checkpoint Trained on Emotic: https://drive.google.com/file/d/1R5zG6oEKGQLvywChMAG8emAfbdTxxciS/view?usp=sharing

run test:
```shell
python3 main.py \
--config config/config_quadruple_stream.py \
--data_path /path/to/emotic \
--mode test --model_pretrained checkpoint_emotic.pth \
--fuse_model se_fusion --fuse_L 128 --fuse_r 1 \
--cat_loss_weight 1.0 --cont_loss_weight 0.0 \
--discrete_loss_weight_type dynamic --discrete_loss_type bce \
--batch_size 32 --epochs 10 --learning_rate 5e-4 \
--scheduler exp --gamma 0.9998 --decay_start 0 \
--warmup 0 --fuse_2_layer \
--experiment_name ct_fer_quad_ep10_se_fusion2
```


Checkpoint Trained on CAER-S: https://drive.google.com/file/d/1bevCKtalxwkx_mKVgb3fKlZb4t9TFZOj/view?usp=sharing

run test:
```shell
python3 main.py \
--config config/config_caer_default.py \
--data_path /path/to/caer \
--mode test --model_pretrained checkpoint_caer.pth
--fuse_model se_fusion  --fuse_2_layer --fuse_L 128 --fuse_r 1 \
--discrete_loss_type ce --trainer caer --context_model_frozen \
--optimizer SGD --sgd_momentum 0.9 --num_worker 6 \
--batch_size 1024 --epochs 200 --learning_rate 0.5 \
--scheduler cosine --decay_start 0 --warmup 0 \
T--experiment_name fer_caer_default
```

