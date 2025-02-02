python3 ../../main_contrastive.py \
    --dataset imagenet100 \
    --encoder resnet18 \
    --data_folder /datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/val \
    --max_epochs 100 \
    --gpus 0,1 \
    --distributed_backend ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --num_workers 8 \
    --min_scale_crop 0.2 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --solarization_prob 0.1 \
    --name vicreg \
    --project contrastive_learning \
    --wandb \
    --method vicreg \
    --proj_hidden_dim 2048 \
    --output_dim 2048 \
    --sim_loss_weight 25.0 \
    --var_loss_weight 25.0 \
    --cov_loss_weight 1.0
