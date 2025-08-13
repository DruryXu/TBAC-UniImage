torchrun --nproc-per-node=8 \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --output_dir path/to/output/dir \
    --config_file pretrain.yaml \
    --base_dir path/to/base/dir \