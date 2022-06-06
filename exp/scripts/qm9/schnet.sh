#!/bin/bash

python -m exp.run_mol_exp \
  --device 0 \
  --start_seed 2 \
  --stop_seed 4 \
  --init_method sum \
  --readout mean \
  --final_readout sum \
  --emb_dim 64 \
  --exp_name cwn \
  --dataset QM9 \
  --train_eval_period 10 \
  --epochs 200 \
  --batch_size 128 \
  --drop_rate 0.0 \
  --drop_position lin2 \
  --max_dim 2 \
  --lr 0.0001 \
  --graph_norm bn \
  # --model qm9_embed_sparse_cin \
  --model qm9_schnet \
  --nonlinearity relu \
  --num_layers 2 \
  --max_ring_size 18 \
  --task_type regression \
  --eval_metric mae \
  --minimize \
  --lr_scheduler 'ReduceLROnPlateau' \
  --use_coboundaries True \
  --use_edge_features \
  --early_stop \
  --lr_scheduler_patience 10 \
  --dump_curves \
  --preproc_jobs 32
