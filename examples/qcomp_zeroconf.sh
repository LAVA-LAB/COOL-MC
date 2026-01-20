#!/bin/bash


#csma.2-2.prism
python cool_mc.py \
  --project_name="qcomp" \
  --algorithm=bc_nn_agent \
  --prism_dir="../prism_files" \
  --prism_file_path="zeroconf.prism" \
  --constant_definitions="N=20,K=4,reset=true" \
  --behavioral_cloning="raw_dataset;../prism_files/zeroconf.prism;Pmax=? [ F (l=4 & ip=1) ];N=20,K=24,reset=true" \
  --bc_epochs=100 \
  --eval_interval=1 \
  --layers=3 \
  --neurons=128 \
  --lr=0.001 \
  --batch_size=32 \
  --num_episodes=2 \
  --seed=42 \
  --deploy=1


python cool_mc.py \
  --parent_run_id="last" \
  --project_name="qcomp" \
  --algorithm=bc_nn_agent \
  --prism_dir="../prism_files" \
  --prism_file_path="zeroconf.prism" \
  --constant_definitions="N=20,K=4,reset=true" \
  --task="rl_model_checking" \
  --prop="P=? [ F (l=4 & ip=1) ]" \
  --seed=42

