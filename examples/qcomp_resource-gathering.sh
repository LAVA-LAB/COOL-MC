#!/bin/bash


#csma.2-2.prism
python cool_mc.py \
  --project_name="qcomp" \
  --algorithm=bc_nn_agent \
  --prism_dir="../prism_files" \
  --prism_file_path="resource_gathering.prism" \
  --constant_definitions="B=50,GOLD_TO_COLLECT=5,GEM_TO_COLLECT=5" \
  --behavioral_cloning="raw_dataset;../prism_files/resource_gathering.prism;Pmax=? [F \"success\"];B=50,GOLD_TO_COLLECT=5,GEM_TO_COLLECT=5" \
  --bc_epochs=25 \
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
  --prism_file_path="resource_gathering.prism" \
  --constant_definitions="B=50,GOLD_TO_COLLECT=5,GEM_TO_COLLECT=5" \
  --task="rl_model_checking" \
  --prop="P=? [F \"success\"]" \
  --seed=42

