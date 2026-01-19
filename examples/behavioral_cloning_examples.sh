#!/bin/bash

python cool_mc.py \
  --project_name="bc_taxi_examples" \
  --rl_algorithm=bc_nn_agent \
  --prism_dir="../prism_files" \
  --prism_file_path="transporter.prism" \
  --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" \
  --behavioral_cloning="raw_dataset;../prism_files/transporter.prism;Pmax=? [ F jobs_done=2 ];MAX_JOBS=2,MAX_FUEL=10" \
  --bc_epochs=25 \
  --eval_interval=100 \
  --layers=3 \
  --neurons=128 \
  --lr=0.001 \
  --batch_size=32 \
  --num_episodes=101 \
  --prop="Pmax=? [ F jobs_done=2 ]" \
  --seed=42 \
  --deploy=1


python cool_mc.py \
  --parent_run_id="last" \
  --project_name="bc_taxi_examples" \
  --rl_algorithm=bc_nn_agent \
  --prism_dir="../prism_files" \
  --prism_file_path="transporter.prism" \
  --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" \
  --task="rl_model_checking" \
  --prop="P=? [ F jobs_done=2 ]" \
  --seed=42


python cool_mc.py \
  --parent_run_id="last" \
  --project_name="bc_taxi_examples" \
  --rl_algorithm=bc_nn_agent \
  --prism_dir="../prism_files" \
  --prism_file_path="transporter.prism" \
  --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" \
  --task="rl_model_checking" \
  --prop="P=? [ F jobs_done=2 ]" \
  --interpreter="decision_tree" \
  --seed=42

