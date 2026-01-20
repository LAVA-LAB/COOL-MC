#!/bin/bash



#csma.2-2.prism
python cool_mc.py \
  --project_name="qcomp" \
  --rl_algorithm=bc_nn_agent \
  --prism_dir="../prism_files" \
  --prism_file_path="csma.2-2.v1.prism" \
  --constant_definitions="" \
  --behavioral_cloning="raw_dataset;../prism_files/csma.2-2.v1.prism;Pmax=? [ !\"collision_max_backoff\" U \"all_delivered\" ];" \
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
  --rl_algorithm=bc_nn_agent \
  --prism_dir="../prism_files" \
  --prism_file_path="csma.2-2.v1.prism" \
  --constant_definitions="" \
  --task="rl_model_checking" \
  --prop="Pmax=? [ !\"collision_max_backoff\" U \"all_delivered\" ]" \
  --seed=42

