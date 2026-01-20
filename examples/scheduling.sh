#!/bin/bash


#csma.2-2.prism
python cool_mc.py \
  --project_name="scheduling" \
  --algorithm=bc_nn_agent \
  --prism_dir="../prism_files" \
  --prism_file_path="scheduling_task.prism" \
  --constant_definitions="" \
  --behavioral_cloning="raw_dataset;../prism_files/scheduling_task.prism;Pmax=? [ F \"jobs_done\" ];" \
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
  --project_name="scheduling" \
  --algorithm=bc_nn_agent \
  --prism_dir="../prism_files" \
  --prism_file_path="scheduling_task.prism" \
  --constant_definitions="" \
  --task="rl_model_checking" \
  --prop="P=? [ F \"jobs_done\" ]" \
  --seed=42

