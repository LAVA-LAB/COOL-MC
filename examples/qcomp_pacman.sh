#!/bin/bash


#csma.2-2.prism
python cool_mc.py \
  --project_name="qcomp" \
  --algorithm=bc_nn_agent \
  --prism_dir="../prism_files" \
  --prism_file_path="pacman.prism" \
  --constant_definitions="MAXSTEPS=5" \
  --behavioral_cloning="raw_dataset;../prism_files/pacman.prism;Pmin=? [F \"Crash\"];MAXSTEPS=5" \
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
  --prism_file_path="pacman.prism" \
  --constant_definitions="MAXSTEPS=5" \
  --task="rl_model_checking" \
  --prop="P=? [F \"Crash\"]" \
  --seed=42

