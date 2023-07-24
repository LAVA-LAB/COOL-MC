#!/bin/bash


interpreter_name="feature_pruner"


python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]" --interpreter="$interpreter_name;0.4;graph_vis.txt;passenger"




python clean_projects.py


