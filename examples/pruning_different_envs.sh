#python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ ((true U (x=red_x & y=red_y)) U passenger=true)  U (gas_x=x & gas_y=y) ]" --interpreter="feature_pruner;1;fuel"
#python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --interpreter="feature_pruner;1;fuel"
#python clean_projects.py
#python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --interpreter="feature_pruner;1;passenger_loc_x"
#python clean_projects.py

#python cool_mc.py --parent_run_id="62d9a26e471c444fa04e5f981cb42a54" --project_name="freeway_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F TOP_MIDDLE_CELL=true ]" --interpreter="feature_pruner;1;px0"
#python clean_projects.py

python cool_mc.py --parent_run_id="3866b9ab830c44c6bd8949d825af5694" --project_name="avoid_experiments" --constant_definitions="xMax=4,yMax=4,slickness=0.1" --prism_dir="../prism_files" --seed=128 --prism_file_path="avoid.prism" --task="rl_model_checking" --prop="P=? [ F<=100 COLLISION=true ]" --interpreter="feature_pruner;1;x"
python clean_projects.py

python cool_mc.py --parent_run_id="4474e97b8a854ea591f3d584394fcb3a" --project_name="crazy_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="crazy_climber.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F COLLISION=true ]" --interpreter="feature_pruner;1;px1"
python clean_projects.py
