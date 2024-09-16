# Feature Rank Comparison
#python cool_mc.py --parent_run_id="54cfe828d1664d9890db735f2360ab6c" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --interpreter="temporal_feature_rank;P=? [ G n_feature>2 ];fuel"
#python cool_mc.py --parent_run_id="54cfe828d1664d9890db735f2360ab6c" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --interpreter="temporal_feature_rank;P=? [ G n_feature>2 ];x"
#python cool_mc.py --parent_run_id="54cfe828d1664d9890db735f2360ab6c" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --interpreter="temporal_feature_rank;P=? [ G n_feature>2 ];y"
#python cool_mc.py --parent_run_id="54cfe828d1664d9890db735f2360ab6c" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]" --interpreter="temporal_feature_rank;P=? [ G n_feature>2 ];fuel"
#python cool_mc.py --parent_run_id="54cfe828d1664d9890db735f2360ab6c" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]" --interpreter="temporal_feature_rank;P=? [ G n_feature>2 ];x"
#python cool_mc.py --parent_run_id="54cfe828d1664d9890db735f2360ab6c" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]" --interpreter="temporal_feature_rank;P=? [ G n_feature>2 ];fuel"



# Temporal Critical State Analysis
#python cool_mc.py --parent_run_id="54cfe828d1664d9890db735f2360ab6c" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]" --interpreter="temporal_critical_state_interpreter;P=? [ F (n_feature=1 & (X n_feature=1)) ];700"
#python cool_mc.py --parent_run_id="54cfe828d1664d9890db735f2360ab6c" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --interpreter="temporal_critical_state_interpreter;P=? [ F (n_feature=1 & (X n_feature=1)) ];700"


# feature rank = n_feature, critical state = n_feature2
#python cool_mc.py --parent_run_id="54cfe828d1664d9890db735f2360ab6c" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --interpreter="temporal_feature_2d;P=? [ n_feature>0 U (n_feature=0 & n_feature2=1 & fuel<4) ];fuel;700"
#python cool_mc.py --parent_run_id="54cfe828d1664d9890db735f2360ab6c" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --interpreter="temporal_feature_2d;P=? [ n_feature>0 U (n_feature=0 & n_feature2=1 & fuel>=4) ];fuel;700"

# Temporal Action Analysis
#python cool_mc.py --parent_run_id="54cfe828d1664d9890db735f2360ab6c" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --interpreter="temporal_action_analysis;P=? [ F (n_feature=0 & (X n_feature=0))]"



# LRA
#python cool_mc.py --parent_run_id="54cfe828d1664d9890db735f2360ab6c" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --interpreter="temporal_feature_2d;LRA=? [ n_feature2=1 ];fuel;700"


# ADVERSARIAL ATTACKS
#python cool_mc.py --parent_run_id="54cfe828d1664d9890db735f2360ab6c" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]" --interpreter="temporal_adv_attacks;P=? [ F (n_feature=1 & (X n_feature=1)) ];5"


# Cleaning Robot
#python cool_mc.py --parent_run_id="a3fc7ab05b98476ca3f4c9a8ab592db9" --project_name="cleaning_system_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="cleaning_system.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F dirt1<0]" --interpreter="temporal_critical_state_interpreter;P=? [ F (n_feature=1 & room_blocked=1) ];700"

# Temporal Action Analysis
#python cool_mc.py --parent_run_id="a3fc7ab05b98476ca3f4c9a8ab592db9" --project_name="cleaning_system_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="cleaning_system.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F dirt1<0]" --interpreter="temporal_action_analysis;P=? [ F (n_feature=0 & (X n_feature=0))]"

# Avoid Experiments
#python cool_mc.py --task=safe_training --project_name="avoid_experiments" --rl_algorithm=dqn_agent --prism_file_path="avoid.prism" --constant_definitions="xMax=12,yMax=12,slickness=0.1" --prop=""  --reward_flag=1 --seed=128 --epsilon=0.5 --layers=2 --neurons=128 --epsilon_min=0.01  --num_episodes=3647 --eval_interval=100 --epsilon_dec=0.9999 --lr=0.001 --replay_buffer_size=200000 --training_threshold=10000
# 738491a041614574b9d6a6501a915356 FOR temporal explainability
#python cool_mc.py --parent_run_id="738491a041614574b9d6a6501a915356" --project_name="avoid_experiments" --constant_definitions="xMax=4,yMax=4,slickness=0.1" --prism_dir="../prism_files" --seed=128 --prism_file_path="avoid.prism" --task="rl_model_checking" --prop="P=? [ F<=100 COLLISION=true ]" --interpreter="temporal_critical_state_interpreter;P=? [ F (n_feature=1) ];100"