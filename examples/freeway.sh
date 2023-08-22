#python cool_mc.py --project_name="freeway_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="freeway.prism" --seed=128 --num_episodes=1000 --eval_interval=100 --epsilon_dec=0.99999 --epsilon_min=0.1 --gamma=0.99 --epsilon=1 --replace=301 --reward_flag=1 --wrong_action_penalty=0 --prop="" --max_steps=20 --rl_algorithm=ppo
#P=? [ F "goal" ]:       0.700000000120131
#Model Size:             496
#Number of Transitions:  2080
#Model Building Time:    23.748758792877197
#Model Checking Time:    0.009535551071166992
python cool_mc.py --parent_run_id="last" --project_name="freeway_experiments" --constant_definitions="" --prism_dir="../prism_files" --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" --preprocessor="old"




