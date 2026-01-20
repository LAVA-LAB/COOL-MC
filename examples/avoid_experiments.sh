#python cool_mc.py --task=safe_training --project_name="avoid_experiments" --algorithm=dqn_agent --prism_file_path="avoid.prism" --constant_definitions="xMax=12,yMax=12,slickness=0.1" --prop=""  --reward_flag=1 --seed=128 --epsilon=0.5 --layers=2 --neurons=128 --epsilon_min=0.01  --num_episodes=3647 --eval_interval=100 --epsilon_dec=0.9999 --lr=0.001 --replay_buffer_size=200000 --training_threshold=10000

# Possible with 16GB RAM, but not with naive monolithic model checking.
#P=? [ F<=100 COLLISION=true ]:  0.09587676163584996
#Model Size:             4826809
#Number of Transitions:  143596599
#Model Building Time:    4716.193306922913
#Model Checking Time:    174.98764061927795
#Constant definitions:   xMax=12,yMax=12,slickness=0.1
python cool_mc.py --parent_run_id="9326488d9f324c7cb9ca57525a346c50" --project_name="avoid_experiments" --constant_definitions="xMax=12,yMax=12,slickness=0.1" --prism_dir="../prism_files" --seed=128 --prism_file_path="avoid.prism" --task="rl_model_checking" --prop="P=? [ F<=100 COLLISION=true ]"
# Out of memory (16GB RAM)
python cool_mc.py --parent_run_id="9326488d9f324c7cb9ca57525a346c50" --project_name="avoid_experiments" --constant_definitions="xMax=13,yMax=13,slickness=0.1" --prism_dir="../prism_files" --prism_file_path="avoid.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F<=100 COLLISION=true ]"
# Out of memory (16GB RAM)
python cool_mc.py --parent_run_id="9326488d9f324c7cb9ca57525a346c50" --project_name="avoid_experiments" --constant_definitions="xMax=14,yMax=14,slickness=0.1" --prism_dir="../prism_files" --prism_file_path="avoid.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F<=100 COLLISION=true ]"
