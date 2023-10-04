python cool_mc.py --num_episodes 10000 --project_name="avoid_experiments" --constant_definitions="xMax=4,yMax=4,slickness=0.1" --prism_dir="../prism_files" --prism_file_path="avoid.prism" --seed=128 --rl_algorithm="ppo" --reward_flag=1 --max_steps=100
#P=? [ F<=100 COLLISION=true ]:  0.9487411579712197
#Model Size:             15625
#Number of Transitions:  892165
#Model Building Time:    2767.5066463947296
#Model Checking Time:    0.8593230247497559
#Constant definitions:   xMax=4,yMax=4,slickness=0.1
#python cool_mc.py --parent_run_id="12ccd6371d6f48468f19822f30626170" --project_name="avoid_experiments" --constant_definitions="xMax=4,yMax=4,slickness=0.1" --prism_dir="../prism_files" --seed=128 --prism_file_path="avoid.prism" --task="rl_model_checking" --prop="P=? [ F<=100 COLLISION=true ]"  --preprocessor=""

# Time out.
#python cool_mc.py --parent_run_id="12ccd6371d6f48468f19822f30626170" --project_name="avoid_experiments" --constant_definitions="xMax=11,yMax=11,slickness=0.1" --prism_dir="../prism_files" --seed=128 --prism_file_path="avoid.prism" --task="rl_model_checking" --prop="P=? [ F<=100 COLLISION=true ]"

# Possible with 16GB RAM, but not with naive monolithic model checking nor stochastic policies.
#P=? [ F<=100 COLLISION=true ]:  0.09587676163584996
#Model Size:             4826809
#Number of Transitions:  143596599#
#python cool_mc.py --parent_run_id="last" --project_name="avoid_experiments" --constant_definitions="xMax=12,yMax=12,slickness=0.1" --prism_dir="../prism_files" --seed=128 --prism_file_path="avoid.prism" --task="rl_model_checking" --prop="P=? [ F<=100 COLLISION=true ]"
#Model Building Time:    4716.193306922913
#Model Checking Time:    174.98764061927795
#Constant definitions:   xMax=12,yMax=12,slickness=0.1
#python cool_mc.py --parent_run_id="12ccd6371d6f48468f19822f30626170" --project_name="avoid_experiments" --constant_definitions="xMax=12,yMax=12,slickness=0.1" --prism_dir="../prism_files" --seed=128 --prism_file_path="avoid.prism" --task="rl_model_checking" --prop="P=? [ F<=100 COLLISION=true ]"
# Out of memory (16GB RAM)
#python cool_mc.py --parent_run_id="last" --project_name="avoid_experiments" --constant_definitions="xMax=13,yMax=13,slickness=0.1" --prism_dir="../prism_files" --prism_file_path="avoid.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F<=100 COLLISION=true ]"
# Out of memory (16GB RAM)
#python cool_mc.py --parent_run_id="last" --project_name="avoid_experiments" --constant_definitions="xMax=14,yMax=14,slickness=0.1" --prism_dir="../prism_files" --prism_file_path="avoid.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F<=100 COLLISION=true ]"
