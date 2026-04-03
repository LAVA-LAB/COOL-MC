# Transition Updater Example
# 1) Train a DQN agent on the transporter environment (short run)
# 2) Run model checking WITHOUT transition updater (baseline)
# 3) Run model checking WITH epsilon transition updater (interval model)

# Step 1: Train
python cool_mc.py --num_episodes 102 --project_name="transition_updater_example" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128

# Step 2: Model check (standard)
python cool_mc.py --parent_run_id="last" --num_episodes 102 --project_name="transition_updater_example" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]"

# Step 3: Model check with epsilon transition updater (eps=0.05)
python cool_mc.py --parent_run_id="last" --num_episodes 102 --project_name="transition_updater_example" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]" --transition_updater="epsilon;eps=0.05"
