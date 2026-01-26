# Taxi PPO with shielding example
#
# A) SHIELDED TRAINING
#    1. Pre-train PPO agent with behavioral cloning from optimal policy
#    2. Fine-tune with PPO WITH post-shielding (actions corrected during training)
#    3. Verify the trained agent
#
# Two training modes:
# A) UNSHIELDED TRAINING (default)
#    1. Pre-train PPO agent with behavioral cloning from optimal policy
#    2. Fine-tune with PPO (no shielding during training)
#    3. Verify the trained agent


##############################################################################
# A) SHIELDED TRAINING
##############################################################################
# The shield corrects unsafe actions during training using optimal policy

# Step 1: Behavioral cloning pre-training with post-shielding
python cool_mc.py \
    --project_name="taxi_ppo_shielded_training" \
    --constant_definitions="MAX_JOBS=3,MAX_FUEL=10" \
    --prism_dir="../prism_files" \
    --prism_file_path="transporter_with_rewards.prism" \
    --seed=128 \
    --algorithm="ppo_agent" \
    --layers=4 \
    --neurons=512 \
    --lr=0.0003 \
    --batch_size=64 \
    --num_episodes=2500 \
    --eval_interval=100 \
    --gamma=0.99 \
    --reward_flag=1 \
    --wrong_action_penalty=0 \
    --prop="Pmax=? [ F jobs_done=3 ]" \
    --max_steps=10000 \
    --behavioral_cloning="raw_dataset;../prism_files/transporter_with_rewards.prism;Pmax=? [ F jobs_done=2 ];MAX_JOBS=2,MAX_FUEL=10" \
    --bc_epochs=75 \
    --postprocessor="post_shielding;raw_dataset;../prism_files/transporter_with_rewards.prism;Pmax=? [ F jobs_done=3 ];MAX_JOBS=3,MAX_FUEL=10"

# Step 2: Verify the shielded-trained agent
python cool_mc.py \
    --parent_run_id="last" \
    --project_name="taxi_ppo_shielded_training" \
    --constant_definitions="MAX_JOBS=3,MAX_FUEL=10" \
    --prism_dir="../prism_files" \
    --prism_file_path="transporter_with_rewards.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop="P=? [ F jobs_done=3 ]"

##############################################################################
# B) UNSHIELDED TRAINING
##############################################################################



python cool_mc.py \
    --project_name="taxi_ppo_shielded_training" \
    --constant_definitions="MAX_JOBS=3,MAX_FUEL=10" \
    --prism_dir="../prism_files" \
    --prism_file_path="transporter_with_rewards.prism" \
    --seed=128 \
    --algorithm="ppo_agent" \
    --layers=4 \
    --neurons=512 \
    --lr=0.0003 \
    --batch_size=64 \
    --num_episodes=2500 \
    --eval_interval=100 \
    --gamma=0.99 \
    --reward_flag=1 \
    --wrong_action_penalty=0 \
    --prop="Pmax=? [ F jobs_done=3 ]" \
    --max_steps=10000 \
    --behavioral_cloning="raw_dataset;../prism_files/transporter_with_rewards.prism;Pmax=? [ F jobs_done=2 ];MAX_JOBS=2,MAX_FUEL=10" \
    --bc_epochs=75

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="taxi_ppo_shielded_training" \
    --constant_definitions="MAX_JOBS=3,MAX_FUEL=10" \
    --prism_dir="../prism_files" \
    --prism_file_path="transporter_with_rewards.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop="P=? [ F jobs_done=3 ]"


