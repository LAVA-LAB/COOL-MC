# Taxi PPO

python cool_mc.py \
    --project_name="taxi_ppo_shielded_training" \
    --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" \
    --prism_dir="../prism_files" \
    --prism_file_path="transporter_with_rewards.prism" \
    --seed=128 \
    --algorithm="ppo_agent" \
    --layers=4 \
    --neurons=512 \
    --lr=0.0003 \
    --batch_size=64 \
    --num_episodes=100 \
    --eval_interval=100 \
    --gamma=0.99 \
    --reward_flag=1 \
    --wrong_action_penalty=0 \
    --prop="" \
    --max_steps=100 \
    --behavioral_cloning="raw_dataset;../prism_files/transporter_with_rewards.prism;Pmax=? [ F jobs_done=2 ];MAX_JOBS=2,MAX_FUEL=10" \
    --bc_epochs=65 \
    --deploy=1

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="taxi_ppo_shielded_training" \
    --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" \
    --prism_dir="../prism_files" \
    --prism_file_path="transporter_with_rewards.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop="P=? [ F jobs_done=2 ]"


