#python cool_mc.py --task=safe_training --project_name="ttt" --rl_algorithm=turn_ppo --prism_file_path="ttt.prism" --constant_definitions="" --prop="" --reward_flag=1 --seed=128 --epsilon=0.5 --layers=2 --neurons=128 --epsilon_min=0.01  --num_episodes=101 --eval_interval=100
#python cool_mc.py --parent_run_id="last" --task=rl_model_checking --project_name="ttt" --constant_definitions="" --prop="P=? [PLAYER1_WON = true ]"
#python cool_mc.py --parent_run_id="last" --task=rl_model_checking --project_name="ttt" --constant_definitions="" --prop="P=? [F PLAYER2_WON = true ]"
# ALL_BLOCKED & A_PLAYER_ONE=false & cell00!=3
python cool_mc.py --parent_run_id="last" --task=rl_model_checking --project_name="ttt" --constant_definitions="" --prop="P=? [F ALL_BLOCKED & A_PLAYER_ONE=false & cell00!=3 ]"
