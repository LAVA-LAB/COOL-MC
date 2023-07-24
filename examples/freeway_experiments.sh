# 424bd610a4184c65a3510468d4ba6132,64a784ae6aaa49d6b35d31654af77603
#python cool_mc.py --project_name="freeway_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="freeway.prism" --seed=128 --layers=3 --neurons=512 --lr=0.0001 --batch_size=32 --num_episodes=1000000 --eval_interval=100 --epsilon_dec=0.99999 --epsilon_min=0.1 --gamma=0.99 --epsilon=1 --replace=301 --reward_flag=1 --wrong_action_penalty=0 --prop="" --max_steps=100 --replay_buffer_size=300000

python cool_mc.py --parent_run_id="62d9a26e471c444fa04e5f981cb42a54" --project_name="freeway_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F TOP_MIDDLE_CELL=true ]" --interpreter=""
#python cool_mc.py --parent_run_id="last" --project_name="freeway_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]" --interpreter=""
#python cool_mc.py --parent_run_id="last" --project_name="freeway_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --interpreter="ln_unstructured_pruning_interpreter"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]"
# Remapping
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --preprocessor="feature_remapping;fuel=[0,1]"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --preprocessor="feature_remapping;fuel=[0,1,2]"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --preprocessor="feature_remapping;fuel=[0,1,2,3]"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --preprocessor="feature_remapping;fuel=[0,1,2,3,4]"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --preprocessor="feature_remapping;fuel=[0,1,2,3,4,5]"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --preprocessor="feature_remapping;fuel=[0,1,2,3,4,5,6]"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --preprocessor="feature_remapping;fuel=[0,1,2,3,4,5,6,7]"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --preprocessor="feature_remapping;fuel=[0,1,2,3,4,5,6,7,8]"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --preprocessor="feature_remapping;fuel=[0,1,2,3,4,5,6,7,8,9]"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --preprocessor="feature_remapping;fuel=[0,1,2,3,4,5,6,7,8,9,10]"
# Different Constant definitions
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=1" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=2" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=3" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=4" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=5" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=6" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=7" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=8" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=9" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]"
# Permissive Policy done2
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmin=? [ F jobs_done=2 ]" --preprocessor="policy_abstraction;fuel=[9,11]"

#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmin=? [ F jobs_done=2 ]" --preprocessor="policy_abstraction;fuel=[8,11]"


#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmin=? [ F jobs_done=2 ]" --preprocessor="policy_abstraction;fuel=[7,11]"


#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmin=? [ F jobs_done=2 ]" --preprocessor="policy_abstraction;fuel=[6,11]"


#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmin=? [ F jobs_done=2 ]" --preprocessor="policy_abstraction;fuel=[5,11]"


#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmin=? [ F jobs_done=2 ]" --preprocessor="policy_abstraction;fuel=[4,11]"


#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmin=? [ F jobs_done=2 ]" --preprocessor="policy_abstraction;fuel=[3,11]"


#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmin=? [ F jobs_done=2 ]" --preprocessor="policy_abstraction;fuel=[2,11]"


#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmin=? [ F jobs_done=2 ]" --preprocessor="policy_abstraction;fuel=[1,11]"

#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmin=? [ F jobs_done=2 ]" --preprocessor="policy_abstraction;fuel=[0,11]"

# Permissive Policy done1
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmin=? [ F jobs_done=1 ]" --preprocessor="policy_abstraction;fuel=[9,11]"

#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmin=? [ F jobs_done=1 ]" --preprocessor="policy_abstraction;fuel=[8,11]"


#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmin=? [ F jobs_done=1 ]" --preprocessor="policy_abstraction;fuel=[7,11]"


#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmin=? [ F jobs_done=1 ]" --preprocessor="policy_abstraction;fuel=[6,11]"


#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmin=? [ F jobs_done=1 ]" --preprocessor="policy_abstraction;fuel=[5,11]"


#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmin=? [ F jobs_done=1 ]" --preprocessor="policy_abstraction;fuel=[4,11]"


#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmin=? [ F jobs_done=1 ]" --preprocessor="policy_abstraction;fuel=[3,11]"


#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmin=? [ F jobs_done=1 ]" --preprocessor="policy_abstraction;fuel=[2,11]"


#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmin=? [ F jobs_done=1 ]" --preprocessor="policy_abstraction;fuel=[1,11]"

#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmin=? [ F jobs_done=1 ]" --preprocessor="policy_abstraction;fuel=[0,11]"

# Permissive Policy empty
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmax=? [ F \"empty\" ]" --preprocessor="policy_abstraction;fuel=[9,11]"

#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmax=? [ F \"empty\" ]" --preprocessor="policy_abstraction;fuel=[8,11]"


#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmax=? [ F \"empty\" ]" --preprocessor="policy_abstraction;fuel=[7,11]"


#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmax=? [ F \"empty\" ]" --preprocessor="policy_abstraction;fuel=[6,11]"


#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmax=? [ F \"empty\" ]" --preprocessor="policy_abstraction;fuel=[5,11]"


#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmax=? [ F \"empty\" ]" --preprocessor="policy_abstraction;fuel=[4,11]"


#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmax=? [ F \"empty\" ]" --preprocessor="policy_abstraction;fuel=[3,11]"


#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmax=? [ F \"empty\" ]" --preprocessor="policy_abstraction;fuel=[2,11]"


#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmax=? [ F \"empty\" ]" --preprocessor="policy_abstraction;fuel=[1,11]"

#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmax=? [ F \"empty\" ]" --preprocessor="policy_abstraction;fuel=[0,11]"

# FFGSM
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --preprocessor="ffgsm;1;fuel"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]" --preprocessor="ffgsm;1;fuel"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F  \"empty\" ]" --preprocessor="ffgsm;1;fuel"

# FGSM
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --preprocessor="fgsm;0.1"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]" --preprocessor="fgsm;0.1"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F  \"empty\" ]" --preprocessor="fgsm;0.1"

#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --preprocessor="fgsm;0.1#rounder;round"

#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]" --preprocessor="fgsm;0.1#rounder;round"

#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]" --preprocessor="fgsm;0.1#rounder;round"

#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --preprocessor="fgsm;0.1#rounder;floor"

#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]" --preprocessor="fgsm;0.1#rounder;floor"

#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]" --preprocessor="fgsm;0.1#rounder;floor"

# DeepFool
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --preprocessor="deepfool;0.01;50"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]" --preprocessor="deepfool;0.01;50"
#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]" --preprocessor="deepfool;0.01;50"

#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --preprocessor="deepfool;0.01;50#rounder;round"

#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]" --preprocessor="deepfool;0.01;50#rounder;round"

#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]" --preprocessor="deepfool;0.01;50#rounder;round"

#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --preprocessor="deepfool;0.01;50#rounder;floor"

#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]" --preprocessor="deepfool;0.01;50#rounder;floor"

#python cool_mc.py --parent_run_id="last" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]" --preprocessor="deepfool;0.01;50#rounder;floor"


#python cool_mc.py --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --layers=4 --neurons=512 --lr=0.0001 --batch_size=32 --num_episodes=10000 --eval_interval=100 --epsilon_dec=0.99999 --epsilon_min=0.1 --gamma=0.99 --epsilon=1 --replace=301 --reward_flag=0 --wrong_action_penalty=0 --prop="Pmax=? [ F jobs_done=2 ]" --max_steps=100 --replay_buffer_size=300000 --replace=304 --postprocessor="state_nstate_swapper"

#python cool_mc.py --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --layers=4 --neurons=512 --lr=0.0001 --batch_size=32 --num_episodes=10000 --eval_interval=100 --epsilon_dec=0.99999 --epsilon_min=0.1 --gamma=0.99 --epsilon=1 --replace=301 --reward_flag=0 --wrong_action_penalty=0 --prop="Pmax=? [ F jobs_done=2 ]" --max_steps=100 --replay_buffer_size=10000 --replace=304


#python cool_mc.py --parent_run_id="424bd610a4184c65a3510468d4ba6132" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmax=? [ F \"empty\" ]" --preprocessor="integer_l1_robustness;1;fuel"

#python cool_mc.py --parent_run_id="424bd610a4184c65a3510468d4ba6132" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmax=? [ F \"empty\" ]" --preprocessor="integer_l1_robustness;1;passenger_loc_x"

#python cool_mc.py --parent_run_id="424bd610a4184c65a3510468d4ba6132" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmax=? [ F \"empty\" ]" --preprocessor="integer_l1_robustness;1;done"

#python cool_mc.py --parent_run_id="424bd610a4184c65a3510468d4ba6132" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmax=? [ F \"empty\" ]" --preprocessor="integer_l1_robustness;2;fuel"
