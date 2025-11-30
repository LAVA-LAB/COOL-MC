# Frozen Lake
python cool_mc.py --rl_algorithm="qreinforce" --num_episodes 10000 --project_name="frozen_lake_experiment" --constant_definitions="start_position=0,control=0.99" --prism_dir="../prism_files" --prism_file_path="frozen_lake.prism" --seed=124412 --reward_flag=1 --max_steps=500 2>&1 | tee -a quantum_frozen.log
python cool_mc.py --parent_run_id="last" --project_name="frozen_lake_experiment" --constant_definitions="start_position=0,control=0.99" --prism_dir="../prism_files" --prism_file_path="frozen_lake.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"at_frisbee\" ]" 2>&1 | tee -a quantum_frozen.log
python cool_mc.py --parent_run_id="last" --project_name="frozen_lake_experiment" --constant_definitions="start_position=0,control=0.99" --prism_dir="../prism_files" --prism_file_path="frozen_lake.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ pos<=3 U pos=7 ]" 2>&1 | tee -a quantum_frozen.log

# Bit-flip noise range
for p in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1; do python cool_mc.py --preprocessor "quantum_bit_flip:$p" --parent_run_id="last" --project_name="frozen_lake_experiment" --prism_dir="../prism_files" --prism_file_path="frozen_lake.prism" --constant_definitions="start_position=0,control=0.99" --task="rl_model_checking" --prop="P=? [ F \"at_frisbee\" ]" 2>&1 | tee -a frozen_noise_range1.log; done

# Phase-flip noise range
for p in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1; do python cool_mc.py --preprocessor "quantum_phase_flip:$p" --parent_run_id="last" --project_name="frozen_lake_experiment" --prism_dir="../prism_files" --prism_file_path="frozen_lake.prism" --constant_definitions="start_position=0,control=0.99" --task="rl_model_checking" --prop="P=? [ F \"at_frisbee\" ]" 2>&1 | tee -a frozen_noise_range2.log; done

# Depolarizing noise range
for p in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1; do python cool_mc.py --preprocessor "quantum_depolarizing_noise:$p" --parent_run_id="last" --project_name="frozen_lake_experiment" --prism_dir="../prism_files" --prism_file_path="frozen_lake.prism" --constant_definitions="start_position=0,control=0.99" --task="rl_model_checking" --prop="P=? [ F \"at_frisbee\" ]" 2>&1 | tee -a frozen_noise_range3.log; done

# Amplitude damping noise range
for p in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1; do python cool_mc.py --preprocessor "quantum_amplitude_damping:$p" --parent_run_id="last" --project_name="frozen_lake_experiment" --prism_dir="../prism_files" --prism_file_path="frozen_lake.prism" --constant_definitions="start_position=0,control=0.99" --task="rl_model_checking" --prop="P=? [ F \"at_frisbee\" ]" 2>&1 | tee -a frozen_noise_range4.log; done


python cool_mc.py --rl_algorithm="reinforce" --num_episodes 10000 --project_name="frozen_lake_experiment" --constant_definitions="start_position=0,control=0.99" --prism_dir="../prism_files" --prism_file_path="frozen_lake.prism" --seed=124412 --reward_flag=1 --max_steps=500 2>&1 | tee -a quantum_frozen.log
python cool_mc.py --parent_run_id="last" --project_name="frozen_lake_experiment" --constant_definitions="start_position=0,control=0.99" --prism_dir="../prism_files" --prism_file_path="frozen_lake.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"at_frisbee\" ]" 2>&1 | tee -a quantum_frozen.log
python cool_mc.py --parent_run_id="last" --project_name="frozen_lake_experiment" --constant_definitions="start_position=0,control=0.99" --prism_dir="../prism_files" --prism_file_path="frozen_lake.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ pos<=3 U pos=7 ]" 2>&1 | tee -a quantum_frozen.log


# Quantum REINFORCE
python cool_mc.py --rl_algorithm="qreinforce" --num_episodes 10000 --project_name="ski_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="ski2.prism" --reward_flag=1 --seed=128 --lr=0.01 --num_qubits=4 --layers=2 2>&1 | tee -a quantum_ski.log
python cool_mc.py --parent_run_id="last" --project_name="ski_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="ski2.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" 2>&1 | tee -a quantum_ski.log
#python cool_mc.py --preprocessor "quantum_bit_flip:0.01" --parent_run_id="last" --project_name="ski_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="ski2.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" 2>&1 | tee -a quantum_ski.log
#python cool_mc.py --preprocessor "quantum_depolarizing_noise:0.01" --parent_run_id="last" --project_name="ski_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="ski2.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" 2>&1 | tee -a quantum_ski.log
#python cool_mc.py --preprocessor "quantum_phase_flip:0.005" --parent_run_id="last" --project_name="ski_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="ski2.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" 2>&1 | tee -a quantum_ski.log
#python cool_mc.py --preprocessor "quantum_generalized_amplitude_damping:0.01:0.1" --parent_run_id="last" --project_name="ski_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="ski2.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" 2>&1 | tee -a quantum_ski.log
#python cool_mc.py --preprocessor "quantum_depolarizing_noise:0.01#quantum_bit_flip:0.005#quantum_phase_flip:0.005" --parent_run_id="last" --project_name="ski_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="ski2.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" 2>&1 | tee -a quantum_ski.log


# Bit-flip noise range
for p in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1; do python cool_mc.py --preprocessor "quantum_bit_flip:$p" --parent_run_id="last" --project_name="ski_experiments" --prism_dir="../prism_files" --prism_file_path="ski2.prism" --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" 2>&1 | tee -a noise_range1.log; done

# Phase-flip noise range
for p in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1; do python cool_mc.py --preprocessor "quantum_phase_flip:$p" --parent_run_id="last" --project_name="ski_experiments" --prism_dir="../prism_files" --prism_file_path="ski2.prism" --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" 2>&1 | tee -a noise_range2.log; done

# Depolarizing noise range
for p in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1; do python cool_mc.py --preprocessor "quantum_depolarizing_noise:$p" --parent_run_id="last" --project_name="ski_experiments" --prism_dir="../prism_files" --prism_file_path="ski2.prism" --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" 2>&1 | tee -a noise_range3.log; done

# Amplitude damping noise range
for p in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1; do python cool_mc.py --preprocessor "quantum_amplitude_damping:$p" --parent_run_id="last" --project_name="ski_experiments" --prism_dir="../prism_files" --prism_file_path="ski2.prism" --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" 2>&1 | tee -a noise_range4.log; done

# REINFORCE SKI
python cool_mc.py --rl_algorithm="reinforce" --num_episodes 10000 --project_name="ski_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="ski2.prism" --reward_flag=1 --seed=128 --lr=0.01 --neurons=128 --max_steps=50 --wrong_action_penalty=10 2>&1 | tee -a quantum_ski.log
python cool_mc.py --parent_run_id="last" --project_name="ski_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="ski2.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" 2>&1 | tee -a quantum_ski.log 2>&1 | tee -a quantum_ski.log


# Freeway
python cool_mc.py --rl_algorithm="qreinforce" --num_episodes 10000 --project_name="freeway_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="freeway.prism" --reward_flag=1 --seed=128 --lr=0.01 2>&1 | tee -a quantum_freeway.log
python cool_mc.py --parent_run_id="last" --project_name="freeway_experiments" --constant_definitions="" --prism_dir="../prism_files" --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" 2>&1 | tee -a quantum_freeway.log
#python cool_mc.py --parent_run_id="last" --preprocessor "quantum_bit_flip:0.01" --project_name="freeway_experiments" --constant_definitions="" --prism_dir="../prism_files" --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" 2>&1 | tee -a quantum_freeway.log
#python cool_mc.py --parent_run_id="last" --preprocessor "quantum_depolarizing_noise:0.01" --project_name="freeway_experiments" --constant_definitions="" --prism_dir="../prism_files" --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" 2>&1 | tee -a quantum_freeway.log
#python cool_mc.py --parent_run_id="last" --preprocessor "quantum_phase_flip:0.005" --project_name="freeway_experiments" --constant_definitions="" --prism_dir="../prism_files" --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" 2>&1 | tee -a quantum_freeway.log
#python cool_mc.py --parent_run_id="last" --preprocessor "quantum_depolarizing_noise:0.01#quantum_bit_flip:0.005#quantum_phase_flip:0.005" --project_name="freeway_experiments" --constant_definitions="" --prism_dir="../prism_files" --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" 2>&1 | tee -a quantum_freeway.log

# Freeway Bit-flip noise range
for p in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1; do python cool_mc.py --preprocessor "quantum_bit_flip:$p" --parent_run_id="last" --project_name="freeway_experiments" --prism_dir="../prism_files" --prism_file_path="freeway.prism" --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" 2>&1 | tee -a freeway_noise_range1.log; done

# Freeway Phase-flip noise range
for p in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1; do python cool_mc.py --preprocessor "quantum_phase_flip:$p" --parent_run_id="last" --project_name="freeway_experiments" --prism_dir="../prism_files" --prism_file_path="freeway.prism" --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" 2>&1 | tee -a freeway_noise_range2.log; done

# Freeway Depolarizing noise range
for p in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1; do python cool_mc.py --preprocessor "quantum_depolarizing_noise:$p" --parent_run_id="last" --project_name="freeway_experiments" --prism_dir="../prism_files" --prism_file_path="freeway.prism" --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" 2>&1 | tee -a freeway_noise_range3.log; done

# Freeway Amplitude damping noise range
for p in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1; do python cool_mc.py --preprocessor "quantum_amplitude_damping:$p" --parent_run_id="last" --project_name="freeway_experiments" --prism_dir="../prism_files" --prism_file_path="freeway.prism" --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" 2>&1 | tee -a freeway_noise_range4.log; done

python cool_mc.py --rl_algorithm="reinforce" --num_episodes 10000 --project_name="freeway_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="freeway.prism" --reward_flag=1 --seed=128 --lr=0.01 2>&1 | tee -a quantum_freeway.log
python cool_mc.py --parent_run_id="last" --project_name="freeway_experiments" --constant_definitions="" --prism_dir="../prism_files" --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" 2>&1 | tee -a quantum_freeway.log


