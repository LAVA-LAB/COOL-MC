#!/bin/bash


interpreter_name="random_unstructured_pruning_interpreter"

# Taxi First
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --interpreter="$interpreter_name;$percentage;all_pruning.txt;0"
    # Call your python script with the --config argument
done


python clean_projects.py

# jobs_done=1 First
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]" --interpreter="$interpreter_name;$percentage;all_pruning.txt;0"
    # Call your python script with the --config argument
done

python clean_projects.py



# Taxi empty First
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]" --interpreter="$interpreter_name;$percentage;all_pruning.txt;0"
    # Call your python script with the --config argument
done

python clean_projects.py



interpreter_name="l1_unstructured_pruning_interpreter"

# Taxi First
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --interpreter="$interpreter_name;$percentage;all_pruning.txt;0"
    # Call your python script with the --config argument
done


python clean_projects.py

# jobs_done=1 First
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]" --interpreter="$interpreter_name;$percentage;all_pruning.txt;0"
    # Call your python script with the --config argument
done

python clean_projects.py



# Taxi empty First
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]" --interpreter="$interpreter_name;$percentage;all_pruning.txt;0"
    # Call your python script with the --config argument
done

python clean_projects.py


