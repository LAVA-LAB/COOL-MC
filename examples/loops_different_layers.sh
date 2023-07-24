#!/bin/bash

interpreter_name="l1_unstructured_pruning_interpreter"

# Taxi First
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --interpreter="$interpreter_name;$percentage;different_layers_pruning.txt;-5"
    # Call your python script with the --config argument
done

python clean_projects.py

# Taxi Second
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --interpreter="$interpreter_name;$percentage;different_layers_pruning.txt;-4"
    # Call your python script with the --config argument
done

python clean_projects.py

# Taxi Third
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --interpreter="$interpreter_name;$percentage;different_layers_pruning.txt;-3"
    # Call your python script with the --config argument
done

python clean_projects.py

# Taxi Last
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --interpreter="$interpreter_name;$percentage;different_layers_pruning.txt;-2"
    # Call your python script with the --config argument
done

python clean_projects.py

# Taxi Last
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --interpreter="$interpreter_name;$percentage;different_layers_pruning.txt;-1"
    # Call your python script with the --config argument
done


### jobs=1
# Taxi First
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]" --interpreter="$interpreter_name;$percentage;different_layers_pruning.txt;-5"
    # Call your python script with the --config argument
done

python clean_projects.py

# Taxi Second
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]" --interpreter="$interpreter_name;$percentage;different_layers_pruning.txt;-4"
    # Call your python script with the --config argument
done

python clean_projects.py

# Taxi Third
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]" --interpreter="$interpreter_name;$percentage;different_layers_pruning.txt;-3"
    # Call your python script with the --config argument
done

python clean_projects.py

# Taxi Last
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]" --interpreter="$interpreter_name;$percentage;different_layers_pruning.txt;-2"
    # Call your python script with the --config argument
done

python clean_projects.py

# Taxi Last
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]" --interpreter="$interpreter_name;$percentage;different_layers_pruning.txt;-1"
    # Call your python script with the --config argument
done


### Empty

# Taxi First
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]" --interpreter="$interpreter_name;$percentage;different_layers_pruning.txt;-5"
    # Call your python script with the --config argument
done

python clean_projects.py

# Taxi Second
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]" --interpreter="$interpreter_name;$percentage;different_layers_pruning.txt;-4"
    # Call your python script with the --config argument
done

python clean_projects.py

# Taxi Third
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]" --interpreter="$interpreter_name;$percentage;different_layers_pruning.txt;-3"
    # Call your python script with the --config argument
done

python clean_projects.py

# Taxi Last
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]" --interpreter="$interpreter_name;$percentage;different_layers_pruning.txt;-2"
    # Call your python script with the --config argument
done

python clean_projects.py

# Taxi Last
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]" --interpreter="$interpreter_name;$percentage;different_layers_pruning.txt;-1"
    # Call your python script with the --config argument
done


# passenger -> gas
# Taxi First
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ ((true U (x=red_x & y=red_y)) U passenger=true)  U (gas_x=x & gas_y=y) ]" --interpreter="$interpreter_name;$percentage;different_layers_pruning.txt;-5"
    # Call your python script with the --config argument
done

python clean_projects.py

# Taxi Second
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ ((true U (x=red_x & y=red_y)) U passenger=true)  U (gas_x=x & gas_y=y) ]" --interpreter="$interpreter_name;$percentage;different_layers_pruning.txt;-4"
    # Call your python script with the --config argument
done

python clean_projects.py

# Taxi Third
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ ((true U (x=red_x & y=red_y)) U passenger=true)  U (gas_x=x & gas_y=y) ]" --interpreter="$interpreter_name;$percentage;different_layers_pruning.txt;-3"
    # Call your python script with the --config argument
done

python clean_projects.py

# Taxi Last
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ ((true U (x=red_x & y=red_y)) U passenger=true)  U (gas_x=x & gas_y=y) ]" --interpreter="$interpreter_name;$percentage;different_layers_pruning.txt;-2"
    # Call your python script with the --config argument
done

python clean_projects.py

# Taxi Last
for i in $(seq 0 1 100); do
    # Calculate percentage (dividing by 10 gives us the decimal value)
    percentage=$(awk -v i="$i" 'BEGIN{printf "%.2f\n", i/100}')
    echo "$percentage"
    python cool_mc.py --parent_run_id="64a784ae6aaa49d6b35d31654af77603" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ ((true U (x=red_x & y=red_y)) U passenger=true)  U (gas_x=x & gas_y=y) ]" --interpreter="$interpreter_name;$percentage;different_layers_pruning.txt;-1"
    # Call your python script with the --config argument
done
