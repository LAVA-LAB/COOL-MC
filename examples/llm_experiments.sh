#mlflow run safe_training/ --env-manager=local -P num_episodes=102
#python cool_mc.py --num_episodes 102 --project_name="taxi_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128
#python cool_mc.py --rl_algorithm=llm_agent-gemma3:4b --project_name="taxi_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]"
#python cool_mc.py --rl_algorithm=llm_agent-deepseek-r1:1.5b --project_name="taxi_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter2.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]"
#python cool_mc.py --rl_algorithm=llm_agent-gemma3n:e2b --project_name="taxi_examples" --constant_definitions="start_position=0,control=0.333" --prism_dir="../prism_files" --prism_file_path="frozen_lake.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F AT_FRISBEE=true ]"
#python cool_mc.py --rl_algorithm=llm_agent-gemma3:1b --project_name="taxi_examples" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="stock_market.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"bankruptcy\" ]"

#python cool_mc.py --parent_run_id="last" --num_episodes 102 --project_name="taxi_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128
#python cool_mc.py --parent_run_id="last" --num_episodes 102 --project_name="taxi_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128

#"deepseek-r1:1.5b"
#"llama3.2:3b"
#"gemma3n:e4b"
#"gemma3:4b"
#"mistral:v0.3", # 7B
#"llama3.1:8b"

#"deepseek-r1:1.5b"
#python cool_mc.py --rl_algorithm=llm_agent-deepseek-r1:1.5b --project_name="llm_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]"
#python cool_mc.py --rl_algorithm=llm_agent-deepseek-r1:1.5b --project_name="llm_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter2.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]"
#python cool_mc.py --rl_algorithm=llm_agent-deepseek-r1:1.5b --project_name="llm_examples" --constant_definitions="start_position=0,control=0.333" --prism_dir="../prism_files" --prism_file_path="frozen_lake.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"in_water\" ]"
python cool_mc.py --rl_algorithm=llm_agent-deepseek-r1:1.5b --project_name="llm_examples" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="stock_market.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"bankruptcy\" ]"

#"llama3.2:3b"
#python cool_mc.py --rl_algorithm=llm_agent-llama3.2:3b --project_name="llm_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]"
#python cool_mc.py --rl_algorithm=llm_agent-llama3.2:3b --project_name="llm_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter2.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]"
#python cool_mc.py --rl_algorithm=llm_agent-llama3.2:3b --project_name="llm_examples" --constant_definitions="start_position=0,control=0.333" --prism_dir="../prism_files" --prism_file_path="frozen_lake.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"in_water\" ]"
#python cool_mc.py --rl_algorithm=llm_agent-llama3.2:3b --project_name="llm_examples" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="stock_market.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"bankruptcy\" ]"

#"gemma3n:e4b"
#python cool_mc.py --rl_algorithm=llm_agent-gemma3n:e4b --project_name="llm_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]"
#python cool_mc.py --rl_algorithm=llm_agent-gemma3n:e4b --project_name="llm_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter2.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]"
#python cool_mc.py --rl_algorithm=llm_agent-gemma3n:e4b --project_name="llm_examples" --constant_definitions="start_position=0,control=0.333" --prism_dir="../prism_files" --prism_file_path="frozen_lake.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"in_water\" ]"
#python cool_mc.py --rl_algorithm=llm_agent-gemma3n:e4b --project_name="llm_examples" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="stock_market.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"bankruptcy\" ]"

#"gemma3:4b"
#python cool_mc.py --rl_algorithm=llm_agent-gemma3:4b --project_name="llm_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]"
#python cool_mc.py --rl_algorithm=llm_agent-gemma3:4b --project_name="llm_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter2.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]"
#python cool_mc.py --rl_algorithm=llm_agent-gemma3:4b --project_name="llm_examples" --constant_definitions="start_position=0,control=0.333" --prism_dir="../prism_files" --prism_file_path="frozen_lake.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"in_water\" ]"
#python cool_mc.py --rl_algorithm=llm_agent-gemma3:4b --project_name="llm_examples" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="stock_market.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"bankruptcy\" ]"

#"mistral:v0.3", # 7B
#python cool_mc.py --rl_algorithm=llm_agent-mistral:v0.3 --project_name="llm_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]"
#python cool_mc.py --rl_algorithm=llm_agent-mistral:v0.3 --project_name="llm_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter2.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]"
#python cool_mc.py --rl_algorithm=llm_agent-mistral:v0.3 --project_name="llm_examples" --constant_definitions="start_position=0,control=0.333" --prism_dir="../prism_files" --prism_file_path="frozen_lake.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"in_water\" ]"
#python cool_mc.py --rl_algorithm=llm_agent-mistral:v0.3 --project_name="llm_examples" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="stock_market.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"bankruptcy\" ]"

#"deepseek-r1:7b"
python cool_mc.py --rl_algorithm=llm_agent-deepseek-r1:7b --project_name="llm_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]"
#python cool_mc.py --rl_algorithm=llm_agent-deepseek-r1:7b --project_name="llm_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter2.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]"
python cool_mc.py --rl_algorithm=llm_agent-deepseek-r1:7b --project_name="llm_examples" --constant_definitions="start_position=0,control=0.333" --prism_dir="../prism_files" --prism_file_path="frozen_lake.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"in_water\" ]"
python cool_mc.py --rl_algorithm=llm_agent-deepseek-r1:7b --project_name="llm_examples" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="stock_market.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"bankruptcy\" ]"

#"llama3.1:8b"
#python cool_mc.py --rl_algorithm=llm_agent-llama3.1:8b --project_name="llm_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]"
#python cool_mc.py --rl_algorithm=llm_agent-llama3.1:8b --project_name="llm_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter2.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]"
#python cool_mc.py --rl_algorithm=llm_agent-llama3.1:8b --project_name="llm_examples" --constant_definitions="start_position=0,control=0.333" --prism_dir="../prism_files" --prism_file_path="frozen_lake.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"in_water\" ]"
#python cool_mc.py --rl_algorithm=llm_agent-llama3.1:8b --project_name="llm_examples" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="stock_market.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"bankruptcy\" ]"

#"deepseek-r1:8b"
#python cool_mc.py --rl_algorithm=llm_agent-deepseek-r1:8b --project_name="llm_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"empty\" ]"
#python cool_mc.py --rl_algorithm=llm_agent-deepseek-r1:8b --project_name="llm_examples" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter2.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=1 ]"
#python cool_mc.py --rl_algorithm=llm_agent-deepseek-r1:8b --project_name="llm_examples" --constant_definitions="start_position=0,control=0.333" --prism_dir="../prism_files" --prism_file_path="frozen_lake.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"in_water\" ]"
#python cool_mc.py --rl_algorithm=llm_agent-deepseek-r1:8b --project_name="llm_examples" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="stock_market.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F \"bankruptcy\" ]"