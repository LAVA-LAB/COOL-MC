################## Freeway
#RUN_ID="a1a75bb68d894aa0aa6832f660900d04"
#python cool_mc.py --parent_run_id=$RUN_ID --project_name="freeway_experiments" --constant_definitions="" --prism_dir="../prism_files" --task="rl_model_checking" --prop="P=? [ F \"goal\" ]" --preprocessor="old"
#python cool_mc.py --parent_run_id=$RUN_ID --project_name="freeway_experiments" --constant_definitions="" --prism_dir="../prism_files" --task="rl_model_checking" --prop="P=? [ F \"goal\" ]"
#storm --prism "prism_files/freeway.prism" --prop "Pmax=? [F \"goal\"]"
#python cool_mc.py --parent_run_id=$RUN_ID --project_name="freeway_experiments" --constant_definitions="" --prism_dir="../prism_files" --task="rl_model_checking" --prop="P=? [ ((true U px24=0) U px17=0) ]" --preprocessor="old"
#python cool_mc.py --parent_run_id=$RUN_ID --project_name="freeway_experiments" --constant_definitions="" --prism_dir="../prism_files" --task="rl_model_checking" --prop="P=? [ ((true U px24=0) U px17=0) ]"
#storm --prism "prism_files/freeway.prism" --prop "Pmax=? [ ((true U px24=0) U px17=0) ]"

################## Crazy Climber
#RUN_ID="3aeb73d69125471b83d81ad6088aef3b"
#python cool_mc.py --parent_run_id=$RUN_ID --project_name="crazy_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="crazy_climber.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F COLLISION=true ]" --preprocessor="old"
#python cool_mc.py --parent_run_id=$RUN_ID --project_name="crazy_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="crazy_climber.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F COLLISION=true ]"
#storm --prism "prism_files/crazy_climber.prism" --prop "Pmax=? [ F COLLISION=true ]"

################## Avoidance NORMAL
RUN_ID="f8b075e8e8c04ead8af5a9f62614c01b"
#python cool_mc.py --parent_run_id=$RUN_ID --project_name="avoid_experiments" --constant_definitions="xMax=4,yMax=4,slickness=0.1" --prism_dir="../prism_files" --seed=128 --prism_file_path="avoid.prism" --task="rl_model_checking" --prop="P=? [ F<=100 COLLISION=true ]" --preprocessor="old"
#python cool_mc.py --parent_run_id=$RUN_ID --project_name="avoid_experiments" --constant_definitions="xMax=4,yMax=4,slickness=0.1" --prism_dir="../prism_files" --seed=128 --prism_file_path="avoid.prism" --task="rl_model_checking" --prop="P=? [ F<=100 COLLISION=true ]"
#storm --prism "prism_files/avoid.prism" --prop "Pmax=? [ F<=100 COLLISION=true ]" --constants "xMax=4,yMax=4,slickness=0.1"

################## Avoidance Out of memory for naive and stochastic
#python cool_mc.py --parent_run_id=$RUN_ID --project_name="avoid_experiments" --constant_definitions="xMax=12,yMax=12,slickness=0.1" --prism_dir="../prism_files" --seed=128 --prism_file_path="avoid.prism" --task="rl_model_checking" --prop="P=? [ F<=100 COLLISION=true ]" --preprocessor="old"
python cool_mc.py --parent_run_id=$RUN_ID --project_name="avoid_experiments" --constant_definitions="xMax=5,yMax=5,slickness=0.1" --prism_dir="../prism_files" --seed=128 --prism_file_path="avoid.prism" --task="rl_model_checking" --prop="P=? [ F<=100 COLLISION=true ]" --preprocessor="old"
#storm --prism "prism_files/avoid.prism" --prop "Pmax=? [ F<=100 COLLISION=true ]" --constants "xMax=10,yMax=10,slickness=0.1"

#storm --prism "prism_files/avoid.prism" --prop "Pmax=? [ F<=100 COLLISION=true ]"
#storm --prism "prism_files/avoid.prism" --prop "Pmax=? [ F<=100 COLLISION=true ]" --constants "xMax=12,yMax=12,slickness=0.1"



#P=? [ F<=100 COLLISION=true ]:  0.6481465461228078
#Model Size:             46656
#Number of Transitions:  2891208
#Model Building Time:    4688.7135355472565
#Model Checking Time:    3.1103076934814453
#Constant definitions:   xMax=5,yMax=5,slickness=0.1
#Run ID: 184foo6su9c4cwe6m1gh0j82qfzur3bx69utjtlgtg