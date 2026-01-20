#python cool_mc.py --task=safe_training --project_name="djssp_experiments" --algorithm=cooperative_poagents --prism_file_path="scheduling_task.prism" --constant_definitions="" --prop="" --reward_flag=0 --seed=128 --epsilon=0.1 --epsilon_dec=0.9999 --layers=2 --neurons=128 --epsilon_min=0.01  --num_episodes=35000 --eval_interval=100
python cool_mc.py --parent_run_id="last" --task=rl_model_checking --project_name="navigation_experiments" --constant_definitions="" --prop="P=? [F \"jobs_done\" ]"
python cool_mc.py --parent_run_id="last" --task=rl_model_checking --project_name="navigation_experiments" --constant_definitions="" --prop="P=? [F \"collision\" ]"
python cool_mc.py --parent_run_id="last" --task=rl_model_checking --project_name="navigation_experiments" --constant_definitions="" --prop="P=? [F \"no_budget\" ]"
python cool_mc.py --parent_run_id="last" --task=rl_model_checking --project_name="navigation_experiments" --constant_definitions="" --prop="P=? [F \"time_over\" ]"
python cool_mc.py --parent_run_id="last" --task=rl_model_checking --project_name="navigation_experiments" --constant_definitions="" --prop="P=? [F \"wrong_order\" ]"
=======================
2023/01/21 08:49:26 INFO mlflow.projects.utils: === Created directory /tmp/tmpq64fpduj for downloading remote URIs passed to arguments of type 'path' ===
2023/01/21 08:49:26 INFO mlflow.projects.backend.local: === Running command 'python run.py --project_name=navigation_experiments --parent_run_id=last --preprocessor='' --prism_dir=../prism_files --prism_file_path=transporter.prism --constant_definitions='' --prop='P=? [F "jobs_done" ]' --task rl_model_checking --disabled_features '' --seed -1 --training_threshold -1000000000000 --num_episodes 1000 --eval_interval 9 --sliding_window_size 100 --reward_flag 0 --max_steps 100 --wrong_action_penalty 1000 --deploy 0 --postprocessor '' --range_plotting 1 --rl_algorithm dqn_agent --alpha 0.99 --noise_scale 0.01 --layers 2 --neurons 64 --replay_buffer_size 300000 --epsilon 1 --epsilon_dec 0.9999 --epsilon_min 0.1 --gamma 0.99 --replace 304 --lr 0.0001 --batch_size 32' in run with ID '2c0f7d45704f4cfeb3a66f7d4ccb9ceb' ===
Loaded last run b0014c7f56ef46779ee246218fac547c from project djssp_experiments
We use the following command line arguments:
{'task': 'rl_model_checking', 'project_name': 'djssp_experiments', 'parent_run_id': 'b0014c7f56ef46779ee246218fac547c', 'prism_dir': '../prism_files', 'prism_file_path': 'scheduling_task.prism', 'constant_definitions': '', 'disabled_features': '', 'seed': -1, 'training_threshold': -1000000000000, 'num_episodes': 1000, 'eval_interval': 9, 'sliding_window_size': 100, 'reward_flag': False, 'max_steps': 100, 'wrong_action_penalty': 1000, 'deploy': 0, 'preprocessor': '', 'postprocessor': '', 'prop': 'P=? [F "jobs_done" ]', 'range_plotting': 1, 'rl_algorithm': 'cooperative_poagents', 'alpha': 0.99, 'noise_scale': 0.01, 'layers': 2, 'neurons': 128, 'replay_buffer_size': 300000, 'epsilon': 1.0, 'epsilon_dec': 0.9999, 'epsilon_min': 0.1, 'gamma': 0.99, 'replace': 304, 'lr': 0.0001, 'batch_size': 32}
Build Cooperative Agents
Agent loaded.
LOAD ENVIRONMENT
Write to file test.drn.
P=? [F "jobs_done" ]:   0.6542160576249938
Model Size:             38391
Number of Transitions:  97018
Model Building Time:    6424.726408958435
Model Checking Time:    0.007097482681274414
Constant definitions:
Run ID: e5pom2fjs12sxifc85ui0srqvzt1z7or
2023/01/21 10:36:33 INFO mlflow.projects: === Run (ID '2c0f7d45704f4cfeb3a66f7d4ccb9ceb') succeeded ===
=======================
2023/01/21 10:36:34 INFO mlflow.projects.utils: === Created directory /tmp/tmpsme29ep5 for downloading remote URIs passed to arguments of type 'path' ===
2023/01/21 10:36:34 INFO mlflow.projects.backend.local: === Running command 'python run.py --project_name=navigation_experiments --parent_run_id=last --preprocessor='' --prism_dir=../prism_files --prism_file_path=transporter.prism --constant_definitions='' --prop='P=? [F "collision" ]' --task rl_model_checking --disabled_features '' --seed -1 --training_threshold -1000000000000 --num_episodes 1000 --eval_interval 9 --sliding_window_size 100 --reward_flag 0 --max_steps 100 --wrong_action_penalty 1000 --deploy 0 --postprocessor '' --range_plotting 1 --rl_algorithm dqn_agent --alpha 0.99 --noise_scale 0.01 --layers 2 --neurons 64 --replay_buffer_size 300000 --epsilon 1 --epsilon_dec 0.9999 --epsilon_min 0.1 --gamma 0.99 --replace 304 --lr 0.0001 --batch_size 32' in run with ID '90ddceaf328a41c893cab69083acbb8f' ===
Loaded last run b0014c7f56ef46779ee246218fac547c from project djssp_experiments
We use the following command line arguments:
{'task': 'rl_model_checking', 'project_name': 'djssp_experiments', 'parent_run_id': 'b0014c7f56ef46779ee246218fac547c', 'prism_dir': '../prism_files', 'prism_file_path': 'scheduling_task.prism', 'constant_definitions': '', 'disabled_features': '', 'seed': -1, 'training_threshold': -1000000000000, 'num_episodes': 1000, 'eval_interval': 9, 'sliding_window_size': 100, 'reward_flag': False, 'max_steps': 100, 'wrong_action_penalty': 1000, 'deploy': 0, 'preprocessor': '', 'postprocessor': '', 'prop': 'P=? [F "collision" ]', 'range_plotting': 1, 'rl_algorithm': 'cooperative_poagents', 'alpha': 0.99, 'noise_scale': 0.01, 'layers': 2, 'neurons': 128, 'replay_buffer_size': 300000, 'epsilon': 1.0, 'epsilon_dec': 0.9999, 'epsilon_min': 0.1, 'gamma': 0.99, 'replace': 304, 'lr': 0.0001, 'batch_size': 32}
Build Cooperative Agents
Agent loaded.
LOAD ENVIRONMENT
Write to file test.drn.
P=? [F "collision" ]:   0.1467036082544459
Model Size:             37041
Number of Transitions:  95668
Model Building Time:    6009.583067893982
Model Checking Time:    0.00981593132019043
Constant definitions:
Run ID: zgj2ux3xppixb7eh6uc5qcml1l5zs2je
2023/01/21 12:16:45 INFO mlflow.projects: === Run (ID '90ddceaf328a41c893cab69083acbb8f') succeeded ===
=======================
2023/01/21 12:16:46 INFO mlflow.projects.utils: === Created directory /tmp/tmpauywx0_z for downloading remote URIs passed to arguments of type 'path' ===
2023/01/21 12:16:46 INFO mlflow.projects.backend.local: === Running command 'python run.py --project_name=navigation_experiments --parent_run_id=last --preprocessor='' --prism_dir=../prism_files --prism_file_path=transporter.prism --constant_definitions='' --prop='P=? [F "no_budget" ]' --task rl_model_checking --disabled_features '' --seed -1 --training_threshold -1000000000000 --num_episodes 1000 --eval_interval 9 --sliding_window_size 100 --reward_flag 0 --max_steps 100 --wrong_action_penalty 1000 --deploy 0 --postprocessor '' --range_plotting 1 --rl_algorithm dqn_agent --alpha 0.99 --noise_scale 0.01 --layers 2 --neurons 64 --replay_buffer_size 300000 --epsilon 1 --epsilon_dec 0.9999 --epsilon_min 0.1 --gamma 0.99 --replace 304 --lr 0.0001 --batch_size 32' in run with ID '77441be8520545e083c33d4e7a523585' ===
Loaded last run b0014c7f56ef46779ee246218fac547c from project djssp_experiments
We use the following command line arguments:
{'task': 'rl_model_checking', 'project_name': 'djssp_experiments', 'parent_run_id': 'b0014c7f56ef46779ee246218fac547c', 'prism_dir': '../prism_files', 'prism_file_path': 'scheduling_task.prism', 'constant_definitions': '', 'disabled_features': '', 'seed': -1, 'training_threshold': -1000000000000, 'num_episodes': 1000, 'eval_interval': 9, 'sliding_window_size': 100, 'reward_flag': False, 'max_steps': 100, 'wrong_action_penalty': 1000, 'deploy': 0, 'preprocessor': '', 'postprocessor': '', 'prop': 'P=? [F "no_budget" ]', 'range_plotting': 1, 'rl_algorithm': 'cooperative_poagents', 'alpha': 0.99, 'noise_scale': 0.01, 'layers': 2, 'neurons': 128, 'replay_buffer_size': 300000, 'epsilon': 1.0, 'epsilon_dec': 0.9999, 'epsilon_min': 0.1, 'gamma': 0.99, 'replace': 304, 'lr': 0.0001, 'batch_size': 32}
Build Cooperative Agents
Agent loaded.
LOAD ENVIRONMENT
Write to file test.drn.
P=? [F "no_budget" ]:   0.008961204643464133
Model Size:             31515
Number of Transitions:  90142
Model Building Time:    4356.604407548904
Model Checking Time:    0.007178306579589844
Constant definitions:
Run ID: titcvl30qr75eg1sondvnwx3vo72drat
2023/01/21 13:29:25 INFO mlflow.projects: === Run (ID '77441be8520545e083c33d4e7a523585') succeeded ===
=======================
2023/01/21 13:29:26 INFO mlflow.projects.utils: === Created directory /tmp/tmprbgvurvh for downloading remote URIs passed to arguments of type 'path' ===
2023/01/21 13:29:26 INFO mlflow.projects.backend.local: === Running command 'python run.py --project_name=navigation_experiments --parent_run_id=last --preprocessor='' --prism_dir=../prism_files --prism_file_path=transporter.prism --constant_definitions='' --prop='P=? [F "time_over" ]' --task rl_model_checking --disabled_features '' --seed -1 --training_threshold -1000000000000 --num_episodes 1000 --eval_interval 9 --sliding_window_size 100 --reward_flag 0 --max_steps 100 --wrong_action_penalty 1000 --deploy 0 --postprocessor '' --range_plotting 1 --rl_algorithm dqn_agent --alpha 0.99 --noise_scale 0.01 --layers 2 --neurons 64 --replay_buffer_size 300000 --epsilon 1 --epsilon_dec 0.9999 --epsilon_min 0.1 --gamma 0.99 --replace 304 --lr 0.0001 --batch_size 32' in run with ID 'a87522609cb64afb9e49b60113313422' ===
Loaded last run b0014c7f56ef46779ee246218fac547c from project djssp_experiments
We use the following command line arguments:
{'task': 'rl_model_checking', 'project_name': 'djssp_experiments', 'parent_run_id': 'b0014c7f56ef46779ee246218fac547c', 'prism_dir': '../prism_files', 'prism_file_path': 'scheduling_task.prism', 'constant_definitions': '', 'disabled_features': '', 'seed': -1, 'training_threshold': -1000000000000, 'num_episodes': 1000, 'eval_interval': 9, 'sliding_window_size': 100, 'reward_flag': False, 'max_steps': 100, 'wrong_action_penalty': 1000, 'deploy': 0, 'preprocessor': '', 'postprocessor': '', 'prop': 'P=? [F "time_over" ]', 'range_plotting': 1, 'rl_algorithm': 'cooperative_poagents', 'alpha': 0.99, 'noise_scale': 0.01, 'layers': 2, 'neurons': 128, 'replay_buffer_size': 300000, 'epsilon': 1.0, 'epsilon_dec': 0.9999, 'epsilon_min': 0.1, 'gamma': 0.99, 'replace': 304, 'lr': 0.0001, 'batch_size': 32}
Build Cooperative Agents
Agent loaded.
LOAD ENVIRONMENT
Write to file test.drn.
P=? [F "time_over" ]:   2.3384774817948415e-19
Model Size:             37387
Number of Transitions:  96014
Model Building Time:    6099.599311351776
Model Checking Time:    0.006113529205322266
Constant definitions:
Run ID: mpeassib02e4owrdwgmkcinkcbcqeizw
2023/01/21 15:11:08 INFO mlflow.projects: === Run (ID 'a87522609cb64afb9e49b60113313422') succeeded ===
=======================
2023/01/21 15:11:09 INFO mlflow.projects.utils: === Created directory /tmp/tmpdivnckqd for downloading remote URIs passed to arguments of type 'path' ===
2023/01/21 15:11:09 INFO mlflow.projects.backend.local: === Running command 'python run.py --project_name=navigation_experiments --parent_run_id=last --preprocessor='' --prism_dir=../prism_files --prism_file_path=transporter.prism --constant_definitions='' --prop='P=? [F "wrong_order" ]' --task rl_model_checking --disabled_features '' --seed -1 --training_threshold -1000000000000 --num_episodes 1000 --eval_interval 9 --sliding_window_size 100 --reward_flag 0 --max_steps 100 --wrong_action_penalty 1000 --deploy 0 --postprocessor '' --range_plotting 1 --rl_algorithm dqn_agent --alpha 0.99 --noise_scale 0.01 --layers 2 --neurons 64 --replay_buffer_size 300000 --epsilon 1 --epsilon_dec 0.9999 --epsilon_min 0.1 --gamma 0.99 --replace 304 --lr 0.0001 --batch_size 32' in run with ID '7514eaee83f74d53bb967f23a2479193' ===
Loaded last run b0014c7f56ef46779ee246218fac547c from project djssp_experiments
We use the following command line arguments:
{'task': 'rl_model_checking', 'project_name': 'djssp_experiments', 'parent_run_id': 'b0014c7f56ef46779ee246218fac547c', 'prism_dir': '../prism_files', 'prism_file_path': 'scheduling_task.prism', 'constant_definitions': '', 'disabled_features': '', 'seed': -1, 'training_threshold': -1000000000000, 'num_episodes': 1000, 'eval_interval': 9, 'sliding_window_size': 100, 'reward_flag': False, 'max_steps': 100, 'wrong_action_penalty': 1000, 'deploy': 0, 'preprocessor': '', 'postprocessor': '', 'prop': 'P=? [F "wrong_order" ]', 'range_plotting': 1, 'rl_algorithm': 'cooperative_poagents', 'alpha': 0.99, 'noise_scale': 0.01, 'layers': 2, 'neurons': 128, 'replay_buffer_size': 300000, 'epsilon': 1.0, 'epsilon_dec': 0.9999, 'epsilon_min': 0.1, 'gamma': 0.99, 'replace': 304, 'lr': 0.0001, 'batch_size': 32}
Build Cooperative Agents
Agent loaded.
LOAD ENVIRONMENT
Write to file test.drn.
P=? [F "wrong_order" ]: 0.2529586316165145
Model Size:             35216
Number of Transitions:  93843
Model Building Time:    5414.832895040512
Model Checking Time:    0.0073549747467041016
Constant definitions:
Run ID: 6bl8ipv6l1z9y2rzs95lbrhaammo5g81
2023/01/21 16:41:26 INFO mlflow.projects: === Run (ID '7514eaee83f74d53bb967f23a2479193') succeeded ===
