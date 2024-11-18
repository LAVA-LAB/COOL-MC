#python cool_mc.py --parent_run_id="54cfe828d1664d9890db735f2360ab6c" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="Pmax=? [ F jobs_done=2 ]" --interpreter="co_graph_critical_states;Pmax=? [ F jobs_done=2 ];100"
#[('layer_0_neuron_4', 0.0004790902254262156), ('layer_0_neuron_3', 0.00047274100332547883), ('layer_0_neuron_5', 0.0004339636895964483), ('layer_0_neuron_2', 0.0004072450211094501), ('layer_0_neuron_7', 0.00038946561241740257), ('layer_0_neuron_0', 0.0003650967313170528), ('layer_0_neuron_1', 0.0003440493107469883), ('layer_0_neuron_6', 0.00034404931074698817)]
#[('layer_0_neuron_5', 0.0004825830589700619), ('layer_0_neuron_4', 0.0004621464849236933), ('layer_0_neuron_6', 0.00045638116423694854), ('layer_0_neuron_0', 0.0004168556372033636), ('layer_0_neuron_8', 0.0003956674414534294), ('layer_0_neuron_3', 0.00038155172088868267), ('layer_0_neuron_1', 0.0003305377426832467), ('layer_0_neuron_2', 0.00032858173015921303), ('layer_0_neuron_7', 0.00028272234688929095)]
#['done', 'fuel', 'jobs_done', 'passenger', 'passenger_dest_x', 'passenger_dest_y', 'passenger_loc_x', 'passenger_loc_y', 'x', 'y']
#python cool_mc.py --parent_run_id="54cfe828d1664d9890db735f2360ab6c" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F x=gas_x & y=gas_y]" --interpreter="co_activation_graph_analysis;Pmax=? [ F jobs_done=1 ];Pmax=? [ F jobs_done=2 ]"
#[('layer_0_neuron_1', 0.0005514800562342107), ('layer_0_neuron_3', 0.0004805523350548058), ('layer_0_neuron_0', 0.0004692043018519154), ('layer_0_neuron_2', 0.00041260639053082387)]
#[('layer_0_neuron_5', 0.0004799147476356023), ('layer_0_neuron_0', 0.0004668586897456996), ('layer_0_neuron_4', 0.0004647435602577199), ('layer_0_neuron_6', 0.0004518666619550499), ('layer_0_neuron_8', 0.0004172957810706614), ('layer_0_neuron_3', 0.0003840376273309715), ('layer_0_neuron_2', 0.0003738061513714168), ('layer_0_neuron_1', 0.0003163217718489396), ('layer_0_neuron_7', 0.00029489946850173225)]
#['done', 'fuel', 'jobs_done', 'passenger', 'passenger_dest_x', 'passenger_dest_y', 'passenger_loc_x', 'passenger_loc_y', 'x', 'y']


# Feature pruning
#python cool_mc.py --parent_run_id="54cfe828d1664d9890db735f2360ab6c" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --interpreter="feature_pruner;1;passenger_dest_y,passenger_dest_x,jobs_done"
#python cool_mc.py --parent_run_id="54cfe828d1664d9890db735f2360ab6c" --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F jobs_done=2 ]" --interpreter="feature_pruner;1;fuel,jobs_done,passenger_dest_x"


# Cleaning agents
python cool_mc.py --parent_run_id="a3fc7ab05b98476ca3f4c9a8ab592db9" --project_name="cleaning_system_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="cleaning_system.prism" --seed=128 --task="rl_model_checking" --prop="Pmax=? [ F dirt1=-2]"

#python cool_mc.py --parent_run_id="a3fc7ab05b98476ca3f4c9a8ab592db9" --project_name="cleaning_system_experiments" --constant_definitions="" --prism_dir="../prism_files" --prism_file_path="cleaning_system.prism" --seed=128 --task="rl_model_checking" --prop="P=? [ F dirt1=-2]" --interpreter="co_activation_graph_analysis;Pmax=? [ F dirt1=-2];Pmax=? [ F energy=0 ]"
