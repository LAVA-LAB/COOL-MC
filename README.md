# COOL-MC
COOL-MC provides a variety of environments for reinforcement learning (RL). It is an interface between Model Checking and Reinforcement Learning.
In particular, it extends the OpenAI Gym to support RL training on PRISM environments and allows verification of the trained RL policies via the Storm model checker.
The general workflow of our approach is as follows. First, we model the RL environment as a MDP in PRISM. Second, we train our RL policy in the PRISM environment or, if available, in the matching OpenAI Gym environment. Third, we verify the trained RL policy via the Storm model checker. Depending on the model checking outcome, we retrain the RL policy or deploy it.

![workflow](https://github.com/DennisGross/COOL-MC/blob/main/documentation/images/workflow_diagram.png)


##### Content
1. Getting Started with COOL-MC
2. Example 1 (Frozen Lake)
3. Example 2 (Taxi)
4. Example 3 (Collision Avoidance)
5. Example 4 (Smart Grid)
6. Web Interface
7. PRISM Modelling Tips
8. RL Agent Training
9. COOL-MC Command Line Arguments
10. Manual Installation


## Getting Started with COOL-MC
We assume that you have docker installed and that you run the following commands in the root of this repository:
1. Download the docker container [here](https://drive.google.com/file/d/10C3PkC6uU0M-FEY58zeVOER8CK9zUO3L/view?usp=sharing) (not up to date).
2. Load the docker container: `docker load --input coolmc.tar`
3. Create a project folder: `mkdir projects`
4. Run the docker container: `docker run --user mycoolmc  -v "$(pwd)/prism_files":"/home/mycoolmc/prism_files" -it coolmc bash`

Please make sure that you either run COOL-MC in on your machine OR in the docker container. Otherwise, it may lead to folder permission problems.

We discuss how to create the docker container yourself, and how to install the tool natively later.

If you are not familiar with PRISM/Storm, here are some references:

- [PRISM Manual](https://www.prismmodelchecker.org/manual/)
- [Storm Getting Started](https://www.stormchecker.org/getting-started.html)

## Example 1 (Frozen Lake)
FrozenLake is a commonly used OpenAI Gym benchmark, where 
the agent has to reach the goal (frisbee) on a frozen lake. The movement direction of the agent is uncertain and only depends in $33.33\%$ of the cases on the chosen direction. In $66.66\%$ of the cases, the movement is noisy.
To demonstrate our tool, we are going to train a RL policy for the OpenAI Gym Frozen Lake environment, verify it, and retrain it in the PRISM environment.
The following command trains the RL policy in the OpenAI Gym FrozenLake-v0 environment.

`python cool_mc.py --task=openai_training --project_name="Frozen Lake Example" --num_episodes=100 --eval_interval=10 --env=FrozenLake-v0 --sliding_window_size=100 --rl_algorithm=dqn_agent --layers=2 --neurons=64 --replay_buffer_size=30000 --epsilon=1 --epsilon_dec=0.9999 --epsilon_min=0.1 --gamma=0.99 --replace=304 --lr=0.001 --batch_size=32`

Project Specific Arguments:
- `task=openai_training` sets the current task to the OpenAI Gym training.
- `project_name=Frozen Lake Example` is the name of YOUR project. Labeling experiments with the same project name allows the comparison between the runs.
- `num_episodes=100` defines the number of training episodes.
- `eval_inteveral=10` defines the evaluation interval of the RL episode.
- `env=FrozenLake-v0` defines the OpenAI Gym.
- `sliding_window=100` is the size of the sliding window for the reward.

Reinforcement Learning Arguments:
- `rl_algorithm=dqn_agent` defines the reinforcement learning algorithm.
- `layers=2` defines the number of neural network layers.
- `neurons=64` defines the number of neurons in each layer.
- `replay_buffer=30000` defines the size of the replay buffer.
- `epsilon=1` defines the starting epsilon value.
- `epsilon_dec=0.9999` defines the epsilon decay (new_epsilon=current_epsilon * epsilon_dec).
- `epsilon_min=0.1` defines the minimal epsilon value.
- `gamma=0.99` defines the gamma value.
- `replace=304` defines the target network replacing interval.
- `lr=0.001` is the learning rate argument.

After the training, we receive an Experiment ID (e.g. 27f2bbe444754ac0bbbce1326a410419).
We need this ID to identify the experiment.
Now it is possible to verify the trained RL policy via the COOL-MC verifier by passing the experiment ID and the model checking arguments:

`python cool_mc.py --project_name "Frozen Lake Example" --parent_run_id=27f2bbe444754ac0bbbce1326a410419 --task rl_model_checking --prism_file_path="frozen_lake3-v1.prism" --constant_definitions="control=0.33,start_position=0" --prop="Pmin=? [F WATER=true]"`

Project Specific Arguments:
- `project_name "Frozen Lake Example"` specifies the project name.
- `parent_run_id=XXXX` reference to the trained RL policy (use experiment ID).
- `task=rl_model_checking` sets the current task to model checking the trained RL policy.

Model Checking Arguments:
- `prism_file_path="frozen_lake3-v1.prism"` specifies the PRISM environment.
- `constant_definitions="control=0.33,start_position=0"` sets the constant definitions for the PRISM environment.
- `prop="Pmin=? [F WATER=true]"` the property query.

It is also possible to plot the property results over a range of PRISM constant definitions. This is useful, when we want to get a overview of the trained RL policy. In the following command, we plot from different frozen lake agent startpositions 0-15 (stepsize 1, 16 excluded) the probability of falling into the water.

`python cool_mc.py --project_name "Frozen Lake Example" --parent_run_id=27f2bbe444754ac0bbbce1326a410419 --task rl_model_checking --prism_file_path="frozen_lake3-v1.prism" --constant_definitions="control=0.33,start_position=[0;1;16]" --prop="Pmin=? [F WATER=true]"`

`--constant_definitions="control=0.33,start_position=[0;1;16]"` calculates the property results from `start_position=0` up to `start_position=15` with a stepsize of `1`.


If we are not satisfied with the property result, we can retrain the RL policy via the OpenAI Gym or the PRISM environment. The following command, retrains the RL policy in the PRISM environment. 

`python cool_mc.py --task=safe_training  --parent_run_id=27f2bbe444754ac0bbbce1326a410419  --reward_flag=1 --project_name="Frozen Lake Example" --num_episodes=100 --eval_interval=10 --sliding_window_size=100 --rl_algorithm=dqn_agent --layers=2 --neurons=64 --replay_buffer_size=30000 --epsilon=1 --epsilon_dec=0.9999 --epsilon_min=0.1 --gamma=0.99 --replace=304 --lr=0.001 --batch_size=32 --prism_file_path="frozen_lake3-v1.prism" --constant_definitions="control=0.33,start_position=0" --prop="Pmin=? [F WATER=true]"`


- `task=safe_training` specifies the safe training task which allows the RL training in the PRISM environment.
- `prop="Pmin=? [F WATER=true]` tracks the agents probability of falling into the water. `Pmin` also specifies that COOL-MC saves only RL agents which lower probabilities of falling into the water.
- `reward_flag=1` uses rewards instead of penalties.

## Example 2 (Taxi with Fuel)
The taxi agent has to pick up passengers and transport them to their destination without running out of fuel. The environment terminates as soon as the taxi agent does the predefined number of jobs. After the job is done, a new guest spawns randomly at one of the predefined locations.

We train a DQN taxi agent in the PRISM environment:

`python cool_mc.py --task=safe_training --project_name="Taxi with Fuel Example" --num_episodes=100 --eval_interval=10 --sliding_window_size=100 --rl_algorithm=dqn_agent --layers=2 --neurons=64 --replay_buffer_size=30000 --epsilon=1 --epsilon_dec=0.9999 --epsilon_min=0.1 --gamma=0.99 --replace=304 --lr=0.001 --batch_size=32 --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [F jobs_done=2]"`


- `project_name="Taxi with Fuel Example"` for the new `Taxi with Fuel Example` project.
- `prop="Pmin=? [F jobs_done=2]"` property query for getting the probability to finish 2 jobs with the trained policy.

After the training, we can verify the trained policy:

`python cool_mc.py --parent_run_id=dd790c269b334e4383b580e7c1da9050 --task=rl_model_checking --project_name="Taxi with Fuel Example" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [F OUT_OF_FUEL=true]"`

State abstraction allows model-checking the trained policy on less precise features without changing the environment. To achieve this, a prepossessing step is applied to the current state in the incremental building process to map the state to a more abstract state for the RL policy. We only have to define a state mapping file and link it to COOL-MC via the command line:

`python cool_mc.py --parent_run_id=dd790c269b334e4383b580e7c1da9050 --task=rl_model_checking --project_name="Taxi with Fuel Example" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [F OUT_OF_FUEL=true]" --abstract_features="../taxi_abstraction.json"`

Permissive Model Checking allows the investigation of the worst-/best-case behaviour of the trained policy for certain state variables.

Minimal Probability of running out of fuel for a fuel level between 4 and 10 (Pmin):
`python cool_mc.py --parent_run_id=dd790c269b334e4383b580e7c1da9050 --task=rl_model_checking --project_name="Taxi with Fuel Example" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmin=? [F OUT_OF_FUEL=true]" --permissive_input="fuel<>=[4;10]"`

Maximal Probability of running out of fuel for a fuel level between 4 and 10 (Pmin):
`python cool_mc.py --parent_run_id=dd790c269b334e4383b580e7c1da9050 --task=rl_model_checking --project_name="Taxi with Fuel Example" --prism_file_path="transporter.prism" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prop="Pmax=? [F OUT_OF_FUEL=true]" --permissive_input="fuel<>=[4;10]"`


## Example 3 (Avoid)
Collision avoidance is an environment which contains one agent and two moving obstacles in a two dimensional grid world. The environment terminates as soon as a collision between the agent and one obstacle happens. The environment contains a slickness parameter, which defines the probability that the agent stays in the same cell.

We train a DQN taxi agent in the PRISM environment:

`python cool_mc.py --task=safe_training --project_name="Avoid Example" --num_episodes=100 --eval_interval=10 --sliding_window_size=100 --rl_algorithm=dqn_agent --layers=2 --neurons=64 --replay_buffer_size=30000 --epsilon=1 --epsilon_dec=0.9999 --epsilon_min=0.1 --gamma=0.99 --replace=304 --lr=0.001 --batch_size=32 --prism_file_path="avoid.prism" --constant_definitions="xMax=4,yMax=4,slickness=0.0" --prop="Pmin=? [F COLLISION=true]" --reward_flag=1`

After the training, we can verify the trained policy:

`python cool_mc.py --parent_run_id=915fd49f5f9342a5b5f124dddfd3f15f --task=rl_model_checking --project_name="Avoid Example" --prism_file_path="avoid.prism" --constant_definitions="xMax=4,yMax=4,slickness=0.0" --prop="Pmin=? [F COLLISION=true]"`


## Example 4 (Smart Grid)
In this environment, a controller controls the distribution of renewable- and non-renewable energy production. The objective is to minimize the production of non-renewable energy by using renewable and storage technologies.
If there is too much energy in the electricity network, the energy production shuts down which may lead to a blackout (terminal state).


`python cool_mc.py --task=safe_training --project_name="Smart Grid Example" --num_episodes=100 --eval_interval=10 --sliding_window_size=100 --rl_algorithm=dqn_agent --layers=2 --neurons=64 --replay_buffer_size=30000 --epsilon=1 --epsilon_dec=0.9999 --epsilon_min=0.1 --gamma=0.99 --replace=304 --lr=0.001 --batch_size=32 --prism_file_path="smart_grid.prism" --constant_definitions="max_consumption=20,renewable_limit=19,non_renewable_limit=16,grid_upper_bound=25" --prop="Tmin=? [F IS_BLACKOUT=true]" --reward_flag=0`

After the training, we can verify the trained policy:

`python cool_mc.py --parent_run_id=c0b0a71a334e4873b045858bc5be15ed --task=rl_model_checking --project_name="Smart Grid Example" --prism_file_path="smart_grid.prism" --constant_definitions="max_consumption=20,renewable_limit=19,non_renewable_limit=16,grid_upper_bound=25" --prop="Tmin=? [F TOO_MUCH_ENERGY=true]"`

## Web-Interface
`bash start_ui.sh` starts the MLFlow server to analyze the RL training process (http://hostname:5000) and a web interface (early alpha version) to control COOL-MC via a GUI (http://hostname:12345).


## PRISM Modelling Tips
We first have to model our RL environment. COOL-MC supports PRISM as modeling language. It can be difficult to design own PRISM environments. Here are some tips how to make sure that your PRISM environment works correctly with COOL-MC:

- Make sure that you only use transition-rewards
- After the agent reaches a terminal state, the storm simulator stops the simulation. Therefore, terminal state transitions will not executed. So, do not use self-looping terminal states.
- To improve the training performance, try to make all actions at every state available. Otherwise, the agent may chooses a not available action and receives a penalty.
- Try to unit test your PRISM environment before RL training. Does it behave as you want?

## RL Agent Training
After we modeled the environment, we can train RL agents on this environment.
It is also possible to develop your own RL agents:
1. Create a AGENT_NAME.py in the src.rl_agents package
2. Create a class AGENT_NAME and inherit all methods from src.agent.Agent
3. Set use_tf_environment to true if you use tf_environments instead of py_environments from tf_agents
4. Override all the needed methods (depends on your agent) + the agent save- and load-method.
5. In src.rl_agents.agent_builder extends the build_agent method with an additional elif branch for your agent
6. Add additional command-line arguments in cool_mc.py (if needed)

Here are some tips that may improve the training progress:

- Try to use the disable_state parameter to disable state variables from PRISM which are only relevant for the PRISM environment architecture.
- Play around with the RL parameters.
- Think especially about the size of the max_steps if you have only one terminal state and if this terminal state is a  negative outcome with a huge penalty. Too large max_steps values lead to always reaching this negative step in the beginning of the training. Instead, decrease the max_steps so that the RL agent may not reach this terminal state and stops training earlier. So the huge penality is not influencing the RL training too much in the beginning.
- The model checking part while RL training can take time. Therefore, the best way to train and verify your model is to first use reward_max. After the RL model may reaches an execptable reward the change the parameter prop_type to min_prop or max_prop and adjust the evaluation intervals.

## COOL-MC Command Line Arguments
The following list contains all the major COOL-MC command line arguments. It does not contain the arguments which are related to the RL algorithms. For a detailed description, we refer to the common.rl_agents package.

| Argument             | Description                                                                                                                                                                                                                                                                                 | Options                                           | Default Value  |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|----------------|
| task                 | The type of task do you want to perform.                                                                                                                                                                                                                                                    | safe_training, openai_training, rl_model_checking | safe_training  |
| project_name         | The name of your project.                                                                                                                                                                                                                                                                   |                                                   | defaultproject |
| parent_run_id        | Reference to previous experiment for retraining or verification.                                                                                                                                                                                                                            | PROJECT_IDs                                       |                |
| num_episodes         | The number of training episodes.                                                                                                                                                                                                                                                            | INTEGER NUMBER                                    | 1000           |
| eval_interval        | Interval for verification while safe_training.                                                                                                                                                                                                                                              | INTEGER NUMBER                                    | 100            |
| sliding_window_size  | Sliding window size for reward averaging over episodes.                                                                                                                                                                                                                                     | INTEGER NUMBER                                    | 100            |
| rl_algorithm         | The name of the RL algorithm.                                                                                                                                                                                                                                                               | dqn_agent, sarsamax                               | dqn_agent      |
| env                  | openai_training parameter for the environment name.                                                                                                                                                                                                                                         | OPENAI GYM NAMES                                  |                |
| prism_dir            | The directory of the PRISM files.                                                                                                                                                                                                                                                           | PATH                                              | ../prism_files |
| prism_file_path      | The name of the PRISM file.                                                                                                                                                                                                                                                                 | STR                                               |                |
| constant_definitions | Constant definitions seperated by a commata.                                                                                                                                                                                                                                                | For example: xMax=4,yMax=4,slickness=0            |                |
| prop                 | Property Query. **For safe_training:** Pmax tries to save RL policies that have higher probabilities. Pmin tries to save RL policies that have  lower probabilities. **For rl_model_checking:** In the case of induced DTMCs min/max  yield to the same property result (do not remove it). |                                                   |                |
| max_steps            | Maximal steps in the safe gym environment.                                                                                                                                                                                                                                                  |                                                   | 100            |
| disabled_features    | Disable features in the state space.                                                                                                                                                                                                                                                        | FEATURES SEPERATED BY A COMMATA                   |                |
| permissive_input     | It allows the investigation of the worst-/best-case behaviour of the trained policy for certain state variables.                                                                                                                                                                            |                                                   |                |
| abstract_features    | It allows model-checking the trained policy on less precise sensors without changing the environment.                                                                                                                                                                                       |                                                   |                |
| wrong_action_penalty | If an action is not available but still chosen by the policy, return a penalty of [DEFINED HERE].                                                                                                                                                                                           |                                                   |                |
| reward_flag          | If true (1), the agent receives rewards instead of penalties.                                                                                                                                                                                                                               |                                                   | 0              |
| range_plotting       | Range Plotting Flag for plotting the range plot on the screen.                                                                                                                                                                                                                              |                                                   |                |
| seed                 | Random seed for PyTorch, Numpy, Python.                                                                                                                                                                                                                                                     | INTEGER NUMBER                                    | None (-1)      |


### permissive_input
It allows the investigation of the worst-/best-case behaviour of the trained policy for certain state variables. Let's assume we have a formal model with state variables $a=[0,5]$, $b=[0,5]$, $c=[0,2]$.
We now want to investigate the permissive policies independent of $c$. Therefore, we generate at each state all the different RL policy actions for the different $c$ assignments and incrementally build the MDP.
- $c=[0,2]$ generates all the actions for the different c-assignments between $[0,2]$.
- $c=[0,1]$ generates all the actions for the different c-assignments between $[0,1]$.
- $c<>=[1,2]$ generates all the actions for the different c-assignments [1,2] only if the c value is between $[1,2]$. Otherwise, treat $c$ normally.

### abstract_features
It allows model-checking the trained policy on less precise sensors without changing the environment.
To achieve this, a prepossessing step is applied to the current state in the incremental building process to map the state to a more abstract state for the RL policy. We can archive the abstraction by:

- Passing the abstraction mapping file via the command line (e.g. taxi_abstraction.json)
- Passing the abstraction interval via command line (x=[0,2,10], maps the other x-values to the closest abstracted x-value).

This argument is reseted after rerunning the project.



## Manual Installation

### Creating the Docker

You can build the container via `docker build -t coolmc .` It is also possible for UNIX users to run the bash script in the bin-folder.

### Tool Installation
Switch to the repository folder and define environment variable `COOL_MC="$PWD"`

#### (1) Install Dependencies
`sudo apt-get update && sudo apt-get -y install build-essential git cmake libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev python3 python-is-python3 python3-setuptools python3-pip graphviz && sudo apt-get install -y --no-install-recommends maven uuid-dev virtualenv`

#### (2) Install Storm
0. `cd $COOL_MC`
1. `git clone https://github.com/moves-rwth/storm.git`
2. `cd storm`
3. `mkdir build`
4. `cd build`
5. `cmake ..`
6. `make -j 1`

For more information about building Storm, click [here](https://www.stormchecker.org/documentation/obtain-storm/build.html).

For testing the installation, follow the follow steps [here](https://www.stormchecker.org/documentation/obtain-storm/build.html#test-step-optional).

#### (3) Install PyCarl
0. `cd $COOL_MC`
1. `git clone https://github.com/moves-rwth/pycarl.git`
2. `cd pycarl`
3. `python setup.py build_ext --jobs 1 develop`

If permission problems: `sudo chmod 777 /usr/local/lib/python3.8/dist-packages/` and run third command again.


#### (4) Install Stormpy

0. `cd $COOL_MC`
1. `git clone https://github.com/moves-rwth/stormpy.git`
2. `cd stormpy`
3. `python setup.py build_ext --storm-dir "${COOL_MC}/build/" --jobs 1 develop`

For more information about the Stormpy installation, click [here](https://moves-rwth.github.io/stormpy/installation.html#installation-steps).

For testing the installation, follow the steps [here](https://moves-rwth.github.io/stormpy/installation.html#testing-stormpy-installation).

#### (5) Install remaining python packages and create project folder
0. `cd $COOL_MC`
1. `pip install -r requirements.txt`
2. `mkdir projects`