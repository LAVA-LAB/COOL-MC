# COOL-MC
COOL-MC is a tool that combines state-of-the-art single-agent and multi-agent reinforcement learning with model checking.
It builds upon the OpenAI gym and the probabilistic model checker Storm.
COOL-MC includes a simulator for training policies in Markov decision processes (MDPs) using the OpenAI gym, a model builder from Storm that allows the verification of these policies using callback functions, and algorithms for obtaining performance bounds on permissive policies.
It also measures the impact of adversarial attacks on policies and temporal logic properties, verifies the robustness of policies against adversarial attacks, and verifies countermeasures against these attacks.

In the following diagram, we see the general workflow of COOL-MC.
First, an agent is trained in an environment concerning a objective.
Second, the policy is verified for further safety specifications.
Then, the user can decide to further train the policy or to deploy the agent.
Note that retraining does not guarantee the safety specification if the
learning objective and the safety specification are mutually disjoint.

![components](https://github.com/LAVA-LAB/COOL-MC/blob/main/images/workflow.png)

The verification process take the environment (modeled as an MDP via PRISM), the learned RL agent's policy, and optionally an adversarial attack configuration.
The verifier outputs than the verification result.

![components](https://github.com/LAVA-LAB/COOL-MC/blob/main/images/verifier.png)

## Architecture
We will first describe the main components of COOL-MC and then delve into the details.

### Training
The following code runs an agent in an environment. The env variable represents the environment, and the reset method is called to reset the environment to its initial state. The step method is used to take an action in the environment, and observe the resulting next state, reward, and whether the episode has finished (indicated by the done flag).

The Preprocessor object preprocesses the raw state. Preprocessing the raw state is often necessary in reinforcement learning because the raw state may be difficult for the agent to work with directly. Preprocessing can also be used to apply adversarial attacks and countermeasure simulations to the environment.

The select_action method of the agent object is called to select an action based on the current state.
The Postprocessor object has a manipulate method that takes in the current state, action, reward, next state, and done flag, and returns modified versions of these variables.
The step_learn method is then called to update the agent's knowledge based on the observed reward and next state.

The episode_learn method is used by the agent for episodic learning (for example in REINFORCE).
Finally, the model_checking_learn is used by the agent to learn via the model checking result.
```
state = env.reset()
done = False
episode_reward = 0
while done == False:
    state = Preprocessors.preprocess_state(state, agent, env)
    action = agent.select_action(state)
    next_state, reward, done = env.step(action)
    episode_reward+=reward
    state, action, reward, next_state, done = Postprocessor.manipulate(state,action,reward,next_state,done)
    agent.step_learn(next_state, reward, done)
agent.episode_learn()
if prop_query == "":
    print("Reward:",episode_reward)
else:
    model_checking_result, model_checking_info = model_checking(env,agent,Preprocessors, Postprocessor)
    agent.model_checking_learn(model_checking_result, model_checking_info, model_checker)
    print("Model Checking Result:", model_checking_result)
```

To run the safe_training component via MLflow, execute `mlflow run safe_training --env-manager=local`


#### RL Wrapper
We wrap all RL agents into a RL wrapper. This RL wrapper handles the interaction with the environment.
Via our generic interface, we can model check any kind of memoryless policy.
Our tool also supports probabilistic policies by always choosing the action with the highest probability in the probability distribution at each state and makes the policy therefore deterministic.


#### Preprocessors
A preprocessor is a tool that is used to modify the states that an RL agent receives before they are passed to the agent for learning. There are several types of preprocessors that can be used, depending on the user's preferences and goals. These include normal preprocessors, which are designed to improve the learning process; adversarial preprocessors, which are used to perform adversarial training or attacks; and defensive preprocessors, which are used to evaluate defense methods.

To use an existing preprocessor, you can use the preprocessor command-line argument `preprocessor`. For example, the following command would divide each state feature by 10 before it is observed by the RL agent: `--preprocessor="normalizer,10"`.
Note, that preprocessors get loaded automatically into the child runs. Use `--preprocessor="None"` to remove the preprocessor in the child run.
Use `--preprocessor="normalizer,12"`, to use another preprocessor in the child run.
For more information about how to use preprocessors, you can refer to the examples and to the preprocessors package.

1. If you want to create your own custom preprocessor, you can follow these steps:
2. Create a new Python script called PREPROCESSORNAME.py in the preprocessors package, and define a new class called PREPROCESSORNAME inside it.
3. Inherit the preprocessor class from the preprocessor.py script. This will give your custom preprocessor all of the necessary methods and attributes of a preprocessor.
4. Override any methods that you want to customize in your custom preprocessor.
5. Import the PREPROCESSORNAME.py script into the preprocessor builder script, which is responsible for building and configuring the preprocessor for your RL agent.
6. Add the new PREPROCESSORNAME to the build_preprocessor function, which is responsible for constructing the preprocessor object. You will need to pass any necessary arguments to the constructor of your PREPROCESSORNAME class when building the preprocessor.

It is important to make sure that your custom preprocessor is compatible with the rest of the RL agent's code, and that it performs the preprocessing tasks that you expect it to. You may need to test your custom preprocessor to ensure that it is working correctly.

It is possible to concat multiple preprocessors after each other: `--preprocessor="normalizer,10#fgsm,1"`.



#### Postprocessor
Postprocessors can be used to postprocess, for example the observed state (before being passed to the replay buffer), or to render the environment.
Poisoning attacks in reinforcement learning (RL) are a type of adversarial attack that can be used to manipulate the training process of an RL agent. In a poisoning attack, the attacker injects malicious data into the training process in order to cause the RL agent to learn a suboptimal or malicious policy.
There are several ways in which poisoning attacks can be carried out in RL. One common method is to manipulate the rewards that the RL agent receives during training. For example, the attacker could artificially inflate the rewards for certain actions, causing the RL agent to prioritize those actions and learn a suboptimal policy.
This can cause the RL agent to learn a policy that is suboptimal or even harmful.

The **Postprocessor** allows the simulation of poissioning attacks during training.
It manipulates the replay buffers of the RL agents.
With the tight integration between RL and model checking, it is possible to analyze the effetivness of poissoning attacks.

To use an existing Postprocessor, you can use the Postprocessor command-line argument `postprocessor`. For example, the following command would randomly change the value if the next state is a terminal state and stores the change value into the RL agent experience: `--postprocessor="random_done"`.
Note, that postprocessors get loaded automatically into the child runs. Use `--postprocessor="None"` to remove the preprocessor in the child run.
Use `--postprocessor="OTHERPOSTPROCESSOR"`, to use another postprocessor in the child run.
For more information about how to use postprocessors, you can refer to the examples and to the postprocessors package.

1. If you want to create your own custom postprocessor, you can follow these steps:
2. Create a new Python script called POSTPROCESSORNAME.py in the postprocessors package, and define a new class called POSTPROCESSORNAME inside it.
3. Inherit the postprocessor class from the postprocessor.py script. This will give your custom postprocessor all of the necessary methods and attributes of a postprocessor.
4. Override any methods that you want to customize in your custom postprocessor.
5. Import the POSTPROCESSORNAME.py script into the postprocessor builder script, which is responsible for building and configuring the postprocessor for your RL agent.
6. Add the new POSTPROCESSORNAME to the build_postprocessor function, which is responsible for constructing the postprocessor object. You will need to pass any necessary arguments to the constructor of your POSTPROCESSORNAME class when building the postprocessor.

It is important to make sure that your custom postprocessor is compatible with the rest of the RL agent's code, and that it performs the manipulating tasks that you expect it to. You may need to test your custom postprocessor to ensure that it is working correctly.

### Model Checking
The callback function is used to incrementally build the induced DTMC, which is then passed to the model checker Storm. The callback function is called for every available action at every reachable state by the policy. It first gets the available actions at the current state. Second, it preprocesses the state and then queries the RL policy for an action. If the chosen action is not available, the callback function chooses the first action in the available action list. The callback function then checks if the chosen action was also the trigger for the current callback function and builds the induced DTMC from there if it was.
```
def callback_function(state_valuation, action_index):
    simulator.restart(state_valuation)
    available_actions = sorted(simulator.available_actions())
    current_action_name = prism_program.get_action_name(action_index)
    # conditions on the action
    state = self.__get_clean_state_dict(state_valuation.to_json(),env.storm_bridge.state_json_example)
    state = Preprocessor.preprocess_state(state, agent, env)
    selected_action = self.__get_action_for_state(env, agent, state)
    if (selected_action in available_actions) is not True:
        selected_action = available_actions[0]
    cond1 = (current_action_name == selected_action)
    return cond1

constructor = stormpy.make_sparse_model_builder(prism_program, options,stormpy.StateValuationFunctionActionMaskDouble(callback_function))
interpreter = InterpreterBuilder.build_interpreter(interpreter_config)
interpreter.interpret()
```

#### Interpretable Reinforcement Learning
Interpretable Reinforcement Learning (IRL) focuses on making the decision-making process of reinforcement learning algorithms more transparent and understandable to human experts. It aims to bridge the gap between complex RL models, which can be difficult to interpret and understand, and human decision-making processes, which are often more intuitive and grounded in domain-specific knowledge.
IRL has a wide range of potential applications, including robotics, autonomous vehicles, healthcare, and many others. It can help to build trust in autonomous systems and facilitate collaboration between humans and machines, which is critical in domains where human expertise is essential.

COOL-MC allows the interpretation of trained RL policies in different ways via the command line argument `--interpreter`.
This argument is used during the RL model checking, which interpretes the RL policy afer it got model checked.
We currently support the interpretation of RL policies via decision trees.
The decision tree is a type of interpretable machine learning method used for classification and regression problems. It is called interpretable because the resulting model is easy to understand and interpret by humans, making it an attractive option for applications where transparency and interpretability are important.
Users have the possibility to extend COOL-MC with further IRL methods via the `common.interpreter` package.


## Setup
This guide will walk you through the setup process for running COOL-MC, including installing Docker and Visual Studio Code, and setting up a local Docker container for the COOL-MC repository.

### 1. Install Docker
To install Docker on your system, you'll need to follow these general steps:
1. Go to the Docker website (https://www.docker.com/) and click on the "Get Docker" button to download the installer for your operating system (Windows, Mac, or Linux).
2. Run the installer and follow the prompts to complete the installation process.
3. Once the installation is complete, you can verify that Docker is installed and running by opening a terminal or command prompt and running the command "docker --version". This should display the version of Docker that you have installed.

### 2. Install Visual Studio Code and some of its extensions
Installing Visual Studio Code (VSCode) is a simple process, and can be done by following these steps:

1. Go to the Visual Studio Code website (https://code.visualstudio.com/) and click on the "Download" button for your operating system (Windows, Mac, or Linux).
2. Once the download is complete, open the installer and follow the prompts to complete the installation process.
3. Add VSCode extensions: docker, Visual Studio Code Dev Containers

### 3. Create your local docker container and verify COOL-MC
Open Remote Explorer, clone repository in container volume, add https://github.com/LAVA-LAB/COOL-MC, write coolmc for each of the prompts.
Afterwards, the docker container will be created (it takes time).

Inside the working directory, install all Python packages `pip install -r requirements.txt

Verify that everything works: `python run_tests.py`

## Examples
For the visualization of the training process start MLFlow server in the background: `mlflow server -h 0.0.0.0 &`.

The taxi agent has to pick up passengers and transport them to their destination without running out of fuel. The environment terminates as soon as the taxi agent does the predefined number of jobs. After the job is done, a new guest spawns randomly at one of the predefined locations.


Run the following command to train a deep RL policy for the taxi environment:
`python cool_mc.py --project_name="taxi_experiments" --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" --prism_dir="../prism_files" --prism_file_path="transporter.prism" --seed=128 --layers=4 --neurons=512 --lr=0.0001 --batch_size=32 --num_episodes=200000 --eval_interval=100 --epsilon_dec=0.99999 --epsilon_min=0.1 --gamma=0.99 --epsilon=1 --replace=301 --reward_flag=0 --wrong_action_penalty=0 --prop="Pmax=? [ F jobs_done=2 ]" --max_steps=100 --replay_buffer_size=300000 --replace=304`

## Experiments
To run the experiments from our papers, use the bash scripts in examples (*_experiments.sh).


## PRISM Modeling Tips
COOL-MC supports PRISM as a modeling language.
To learn how to model environments via PRISM, we refer to the PRISM documentation (https://www.prismmodelchecker.org/).
It can be difficult to design your own PRISM environments. Here are some tips on how to make sure that your PRISM environment works correctly with COOL-MC:

- Utilize transition rewards to ensure accurate modeling of your environment.
- Keep in mind that when the agent reaches a terminal state, the storm simulator will terminate the simulation. As a result, transitions from terminal states will not be executed. Avoid using self-looping terminal states to avoid confusion.
- To enhance training performance, make all actions available at every state. Failure to do so may result in the agent choosing an unavailable action and receiving a penalty.
- Before beginning RL training, it's highly recommended to unit test your PRISM environment to ensure it behaves as intended.


## Command Line Arguments
The following list contains all the major COOL-MC command line arguments. It does not contain the arguments which are related to the RL algorithms. For a detailed description, we refer to the common.rl_agents package.

| Argument             | Description                                                                                                                                                                                                                                                                                 | Options                                           | Default Value  |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|----------------|
| task                 | The type of task do you want to perform.                                                                                                                                                                                                                                                    | safe_training, rl_model_checking | safe_training  |
| project_name         | The name of your project.                                                                                                                                                                                                                                                                   |                                                   | defaultproject |
| parent_run_id        | Reference to previous experiment for retraining or verification.                                                                                                                                                                                                                            | PROJECT_IDs                                       |                |
| num_episodes         | The number of training episodes.                                                                                                                                                                                                                                                            | INTEGER NUMBER                                    | 1000           |
| eval_interval        | Interval for verification while safe_training.                                                                                                                                                                                                                                              | INTEGER NUMBER                                    | 9            |
| sliding_window_size  | Sliding window size for reward averaging over episodes.                                                                                                                                                                                                                                     | INTEGER NUMBER                                    | 100            |
| rl_algorithm         | The name of the RL algorithm.                                                                                                                                                                                                                                                               | [SEE common.rl_agents.agent_builder]                               | dqn_agent      |
| prism_dir            | The directory of the PRISM files.                                                                                                                                                                                                                                                           | PATH                                              | ../prism_files |
| prism_file_path      | The name of the PRISM file.                                                                                                                                                                                                                                                                 | STR                                               |                |
| constant_definitions | Constant definitions seperated by a commata.                                                                                                                                                                                                                                                | For example: xMax=4,yMax=4,slickness=0            |                |
| prop                 | Property Query. **For safe_training:** Pmax tries to save RL policies that have higher probabilities. Pmin tries to save RL policies that have  lower probabilities. **For rl_model_checking:** In the case of induced DTMCs min/max  yield to the same property result (do not remove it). |                                                   |                |
| max_steps            | Maximal steps in the safe gym environment.                                                                                                                                                                                                                                                  |                                                   | 100            |
| disabled_features    | Disable features in the state space.                                                                                                                                                                                                                                                        | FEATURES SEPERATED BY A COMMATA                   |                |
| preprocessor     | Preprocessor configuration (each preprocessor is seperated by an  hashtag)                                                                                                                                                                            |                                                   |                |
| postprocessor    | Postprocessor configuration.                                                                                                                                                                                       |                                                   |                |
| wrong_action_penalty | If an action is not available but still chosen by the policy, return a penalty of [DEFINED HERE].                                                                                                                                                                                           |                                                   |                |
| reward_flag          | If true (1), the agent receives rewards instead of penalties.                                                                                                                                                                                                                               |                                                   | 0              |
| seed                 | Random seed for PyTorch, Numpy, Python.                                                                                                                                                                                                                                                     | INTEGER NUMBER                                    | None (-1)      |


## COOL-MC Publications

COOL-MC: A Comprehensive Tool for Learning and Model Checking:
`@inproceedings{gross2022cool,
  title={COOL-MC: a comprehensive tool for reinforcement learning and model checking},
  author={Gross, Dennis and Jansen, Nils and Junges, Sebastian and P{\'e}rez, Guillermo A},
  booktitle={Dependable Software Engineering. Theories, Tools, and Applications: 8th International Symposium, SETTA 2022, Beijing, China, October 27-29, 2022, Proceedings},
  pages={41--49},
  year={2022},
  organization={Springer}
}`

Targeted Adversarial Attacks on Deep Reinforcement Learning Policies via Model Checking:
`
@inproceedings{Gross2023targeted,
	title = { {Targeted Adversarial Attacks on Deep Reinforcement Learning Policies via Model Checking} },
	author = {         Gross, Dennis and         Sim{\~a}o, Thiago D. and         Jansen, Nils and         P{\'e}rez, Guillermo A.      },
	booktitle = { {ICAART} },
	year = { 2023 },
	url = { https://arxiv.org/abs/2212.05337 }
}
`

Turn-based Multi-Agent Reinforcement Learning Model Checking:
`
@inproceedings{Gross2023turn,
	title = { {Turn-based Multi-Agent Reinforcement Learning Model Checking} },
	author = {Dennis Gross},
	booktitle = { {ICAART} },
	year = { 2023 },
	url = { }
}
`

Model Checking for Adversarial Multi-Agent Reinforcement Learning
with Reactive Defense Methods:
`
@inproceedings{Gross2023cmarl,
	title = { {Model Checking for Adversarial Multi-Agent Reinforcement Learning
with Reactive Defense Methods} },
	author = {Gross, Dennis and         Christoph Schmidl and         Jansen, Nils and         P{\'e}rez, Guillermo A.      },
	booktitle = { {ICAPS} },
	year = { 2023 },
	url = { }
}
`

Do you have any questions or ideas for collaborative work? Contact us via dgross@science.ru.nl.


## Contributors
The main developer of this repository is Dennis Gross.
For any questions or inquiries related to the repository, write him an email via dgross@science.ru.nl.
This tool is developed with help from Nils Jansen, Sebastian Junges, and Guillermo A. Perez.
