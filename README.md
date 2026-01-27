# COOL-MC
Probabilistic model checking is a formal verification technique that exhaustively analyzes all possible states and transitions of a stochastic system to check whether it satisfies a given quantitative specification, such as verifying that the probability of reaching a goal state is above a certain threshold. Unfortunately, this approach faces scalability challenges, as the number of states can grow exponentially with system complexity.

Behavioral cloning can help address this limitation. On a smaller, tractable version of the system, model checking computes an optimal policy (a mapping from states to actions) with respect to the property of interest. However, this policy only covers states present in the smaller system. To handle new states in the larger system, a function approximator is trained to imitate the optimal policy, enabling it to generalize to unseen states. This learned policy is then applied to the larger system. Since the policy fixes actions, the complex system reduces to only reachable states, making verification tractable. This allows us to check whether the larger system still satisfies the property under the learned policy. If so, we obtain formal guarantees at scale.

Beyond behavioral cloning, COOL-MC also enables verification of policies trained via reinforcement learning (RL). In RL, an agent learns a policy with respect to a specified objective, typically maximizing cumulative reward. While the reward function is designed to guide the agent toward behavior that satisfies the specification, it often cannot fully capture complex requirements. COOL-MC allows users to formally verify whether the learned policy actually satisfies the desired specification.

At a high-level, COOL-MC takes a formal environment model of the stochastic system, a specification, and a trained policy as input, and outputs whether the policy satisfies or violates the specification.


![diagram](https://github.com/LAVA-LAB/COOL-MC/blob/main/images/cool-mc.png)

COOL-MC has been extended to support various research directions, with some features currently available in separate branches:

**Explainable AI.** COOL-MC supports multiple approaches for interpreting and explaining trained policies:
- *Co-activation graph analysis* maps neural network inner workings by analyzing neuron activation patterns, providing insight into safe sequential decision-making.
- *Temporal safety explanations* combine local explainable AI with PCTL model checking to explain complex, safe sequential decision-making over time.
- *Safety-oriented pruning* combines neural network pruning with model checking to quantify the effects of pruning and the impact of neural connections on complex safety properties.
- *Rashomon effect analysis* uses formal verification to compare multiple policies that exhibit identical behavior but differ in their internal structure, such as feature attributions.

**Safe Learning.** COOL-MC supports safe learning techniques:
- *Shielding* enforces safety constraints during training and deployment by filtering unsafe actions before execution, ensuring the agent never violates specified safety properties.
- *Behavioral cloning* trains agents by imitating optimal policies computed via model checking, enabling safe policy learning from formally verified demonstrations.

**Secure AI.** COOL-MC enables security analysis of trained policies:
- *Targeted adversarial attacks* measure the exact impact of adversarial noise on temporal logic properties and craft optimal attacks against trained policies.
- *Robustness verification* assesses a policy's robustness against adversarial attacks.
- *Adversarial multi-agent verification* verifies cooperative multi-agent agents in settings with or without adversarial attacks or denoisers.
- *Turn-based multi-agent verification* verifies turn-based multi-agent agents in stochastic multiplayer games.

**Policy Debugging.** COOL-MC provides techniques to analyze and debug learned deterministic and stochastic memoryless policies:
- *Permissive policies* abstract features by grouping ("lumping") feature values together, allowing the policy to select multiple actions per state. This yields best- and worst-case bounds on policy performance, helping to identify states where the policy is sensitive to specific feature values.
- *Feature remapping* transforms feature values before feeding them into the policy, enabling analysis of how policies behave when deployed with different configurations than they were trained on, such as varying sensor capacities.


**Quantum Machine Learning.** COOL-MC verifies trained quantum computing policies under quantum noise conditions, including bit-flip, phase-flip, and depolarizing noise. This enables safety verification before deployment on expensive quantum hardware.

**LLM Policy Verification.** COOL-MC supports verification of large language model decision-making:
- *Memoryless sequential decision-making verification* constructs the reachable portion of an MDP guided by LLM-chosen actions and verifies whether the LLM policy satisfies safety properties.
- *Counterfactual LLM reasoning* enhances trained policy safety post-training by using LLM reasoning to improve and clarify safe agent execution.


Together, these capabilities make COOL-MC a comprehensive tool for training, verifying, explaining, debugging, and securing data-driven policies across classical, quantum, and LLM-based systems.


## 🚀 Setup
This guide will walk you through the setup process for running COOL-MC, including installing Docker and Visual Studio Code, and setting up a local Docker container for the COOL-MC repository.

### 1. 🐳 Install Docker
To install Docker on your system, you'll need to follow these general steps:
1. Go to the Docker website (https://www.docker.com/) and click on the "Get Docker" button to download the installer for your operating system (Windows, Mac, or Linux).
2. Run the installer and follow the prompts to complete the installation process.
3. Once the installation is complete, you can verify that Docker is installed and running by opening a terminal or command prompt and running the command "docker --version". This should display the version of Docker that you have installed.

### 2. 💻 Install Visual Studio Code and some of its extensions
Installing Visual Studio Code (VSCode) is a simple process, and can be done by following these steps:

1. Go to the Visual Studio Code website (https://code.visualstudio.com/) and click on the "Download" button for your operating system (Windows, Mac, or Linux).
2. Once the download is complete, open the installer and follow the prompts to complete the installation process.
3. Add VSCode extensions: docker, Visual Studio Code Dev Containers

### 3. 📦 Create your local docker container and verify COOL-MC
Open Remote Explorer, clone repository in container volume, add https://github.com/LAVA-LAB/COOL-MC, make sure that the working space directory is set to `coolmc`.
Afterwards, the docker container will be created (it takes time).

Verify that everything works: `python run_tests.py` ✅

## 🎯 Getting Started

For detailed examples, see the `examples` directory which contains bash scripts for running various experiments. A full list of command-line arguments with descriptions is documented in `common/utilities/helper.py`.

### 🏋️ Training

COOL-MC allows you to train a policy in an environment modeled as a Markov Decision Process (MDP), specified in the [PRISM language](https://www.prismmodelchecker.org/manual/ThePRISMLanguage/Introduction). You can train agents using two approaches: behavioral cloning (learning from optimal demonstrations) or reinforcement learning (learning through environment interaction).

To visualize training progress, start the MLflow server:
```bash
mlflow server -h 0.0.0.0 &
```

Then open `http://localhost:5000` in your browser. 📊

#### 🎓 Behavioral Cloning

Behavioral cloning trains agents by imitating an optimal policy computed by the Storm model checker. This is useful when you want to:
- Train policies on smaller, tractable models and generalize to larger systems
- Leverage formal verification to generate expert demonstrations
- Quickly bootstrap policy learning with optimal behavior

To train an agent using behavioral cloning:
```bash
python cool_mc.py --task=safe_training \
    --project_name="bc_experiment" \
    --prism_file_path="scheduling_task.prism" \
    --prism_dir="../prism_files" \
    --constant_definitions="" \
    --algorithm=bc_nn_agent \
    --behavioral_cloning="raw_dataset;../prism_files/scheduling_task.prism;Pmax=? [ F \"goal\" ];" \
    --bc_epochs=100 \
    --layers=3 \
    --neurons=128 \
    --lr=0.001 \
    --batch_size=32 \
    --num_episodes=2 \
    --eval_interval=1
```

Key arguments for behavioral cloning:
- `--algorithm=bc_nn_agent`: Use the behavioral cloning neural network agent.
- `--behavioral_cloning`: Specifies the dataset configuration with format `"dataset_type;prism_file;property;"`. The property defines what optimal behavior to learn.
- `--bc_epochs`: Number of supervised learning epochs to train the agent.
- `--num_episodes`: Minimal episodes for environment interaction (usually set to 1-2 for BC).

The behavioral cloning process:
1. Storm computes the optimal policy (scheduler) for the specified property
2. State-action pairs are extracted from the optimal policy
3. A neural network agent is trained via supervised learning to imitate this policy
4. The trained agent can then generalize to unseen states

#### 🤖 Reinforcement Learning

RL agents learn through trial and error, interacting with the environment and receiving rewards. The agent updates its policy to maximize cumulative reward.

To train an RL agent:
```bash
python cool_mc.py --task=safe_training \
    --project_name="my_experiment" \
    --prism_file_path="transporter.prism" \
    --prism_dir="../prism_files" \
    --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" \
    --prop="Pmax=? [ F jobs_done=2 ]" \
    --algorithm=dqn_agent \
    --num_episodes=10000 \
    --eval_interval=100 \
    --max_steps=100
```

Key arguments:
- `--task`: Set to `safe_training` for training with periodic verification.
- `--project_name`: Name of your experiment (results are stored under this name in MLflow).
- `--prism_file_path`: The PRISM file defining your environment.
- `--constant_definitions`: Constants for your PRISM model, separated by commas.
- `--prop`: The property to verify, specified in PCTL (e.g., `Pmax=? [ F goal ]`). During training, `Pmax` indicates that COOL-MC should store policies that maximize the specified probability (in this case, the probability of eventually reaching the goal). Conversely, `Pmin` would store policies that minimize the probability.
- `--algorithm`: The RL algorithm to use (e.g., `dqn_agent`, `reinforce_agent`, `sarsamax`, `hillclimbing`).
- `--num_episodes`: Number of training episodes.
- `--eval_interval`: How often (in episodes) to run verification during training.
- `--max_steps`: Maximum steps per episode.

### ✅ Verification

Once a policy is trained, COOL-MC verifies it by incrementally constructing the induced Discrete Time Markov Chain (DTMC). The process works as follows:

1. **Start from the initial state.** The construction begins at the environment's initial state.
2. **Query the policy.** For the current state, the trained policy is queried to determine which action it selects.
3. **Expand successor states.** Only the states reachable under the selected action are added to the model, along with their transition probabilities.
4. **Repeat.** This process continues for all newly reachable states until no new states are discovered.

The resulting DTMC contains only the states and transitions that the policy actually visits, which is typically much smaller than the full MDP. This induced model is then passed to the [Storm model checker](https://www.stormchecker.org/) to determine whether the policy satisfies the specification.

To verify a previously trained policy:
```bash
python cool_mc.py --task=rl_model_checking \
    --parent_run_id=YOUR_TRAINING_RUN_ID \
    --prism_file_path="transporter.prism" \
    --prism_dir="../prism_files" \
    --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" \
    --prop="P=? [ F jobs_done=2 ]"
```

Key arguments:
- `--task`: Set to `rl_model_checking` for verification only.
- `--parent_run_id`: The MLflow run ID of the trained policy you want to verify.

The verification output indicates whether the policy satisfies or violates the specification, along with the computed probability or expected value. You can view detailed verification results, metrics, and artifacts in the MLflow UI at `http://localhost:5000`. 📊

## 🧩 Core Modules

### 1. 🎮 Safe Gym (`common/safe_gym/`)

**Purpose**: Bridge between PRISM probabilistic models and Gymnasium RL interface.

**Key Components**:
- **SafeGym**: OpenAI Gym-compatible wrapper around Storm simulator
- **StormBridge**: Direct interface to Storm probabilistic model checker
- **ModelChecker**: Constructs induced DTMCs from RL policies
- **StateMapper**: Converts PRISM state valuations to numpy arrays
- **ActionMapper**: Maps action names to integer indices

**Workflow**:
```
PRISM Model (*.prism)
  → Storm Simulator
  → SafeGym (exposes Gymnasium API)
  → RL Agent can train via standard step(action) interface
```

### 2. 🤖 Agents (`common/agents/`)

**Supported Algorithms**:
- **DQN**: Deep Q-Network with experience replay
- **SARSA Max**: Tabular SARSA
- **REINFORCE**: Policy gradient method
- **Hill Climbing**: Evolutionary strategy
- **Behavioral Cloning**: Supervised learning from demonstrations
- **Multi-agent**: Cooperative agents, Turn-based N agents
- ...

**Interface** (defined in `agent.py`):
- `select_action(state)` - Choose action given state
- `store_experience(transition)` - Add to replay buffer
- `step_learn()` - Per-step learning update
- `episodic_learn()` - End-of-episode learning
- ...

### 3. 🔧 Preprocessors (`common/preprocessors/`)

**Categories**:
- **Normalization**: `normalizer`, `rounder`
- **Adversarial Attacks**: `fgsm`, `deepfool`, `ffgsm`
- **Defenses**: `rounder` (quantization)
- **Robustness Testing**: `integer_l1_robustness`

**Purpose**:
- Modify states before agent observation
- Simulate adversarial conditions during training/verification
- Enable defensive training strategies
- Chain multiple preprocessors: `normalizer,10#fgsm,0.1`

### 4. 🔄 Postprocessors (`common/postprocessors/`)

**Purpose**:
- Manipulate experience tuples before storage
- Simulate data poisoning attacks
- Research on replay buffer vulnerabilities

### 5. 📚 Behavioral Cloning Dataset (`common/behavioral_cloning_dataset/`)

**Purpose**: Generate expert demonstration datasets from optimal policies.

**Key Components**:
- **BehavioralCloningDataset**: Base class for dataset generation
- **RawDataset**: Returns unprocessed state-action pairs
- **BehavioralCloningDatasetBuilder**: Factory for dataset creation

**How it Works**:
1. Parse PRISM model and property specification
2. Run Storm model checker to find optimal policy (scheduler)
3. Extract state-action pairs from the optimal scheduler
4. Convert states to numpy arrays using StateMapper
5. Map action names to indices using ActionMapper
6. Return dataset as `{X_train, y_train, X_test, y_test}`

**Use Case**: Train behavioral cloning agents by imitating optimal policies computed by Storm, enabling supervised learning as an alternative to RL.

### 6. 🏷️ State Labelers (`common/state_labelers/`)

**Purpose**: Add custom labels to states in the Storm model after incremental building completes, enabling property specifications over agent behavior (e.g., `P=? ["critical" U "goal"]`).

**Key Components**:
- **StateLabeler**: Base class defining the labeling interface
- **CriticalStateLabeler**: Labels states based on Q-value spread (max - min)
- **TopTwoGapLabeler**: Labels states based on gap between top two Q-values
- **StateLabelerBuilder**: Factory for building labelers from configuration strings

**Available Labelers**:
- `critical_state;threshold=<float>`: Labels states as `critical` (uncertain) when the gap between max and min Q-values is below the threshold, or `non_critical` (confident) otherwise
- `top_two_gap;threshold=<float>`: Labels states as `confident` when the gap between the highest and second-highest Q-values is above the threshold, or `not_confident` otherwise
- ...

**Use Case**: AI Explainability over time.

### 7. 🔍 Interpreters (`common/interpreter/`)

**Purpose**:
- Extract interpretable representations from RL policies
- **Decision Tree Interpreter**: Converts neural network policies to decision trees
- Provides human-readable policy explanations

### 8. 🛠️ Utilities (`common/utilities/`)

**Key Modules**:
- **training.py**: Main training loop implementation
- **project.py**: Project state management, MLflow integration
- **mlflow_bridge.py**: Experiment tracking wrapper
- **helper.py**: CLI parsing, random seed management


## Publications

If you use COOL-MC in your research, please cite the relevant publication(s):

| Year | Title | Venue | Link |
|------|-------|-------|------|
| 2022 | COOL-MC: A Comprehensive Tool for Reinforcement Learning and Model Checking | SETTA | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-21213-0_3) |
| 2023 | Targeted Adversarial Attacks on Deep Reinforcement Learning Policies via Model Checking | ICAART | [arXiv](https://arxiv.org/abs/2212.05337) |
| 2023 | Turn-based Multi-Agent Reinforcement Learning Model Checking | ICAART | [arXiv](https://arxiv.org/abs/2501.03187) |
| 2023 | Model Checking for Adversarial Multi-Agent Reinforcement Learning with Reactive Defense Methods | ICAPS | [Paper](https://ojs.aaai.org/index.php/ICAPS/article/view/27191) |
| 2024 | Safety-Oriented Pruning and Interpretation of Reinforcement Learning Policies | ESANN | [arXiv](https://arxiv.org/abs/2409.10218) |
| 2024 | Probabilistic Model Checking of Stochastic Reinforcement Learning Policies | ICAART | [arXiv](https://arxiv.org/abs/2403.18725) |
| 2024 | Enhancing RL Safety with Counterfactual LLM Reasoning | ICTSS | [arXiv](https://arxiv.org/abs/2409.10188) |
| 2025 | Co-Activation Graph Analysis of Safety-Verified and Explainable Deep Reinforcement Learning Policies | ICAART | [arXiv](https://arxiv.org/abs/2501.03142) |
| 2025 | PCTL Model Checking for Temporal RL Policy Safety Explanations | SAC | [Paper](https://web-backend.simula.no/sites/default/files/2025-08/SAC_2025___PCTL_Model_Checking_for_Temporal_RL_Policy_Safety_Explanations.pdf) |
| 2025 | Translating the Rashomon Effect to Sequential Decision-Making Tasks | arXiv | [arXiv](https://arxiv.org/abs/2512.17470) |
| 2025 | Formal Verification of Noisy Quantum Reinforcement Learning Policies | arXiv | [arXiv](https://arxiv.org/abs/2512.01502) |
| 2025 | Verifying Memoryless Sequential Decision-making of Large Language Models | OVERLAY | [arXiv](https://arxiv.org/abs/2510.06756) |



</details>

## Collaboration & Contributors

COOL-MC is developed and maintained by [Dennis Gross](https://www.linkedin.com/in/dennis-gro%C3%9F-b25044326/), with contributions from [Nils Jansen](https://www.linkedin.com/in/nils-jansen-48458011a/), [Sebastian Junges](https://www.linkedin.com/in/sebastian-junges/), and [Guillermo A. Pérez](https://gaperez64.github.io/).

We welcome collaboration and are happy to answer questions about COOL-MC. If you are interested in contributing, have ideas for new features, or want to discuss potential research collaborations, feel free to reach out.