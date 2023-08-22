import os
from common.rl_agents.dqn_agent import *
from common.rl_agents.stochastic_dqn_agent import *
from common.rl_agents.cooperative_poagents_wrapper import *
from common.rl_agents.turnbased_n_agents import *
from common.rl_agents.turnbased_n_reinforce_agents import *
from common.rl_agents.hillclimbing_agent import *
from common.rl_agents.sarsa_max_agent import *
from common.rl_agents.reinforce_agent import *
from common.rl_agents.ppo import *
from common.rl_agents.turnbased_n_ppo_agents import *
'''
HOW TO ADD MORE AGENTS?
1) Create a new AGENTNAME.py with an AGENTNAME class
2) Inherit the agent-class
3) Override the methods
4) Import this py-script into this script
5) Add additional agent hyperparameters to the argparser
6) Add to build_agent the building procedure of your agent
'''
class AgentBuilder():

    @staticmethod
    def layers_neurons_to_number_of_neurons(layers, neurons):
        number_of_neurons = []
        for i in range(layers):
            number_of_neurons.append(neurons)
        return number_of_neurons

    @staticmethod
    def build_agent(model_root_folder_path, command_line_arguments, observation_space, action_space, all_actions):
        #print('Build model with', model_root_folder_path, command_line_arguments)
        #print('Environment', observation_space.shape, action_space.n)
        print(model_root_folder_path)
        model_root_folder_path = model_root_folder_path.replace("file://","")
        eps_clip = 0.2          # clip parameter for PPO
        gamma = 0.99            # discount factor

        lr_actor = 0.0003       # learning rate for actor network
        lr_critic = 0.001       # learning rate for critic network

        K_epochs = 15
        try:
            state_dimension = observation_space.shape[0]
        except:
            state_dimension = 1
        agent = None
        if command_line_arguments['rl_algorithm'] == "dqn_agent":
            number_of_neurons = AgentBuilder.layers_neurons_to_number_of_neurons(command_line_arguments['layers'],command_line_arguments['neurons'])
            agent = DQNAgent(state_dimension, number_of_neurons, action_space.n, epsilon=command_line_arguments['epsilon'], epsilon_dec=command_line_arguments['epsilon_dec'], epsilon_min=command_line_arguments['epsilon_min'], gamma=command_line_arguments['gamma'], learning_rate=command_line_arguments['lr'], replace=command_line_arguments['replace'], batch_size=command_line_arguments['batch_size'], replay_buffer_size=command_line_arguments['replay_buffer_size'])
            agent.load(model_root_folder_path)
        elif command_line_arguments['rl_algorithm'] == "stochastic_dqn_agent":
            number_of_neurons = AgentBuilder.layers_neurons_to_number_of_neurons(command_line_arguments['layers'],command_line_arguments['neurons'])
            agent = StochasticDQNAgent(state_dimension, number_of_neurons, action_space.n, gamma=command_line_arguments['gamma'], learning_rate=command_line_arguments['lr'], replace=command_line_arguments['replace'], batch_size=command_line_arguments['batch_size'], replay_buffer_size=command_line_arguments['replay_buffer_size'])
            agent.load(model_root_folder_path)
        elif command_line_arguments['rl_algorithm'] == "ppo":
            old = command_line_arguments['preprocessor'].startswith("old")
            number_of_neurons = AgentBuilder.layers_neurons_to_number_of_neurons(command_line_arguments['layers'],command_line_arguments['neurons'])


            agent = PPO(state_dimension, action_space.n, lr_actor, lr_critic, gamma, K_epochs, eps_clip, old)
            agent.load(model_root_folder_path)
        elif command_line_arguments['rl_algorithm'] == "turn_ppo":
            old = command_line_arguments['preprocessor'].startswith("old")
            number_of_neurons = AgentBuilder.layers_neurons_to_number_of_neurons(command_line_arguments['layers'],command_line_arguments['neurons'])

            agent = TurnBasedNPPOAgents(command_line_arguments, state_dimension, action_space.n, lr_actor, lr_critic, gamma, K_epochs, eps_clip, old)
            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        elif command_line_arguments['rl_algorithm'] == "cooperative_poagents":
            number_of_neurons = AgentBuilder.layers_neurons_to_number_of_neurons(command_line_arguments['layers'],command_line_arguments['neurons'])
            agent = CooperativePOAgents(command_line_arguments, state_dimension, action_space.n, all_actions, number_of_neurons)
            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        elif command_line_arguments['rl_algorithm'] == 'turnbasednagents':
            #print("Build DQN Agent.", state_dimension, action_space.n)
            number_of_neurons = AgentBuilder.layers_neurons_to_number_of_neurons(command_line_arguments['layers'],command_line_arguments['neurons'])
            agent = TurnBasedNAgents(command_line_arguments, state_dimension, action_space.n, number_of_neurons)
            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        elif command_line_arguments['rl_algorithm'] == 'turnbased_n_reinforce_agents':
            #print("Build DQN Agent.", state_dimension, action_space.n)
            number_of_neurons = command_line_arguments['neurons']
            agent = TurnBasedNReinforceAgents(command_line_arguments, state_dimension, action_space.n, number_of_neurons)

            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        elif command_line_arguments['rl_algorithm'] == 'hillclimbing':
            agent = HillClimbingAgent(state_dimension, action_space.n, gamma=command_line_arguments['gamma'], noise_scale= command_line_arguments['noise_scale'])
            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        elif command_line_arguments['rl_algorithm'] == 'sarsamax':
            agent = SarsaMaxAgent(action_space.n, epsilon=command_line_arguments['epsilon'], epsilon_dec=command_line_arguments['epsilon_dec'], epsilon_min=command_line_arguments['epsilon_min'], alpha=command_line_arguments['alpha'], gamma=command_line_arguments['gamma'])
            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        elif command_line_arguments['rl_algorithm'] == 'reinforce':
            agent = ReinforceAgent(state_dimension=state_dimension, number_of_actions=action_space.n, gamma=command_line_arguments['gamma'], hidden_layer_size= command_line_arguments['neurons'],lr=command_line_arguments['lr'])
            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        return agent
