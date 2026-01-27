import os
from common.agents.dqn_agent import *
from common.agents.cooperative_poagents_wrapper import *
from common.agents.turnbased_n_agents import *
from common.agents.hillclimbing_agent import *
from common.agents.sarsa_max_agent import *
from common.agents.reinforce_agent import *
from common.agents.bc_nn_agent import *
from common.agents.ppo_agent import *
from common.agents.stochastic_ppo_agent import *
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
        model_root_folder_path = model_root_folder_path.replace("file://","")
        try:
            state_dimension = observation_space.shape[0]
        except:
            state_dimension = 1
        agent = None
        if command_line_arguments['algorithm'] == "dqn_agent":
            number_of_neurons = AgentBuilder.layers_neurons_to_number_of_neurons(command_line_arguments['layers'],command_line_arguments['neurons'])
            agent = DQNAgent(state_dimension, number_of_neurons, action_space.n, epsilon=command_line_arguments['epsilon'], epsilon_dec=command_line_arguments['epsilon_dec'], epsilon_min=command_line_arguments['epsilon_min'], gamma=command_line_arguments['gamma'], learning_rate=command_line_arguments['lr'], replace=command_line_arguments['replace'], batch_size=command_line_arguments['batch_size'], replay_buffer_size=command_line_arguments['replay_buffer_size'])
            agent.load(model_root_folder_path)
        elif command_line_arguments['algorithm'] == "cooperative_poagents":
            number_of_neurons = AgentBuilder.layers_neurons_to_number_of_neurons(command_line_arguments['layers'],command_line_arguments['neurons'])
            agent = CooperativePOAgents(command_line_arguments, state_dimension, action_space.n, all_actions, number_of_neurons)
            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        elif command_line_arguments['algorithm'] == 'turnbasednagents':
            #print("Build DQN Agent.", state_dimension, action_space.n)
            number_of_neurons = AgentBuilder.layers_neurons_to_number_of_neurons(command_line_arguments['layers'],command_line_arguments['neurons'])
            agent = TurnBasedNAgents(command_line_arguments, state_dimension, action_space.n, number_of_neurons)
            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        elif command_line_arguments['algorithm'] == 'hillclimbing':
            agent = HillClimbingAgent(state_dimension, action_space.n, gamma=command_line_arguments['gamma'], noise_scale= command_line_arguments['noise_scale'])
            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        elif command_line_arguments['algorithm'] == 'sarsamax':
            agent = SarsaMaxAgent(action_space.n, epsilon=command_line_arguments['epsilon'], epsilon_dec=command_line_arguments['epsilon_dec'], epsilon_min=command_line_arguments['epsilon_min'], alpha=command_line_arguments['alpha'], gamma=command_line_arguments['gamma'])
            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        elif command_line_arguments['algorithm'] == 'reinforce':
            agent = ReinforceAgent(state_dimension=state_dimension, number_of_actions=action_space.n, gamma=command_line_arguments['gamma'], hidden_layer_size= command_line_arguments['neurons'],lr=command_line_arguments['lr'])
            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        elif command_line_arguments['algorithm'] == 'bc_nn_agent':
            number_of_neurons = AgentBuilder.layers_neurons_to_number_of_neurons(command_line_arguments['layers'],command_line_arguments['neurons'])
            agent = BCNNAgent(state_dimension, number_of_neurons, action_space.n, learning_rate=command_line_arguments['lr'], batch_size=command_line_arguments['batch_size'])
            if model_root_folder_path!= None:
                agent.load(model_root_folder_path)
        elif command_line_arguments['algorithm'] == 'ppo_agent':
            number_of_neurons = AgentBuilder.layers_neurons_to_number_of_neurons(command_line_arguments['layers'], command_line_arguments['neurons'])
            agent = PPOAgent(state_dimension, number_of_neurons, action_space.n, gamma=command_line_arguments['gamma'], lr=command_line_arguments['lr'], batch_size=command_line_arguments['batch_size'])
            if model_root_folder_path != None:
                agent.load(model_root_folder_path)
        elif command_line_arguments['algorithm'] == 'stochastic_ppo_agent':
            number_of_neurons = AgentBuilder.layers_neurons_to_number_of_neurons(command_line_arguments['layers'], command_line_arguments['neurons'])
            agent = StochasticPPOAgent(state_dimension, number_of_neurons, action_space.n, gamma=command_line_arguments['gamma'], lr=command_line_arguments['lr'], batch_size=command_line_arguments['batch_size'])
            if model_root_folder_path != None:
                agent.load(model_root_folder_path)
        return agent
