from common.utilities.project import Project
import sys
from common.safe_gym.safe_gym import SafeGym
from common.utilities.helper import *
import gym
import random
import math
import numpy as np
import torch
from collections import deque
import gc
from common.behavioral_cloning_dataset.behavioral_cloning_dataset_builder import *



def train(project, env, prop_type=''):
    all_episode_rewards = deque(maxlen=project.command_line_arguments['sliding_window_size'])
    all_property_results = deque(maxlen=project.command_line_arguments['sliding_window_size'])
    best_reward_of_sliding_window = project.mlflow_bridge.get_best_reward(project.command_line_arguments)
    best_property_result = project.mlflow_bridge.get_best_property_result(project.command_line_arguments)
    mdp_reward_result = None
    satisfied = False



    project.agent.load_env(env)
    # Behavioral Cloning Dataset
    behavioral_cloning_dataset_builder = BehavioralCloningDatasetBuilder()
    dataset = behavioral_cloning_dataset_builder.build(project.command_line_arguments['behavioral_cloning'])
    if dataset != None:
        dataset.create(env)
        data = dataset.get_data()
        # Behavioral cloning
        training_epoch, train_accuracy, test_accuracy, train_loss, test_loss = project.agent.behavioral_cloning(env, data, project.command_line_arguments['bc_epochs'])
        if training_epoch != None:
            if train_accuracy != None:
                project.mlflow_bridge.log_property(train_accuracy, "Behavioral Cloning Training Accuracy", training_epoch)
            if test_accuracy != None:
                project.mlflow_bridge.log_property(test_accuracy, "Behavioral Cloning Test Accuracy", training_epoch)
            if train_loss != None:
                project.mlflow_bridge.log_property(train_loss, "Behavioral Cloning Training Loss", training_epoch)
            if test_loss != None:
                project.mlflow_bridge.log_property(test_loss, "Behavioral Cloning Test Loss", training_epoch)
            # Save the trained BC model immediately after training
            project.save()



    try:
        for episode in range(project.command_line_arguments['num_episodes']): 
            state = env.reset()
            done = False
            episode_reward = 0
            while done == False:
                if state.__class__.__name__ == 'int':
                    state = [state]
                if project.preprocessors != None:
                    # Preprocessing
                    # Add preprocessor loop here
                    for preprocessor in project.preprocessors:
                        state = preprocessor.preprocess(project.agent, state, env.action_mapper, "", project.command_line_arguments['deploy'])
                action = project.agent.select_action(state, project.command_line_arguments['deploy'])
                next_state, reward, done, truncated, info = env.step(action)
                if next_state.__class__.__name__ == 'int':
                    next_state = [next_state]
                if project.command_line_arguments['deploy']==False:
                    if project.manipulator != None:
                        # Manipulating
                        state, action, reward, next_state, done = project.manipulator.postprocess(project.agent, state, action, reward, next_state, done)
                    project.agent.store_experience(state, action, reward, next_state, done)
                    project.agent.step_learn()
                state = next_state
                episode_reward+=reward
            # Log rewards
            all_episode_rewards.append(episode_reward)
            project.mlflow_bridge.log_reward(all_episode_rewards[-1], episode)
            reward_of_sliding_window = np.mean(list(all_episode_rewards))
            project.mlflow_bridge.log_avg_reward(reward_of_sliding_window, episode)

            if project.command_line_arguments['deploy']==False:
                project.agent.episodic_learn()

            if episode % project.command_line_arguments['eval_interval']==0 and prop_type != 'reward':
                mdp_reward_result, model_checking_info = env.storm_bridge.model_checker.induced_markov_chain(project.agent, project.preprocessors, env, project.command_line_arguments['constant_definitions'], project.command_line_arguments['prop'])
                project.agent.model_checking_learn(mdp_reward_result, model_checking_info, env.storm_bridge.model_checker)
                all_property_results.append(mdp_reward_result)

                if (all_property_results[-1] == min(all_property_results) and prop_type == "min_prop") or (all_property_results[-1] == max(all_property_results) and prop_type == "max_prop"):
                    best_property_result = all_property_results[-1]
                    project.mlflow_bridge.log_best_property_result(best_property_result, episode)
                    if project.command_line_arguments['deploy']==False:
                        project.save()
                    if (best_property_result <= project.command_line_arguments['training_threshold'] and prop_type == "min_prop" and DEFAULT_TRAINING_THRESHOLD != project.command_line_arguments['training_threshold']) or ( best_property_result >= project.command_line_arguments['training_threshold'] and prop_type == "max_prop" and DEFAULT_TRAINING_THRESHOLD != project.command_line_arguments['training_threshold']):
                        print("Property satisfied!")
                        satisfied = True
                # Log Property result
                project.mlflow_bridge.log_property(all_property_results[-1], 'Property Result', episode)

            # Update best sliding window value
            if reward_of_sliding_window  > best_reward_of_sliding_window and len(all_episode_rewards)>=project.command_line_arguments['sliding_window_size']:
                best_reward_of_sliding_window = reward_of_sliding_window
                project.mlflow_bridge.log_best_reward(best_reward_of_sliding_window, episode)
                if prop_type=='reward' and project.command_line_arguments['deploy']==False:
                    project.save()

                if (prop_type == "reward" and best_reward_of_sliding_window >= project.command_line_arguments['training_threshold']) and DEFAULT_TRAINING_THRESHOLD != project.command_line_arguments['training_threshold']:
                    print("Property satisfied!")
                    satisfied = True

            print(episode, "Episode\tReward", episode_reward, '\tAverage Reward', reward_of_sliding_window, "\tLast Property Result:", mdp_reward_result)
            gc.collect()
            if satisfied:
                break
    except KeyboardInterrupt:
        torch.cuda.empty_cache()
        gc.collect()
    finally:
        torch.cuda.empty_cache()

    return best_reward_of_sliding_window, best_property_result
