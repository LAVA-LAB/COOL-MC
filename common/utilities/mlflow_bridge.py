import os
import random
import string
import shutil
import json
import mlflow
import time
from mlflow.tracking import MlflowClient
from distutils.dir_util import copy_tree
import math


class MlFlowBridge:

    def __init__(self, project_name, task, parent_run_id):
        self.experiment_name = project_name
        self.experiment = None
        self.task = task
        self.parent_run_id = parent_run_id.strip()
        self.client = MlflowClient()
        #mlflow.set_tracking_uri(project_name)
        # Set project for experiment
        try:
            self.experiment_id = self.client.create_experiment(self.experiment_name)
        except:
            self.experiment_id = self.client.get_experiment_by_name(self.experiment_name).experiment_id
        self.experiment = self.client.get_experiment(self.experiment_id)
        self.create_new_run(self.task, self.parent_run_id)


    def create_new_run(self, task, run_id=None):
        if run_id == None or run_id == '':
            # Create new Run
            self.run = self.client.create_run(self.experiment.experiment_id)
            self.client.set_tag(self.run.info.run_id, "task", task)
        else:
            # Choose already existing run
            run = mlflow.get_run(run_id)
            self.run = self.__copy_run(self.experiment, run)

        mlflow.start_run(self.run.info.run_id)


    def set_property_query_as_run_name(self, prop):
        mlflow.tracking.MlflowClient().set_tag(self.run.info.run_id, "mlflow.runName", prop)

    def check_folder_in_tree(self, folder_name, root_folder):
        for root, dirs, files in os.walk(root_folder):
            if folder_name in dirs:
                return True
        return False


    def __copy_run(self, experiment, run):
        # Find unique run_id
        exists = True
        # Make sure that run_id length is not too long
        run_id_length = len(run.info.run_id)+10
        if run_id_length > 50:
            run_id_length = 32
        new_run_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=run_id_length))
        run_path = os.path.join('../mlruns', experiment.experiment_id, run.info.run_id)
        new_run_path = os.path.join('../mlruns', experiment.experiment_id, new_run_id)
        # print working directory
        print("Copying run...")
        while exists:
            new_run_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=run_id_length))
            new_run_path = os.path.join('../mlruns', experiment.experiment_id, new_run_id)
            exists = self.check_folder_in_tree(new_run_id, '../mlruns')
        copy_tree(run_path, new_run_path)
        print("Copied run to " + new_run_path)
        # Modify Meta
        f = open(os.path.join(new_run_path,'meta.yaml'),'r')
        lines = f.readlines()
        f.close()
        lines[0] = lines[0].replace(run.info.run_id, new_run_id)
        lines[5] = lines[5].replace(run.info.run_id, new_run_id)
        lines[6] = lines[6].replace(run.info.run_id, new_run_id)
        lines[7] = lines[7].replace(run.info.run_id, new_run_id)
        lines[7] = lines[7].replace(run.info.run_id, new_run_id)
        lines[11] = 'start_time: ' + str(time.time()*1000).split('.')[0] + '\n'
        f = open(os.path.join(new_run_path,'meta.yaml'),'w')
        f.writelines(lines)
        f.close()
        # Update task
        f = open(os.path.join(new_run_path,'tags','task'),'w')
        f.write(self.task)
        f.close()
        # Delete already existing metrics
        metrics_path = os.path.join(new_run_path, 'metrics')
        shutil.rmtree(metrics_path)
        os.mkdir(metrics_path)
        # Delete already existing params
        params_path = os.path.join(new_run_path, 'params')
        shutil.rmtree(params_path)
        os.mkdir(params_path)
        # Get run
        run = mlflow.get_run(new_run_id)
        return run


    def save_command_line_arguments(self, command_line_arguments):
        with open("command_line_arguments.json", 'w') as f:
            json.dump(command_line_arguments, f, indent=2)
        # Command Line Arguments
        mlflow.log_artifact("command_line_arguments.json", artifact_path="meta")
        os.remove('command_line_arguments.json')
        #print("Saved")


    def load_command_line_arguments(self):
        meta_folder_path = mlflow.get_artifact_uri(artifact_path="meta").replace('/file:/','')
        # If rerun, take all the command line arguments from previous run into account except the following:
        command_line_arguments_file_path = os.path.join(meta_folder_path, 'command_line_arguments.json')[7:]
        #print(command_line_arguments_file_path)
        if os.path.exists(command_line_arguments_file_path):
            with open(command_line_arguments_file_path) as json_file:
                command_line_arguments = json.load(json_file)
                #print(command_line_arguments)
                return command_line_arguments
        return None

    def get_agent_path(self):
        model_folder_path = mlflow.get_artifact_uri(artifact_path="model").replace('file:///workspaces/coolmc/','/workspaces/coolmc/')
        return model_folder_path

    def get_run_id(self):
        return self.get_agent_path().split('/')[-3]

    def get_project_id(self):
        return self.get_agent_path().split('/')[-4]

    def log_reward(self, reward, episode):
        mlflow.log_metric(key='episode_reward', value=reward, step= episode)

    def log_best_reward(self, reward, episode):
        mlflow.log_metric(key='best_sliding_window_reward', value=reward, step= episode)

    def log_avg_reward(self, avg_reward, episode):
        mlflow.log_metric(key='avg_reward', value=avg_reward, step=episode)

    def log_property(self, property_result, property_query, episode):
        mlflow.log_metric(key=property_query, value=property_result, step= episode)

    def log_best_property_result(self, best_property_result, episode):
        mlflow.log_metric(key='best_property_result', value=best_property_result, step= episode)

    def log_result(self, result):
        mlflow.log_param('result', result)


    def get_best_reward(self,command_line_arguments):
        run_path = self.get_agent_path().replace("/artifacts/model","")
        best_reward = -math.inf
        p = os.path.join("/workspaces/coolmc/",run_path, 'metrics','best_sliding_window_reward').strip()
        path_parts = p.split("/")
        path_parts[5] = command_line_arguments['parent_run_id']
        p = "/".join(path_parts)
        if not os.path.exists(p):
            return best_reward
        with open(p) as f:
            lines = f.readlines()
            best_reward = float(lines[-1].split(' ')[-2])
        print("Best Reward: ", best_reward)
        return best_reward

    def get_best_property_result(self,command_line_arguments):
        run_path = self.get_agent_path().replace("/artifacts/model","")
        best_property_result = -math.inf
        p = os.path.join("/workspaces/coolmc/",run_path, 'metrics','best_property_result').strip()
        path_parts = p.split("/")
        path_parts[5] = command_line_arguments['parent_run_id']
        p = "/".join(path_parts)
        if not os.path.exists(p):
            return best_property_result
        with open(p) as f:
            lines = f.readlines()
            best_property_result = float(lines[-1].split(' ')[-2])

        print("Best best_property_result: ", best_property_result)
        return best_property_result


    def log_accuracy(self, acc):
        mlflow.log_param("Decision_Tree_Accuracy", acc)

    def close(self):
        mlflow.end_run()
