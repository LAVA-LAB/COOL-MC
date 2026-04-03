import os
import random
import string
import shutil
import json
import mlflow
import time
from mlflow.tracking import MlflowClient
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
        """Create a new MLflow run and copy artifacts from an existing run.

        MLflow 3.x stores metadata in a database, not in the filesystem, so we
        use the tracking API to create runs and copy artifacts instead of
        manipulating files directly.
        """
        print("Copying run...")
        # Create a new run via the API
        new_run = self.client.create_run(experiment.experiment_id)
        new_run_id = new_run.info.run_id

        # Copy artifacts from the source run to the new run
        src_artifact_uri = run.info.artifact_uri.replace('file://', '')
        dst_artifact_uri = new_run.info.artifact_uri.replace('file://', '')

        if os.path.isdir(src_artifact_uri):
            for item in os.listdir(src_artifact_uri):
                src_item = os.path.join(src_artifact_uri, item)
                dst_item = os.path.join(dst_artifact_uri, item)
                if os.path.isdir(src_item):
                    shutil.copytree(src_item, dst_item)
                else:
                    os.makedirs(dst_artifact_uri, exist_ok=True)
                    shutil.copy2(src_item, dst_item)

        # Set task tag
        self.client.set_tag(new_run_id, "task", self.task)

        print("Copied run to " + new_run_id)
        return mlflow.get_run(new_run_id)


    def save_command_line_arguments(self, command_line_arguments):
        with open("command_line_arguments.json", 'w') as f:
            json.dump(command_line_arguments, f, indent=2)
        # Command Line Arguments
        mlflow.log_artifact("command_line_arguments.json", artifact_path="meta")
        os.remove('command_line_arguments.json')
        #print("Saved")


    def load_command_line_arguments(self):
        meta_folder_path = mlflow.get_artifact_uri(artifact_path="meta")
        # Strip file:// prefix if present (MLflow 2.x adds it, MLflow 3.x does not)
        meta_folder_path = meta_folder_path.replace('file://', '')
        command_line_arguments_file_path = os.path.join(meta_folder_path, 'command_line_arguments.json')
        if os.path.exists(command_line_arguments_file_path):
            with open(command_line_arguments_file_path) as json_file:
                command_line_arguments = json.load(json_file)
                return command_line_arguments
        return None

    def get_agent_path(self):
        model_folder_path = mlflow.get_artifact_uri(artifact_path="model")
        # Strip file:// prefix if present (MLflow 2.x adds it, MLflow 3.x does not)
        model_folder_path = model_folder_path.replace('file://', '')
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


    def get_best_reward(self, command_line_arguments):
        best_reward = -math.inf
        parent_run_id = command_line_arguments['parent_run_id']
        try:
            parent_run = self.client.get_run(parent_run_id)
            value = parent_run.data.metrics.get('best_sliding_window_reward')
            if value is not None:
                best_reward = float(value)
                print("Best Reward: ", best_reward)
        except Exception:
            pass
        return best_reward

    def get_best_property_result(self, command_line_arguments):
        best_property_result = -math.inf
        parent_run_id = command_line_arguments['parent_run_id']
        try:
            parent_run = self.client.get_run(parent_run_id)
            value = parent_run.data.metrics.get('best_property_result')
            if value is not None:
                best_property_result = float(value)
                print("Best best_property_result: ", best_property_result)
        except Exception:
            pass
        return best_property_result


    def log_accuracy(self, acc):
        mlflow.log_param("Decision_Tree_Accuracy", acc)

    def close(self):
        mlflow.end_run()
