from common.agents.agent_builder import AgentBuilder
from common.preprocessors.preprocessor_builder import PreprocessorBuilder
from common.postprocessors.postprocessor_builder import PostprocessorBuilder
from common.state_labelers.state_labeler_builder import StateLabelerBuilder
from common.utilities.mlflow_bridge import MlFlowBridge

class Project():

    def __init__(self, command_line_arguments):
        self.command_line_arguments = command_line_arguments
        self.mlflow_bridge = None
        self.agent = None
        self.preprocessors = None
        self.manipulator = None
        self.state_labelers = None


    def init_mlflow_bridge(self, project_name, task, parent_run_id):
        self.mlflow_bridge = MlFlowBridge(project_name, task, parent_run_id)

    def load_saved_command_line_arguments(self):
        saved_command_line_arguments = self.mlflow_bridge.load_command_line_arguments()
        if saved_command_line_arguments != None:
            old_task = saved_command_line_arguments['task']
            try:
                del saved_command_line_arguments['prop']
            except:
                pass
            try:
                del saved_command_line_arguments['task']
            except:
                pass
            try:
                del saved_command_line_arguments['parent_run_id']
            except:
                pass
            try:
                del saved_command_line_arguments['project_name']
            except:
                pass
            try:
                del saved_command_line_arguments['constant_definitions']
            except:
                pass
            if self.command_line_arguments['preprocessor'] != '':
                # Only delete it if it is not set by the command line (in this case take new one)
                # If "None", later during building none will be created
                try:
                    del saved_command_line_arguments['preprocessor']
                except:
                    pass
            if self.command_line_arguments['behavioral_cloning'] != '':
                # Only delete it if it is not set by the command line (in this case take new one)
                # If "None", later during building none will be created
                try:
                    del saved_command_line_arguments['behavioral_cloning']
                except:
                    pass
            try:
                del saved_command_line_arguments['bc_epochs']
            except:
                pass
            try:
                del saved_command_line_arguments['epsilon']
            except:
                pass
            try:
                del saved_command_line_arguments['epsilon_dec']
            except:
                pass
            try:
                del saved_command_line_arguments['epsilon_min']
            except:
                pass
            try:
                del saved_command_line_arguments['seed']
            except:
                pass
            try:
                del saved_command_line_arguments['interpreter']
            except:
                pass
            try:
                del saved_command_line_arguments['deploy']
            except:
                pass
            try:
                del saved_command_line_arguments['training_threshold']
            except:
                pass
            try:
                del saved_command_line_arguments['num_episodes']
            except:
                pass
            try:
                del saved_command_line_arguments['eval_interval']
            except:
                pass
            try:
                del saved_command_line_arguments['prop_type']
            except:
                pass
            if self.command_line_arguments['postprocessor'] != '':
                # Only delete it if it is not set by the command line (in this case take new one)
                # If "None", later during building none will be created
                try:
                    del saved_command_line_arguments['postprocessor']
                except:
                    pass
            if self.command_line_arguments.get('state_labeler', '') != '':
                # Only delete it if it is not set by the command line (in this case take new one)
                try:
                    del saved_command_line_arguments['state_labeler']
                except:
                    pass
            try:
                del saved_command_line_arguments['range_plotting']
            except:
                pass

            for key in saved_command_line_arguments.keys():
                self.command_line_arguments[key] = saved_command_line_arguments[key]





    def create_agent(self, command_line_arguments, observation_space, number_of_actions, all_actions):
        agent = None
        try:
            model_folder_path = self.mlflow_bridge.get_agent_path()
            # Build agent with the model and the hyperparameters
            agent = AgentBuilder.build_agent(model_folder_path, command_line_arguments, observation_space, number_of_actions, all_actions)
            print("Agent loaded.")
        except Exception as msg:
            # If Model was not saved
            agent = AgentBuilder.build_agent(None, command_line_arguments, observation_space, number_of_actions, all_actions)
        self.agent = agent
        return self.agent

    def create_preprocessor(self, command_line_arguments, observation_space, number_of_actions, state_mapper):
        preprocessor_path = self.mlflow_bridge.get_agent_path().replace("model", "")
        # Build agent with the model and the hyperparameters
        self.preprocessors = PreprocessorBuilder.build_preprocessors(preprocessor_path, command_line_arguments, observation_space, number_of_actions, state_mapper)
        if self.preprocessors != None:
            print("Preprocessors", command_line_arguments['preprocessor'])
            #print("Preprocessor loaded from", preprocessor_path)
        return self.preprocessors


    def create_postprocessor(self, command_line_arguments, observation_space, number_of_actions, state_mapper):
        postprocessor_path = self.mlflow_bridge.get_agent_path().replace("model", "")
        # Build agent with the model and the hyperparameters
        self.manipulator = PostprocessorBuilder.build_postprocessor(postprocessor_path, command_line_arguments, observation_space, number_of_actions, state_mapper)
        if self.manipulator != None:
            print("Postprocessor loaded.")
        return self.manipulator

    def create_state_labelers(self, command_line_arguments):
        """Create state labelers from command line arguments.

        Args:
            command_line_arguments: Dictionary with command line arguments

        Returns:
            List of StateLabeler instances or None
        """
        state_labeler_config = command_line_arguments.get('state_labeler', '')
        self.state_labelers = StateLabelerBuilder.build_state_labelers(state_labeler_config)
        if self.state_labelers is not None:
            print(f"State labelers: {command_line_arguments['state_labeler']}")
        return self.state_labelers

    def save(self):
        # Agent
        if self.agent != None:
            self.agent.save()
        # Preprocessor
        if self.preprocessors != None:
            for preprocessor in self.preprocessors:
                preprocessor.save()
        # Manipulator
        if self.manipulator != None:
            self.manipulator.save()
        # Save Command Line Arguments
        self.mlflow_bridge.save_command_line_arguments(self.command_line_arguments)

    def close(self):
        self.mlflow_bridge.close()
