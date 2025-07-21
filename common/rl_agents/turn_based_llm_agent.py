
import numpy as np
from ollama import Client

def get_llm_response_text(
    model_name: str,
    prompt: str,
    host: str = "http://localhost:11434",
    headers: dict | None = None
) -> str:
    """
    Send a single-user prompt to an Ollama model and return its most likely response text.

    Args:
        model_name: the name (and version) of the model, e.g. "gemma3:4b"
        prompt: the user's content to send
        host: Ollama API endpoint
        headers: any extra headers for the Client

    Returns:
        The most likely text content of the model's response.
    """
    # Default header if none provided
    headers = headers or {"x-some-header": "some-value"}

    # Initialize the client
    client = Client(host=host, headers=headers)

    # Send the chat request with deterministic sampling parameters
    response = client.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={"seed": 42, "temperature": 0}
        
    )

    # Return the model's response content
    return response.message.content


class TurnBasedLLMAgent():

    def __init__(self, first_agent_model, second_agent_model):
        print("TurnBasedLLMAgent initialized with models:", first_agent_model, second_agent_model)
        self.first_agent_model = first_agent_model
        self.second_agent_model = second_agent_model
        self.state_action_pairs = {}
        self.number_of_faulty_outputs1 = 0
        self.number_of_faulty_outputs2 = 0

    def get_faulty_outputs(self):
        """
        Returns the number of faulty outputs of the agent.
        This is used to measure the quality of the agent.
        """
        return [self.number_of_faulty_outputs1, self.number_of_faulty_outputs2]
        

    def select_action(self, state : np.ndarray, deploy : bool =False):
        """
        The agent gets the OpenAI Gym state and makes a decision.

        Args:
            state (np.ndarray): The state of the OpenAI Gym.
            deploy (bool, optional): If True, do not add randomness to the decision making (e.g. deploy=True in model checking). Defaults to False.
        """
        ignore_features = []
        replace_feature_names = {}
        # Get turn feature
        current_turn = self.state_mapper.get_feature_value(state, "turn")
        #print("Current turn:", current_turn)
        if current_turn == 1:
            model_name = self.first_agent_model
        else:
            model_name = self.second_agent_model
        print("Selected model for turn", current_turn, ":", model_name)
        if self.env_name == "15_slippery_sticks":
            ignore_features = ["turn", "done"]
            replace_feature_names = {"remaining_sticks": "remaining sticks"}
        elif self.env_name == "pick_or_grap":
            ignore_features = ["turn", "done"]
            replace_feature_names = {"coins1": "collected player 1 coins", "coins2": "collected player 2 coins", "remaining_coins": "remaining coins in the pot"}
        

        actions = self.action_mapper.actions
        prompt = self.env_description + "\n" + "" + self.state_mapper.get_textual_state_representation(state, ignore_features, replace_feature_names=replace_feature_names) + "\n" +  f"You are player {current_turn}. What is the next action to take? Please answer with the EXACT action name "+ str(self.action_mapper.actions)+" only."

        if prompt not in self.state_action_pairs.keys():
            print(prompt)
            #print("Prompt for LLM:\n\n", prompt)
            response_text = get_llm_response_text(
                model_name=model_name,
                prompt=prompt
            )
            
            response_text = response_text.split("</think>")[1].strip() if "</think>" in response_text else response_text.strip()
            print("LLM response:\n\n",  response_text)
            #print("LLM response:\n\n",  response_text)
            
            selected_action_name = self.filter_action_from_response(response_text)
            #print("Selected action from LLM:", selected_action_name)
            if selected_action_name is None:
                #print("No action matched in response, defaulting to first action.")
                
                selected_action_name = actions[0]  # Default to the first action if none matched
                if current_turn == 0:
                    self.number_of_faulty_outputs1 += 1
                else:
                    self.number_of_faulty_outputs2 += 1
                
            action_index = self.action_mapper.action_name_to_action_index(selected_action_name)
            self.state_action_pairs[prompt] = action_index
        else:
            #print("Using cached response for prompt.")
            action_index = self.state_action_pairs[prompt]

        return action_index


    def filter_action_from_response(self, response_text: str):
        print(self.action_mapper.actions)
        for action in self.action_mapper.actions:
            if action.lower() in response_text.lower():
                #print("Matched action:", action)
                return action
        return None

    def save(self):
        """
        Saves the RL agent in the MLFlow experiment.
        """
        pass

    def get_hyperparameters(self):
        """
        Get the RL agent hyperparameters
        """
        pass


    def load_env(self, env):
        """
        Further loading, when the environment is loaded.
        """
        self.action_mapper = env.action_mapper
        self.state_mapper = env.storm_bridge.state_mapper
        self.env_name = env.env_name
        # Read file
        with open(env.storm_bridge.path, 'r') as file:
            # Read line by line
            self.env_description = file.read().splitlines()[0][6:]  # Skip "//" at the beginning
            
