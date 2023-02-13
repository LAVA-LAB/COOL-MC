class Interpreter:

    def __init__(self, config):
        self.config = config

    def interpret(self, env, rl_agent, model_checking_info):
        raise NotImplementedError()


