class Interpreter:

    def __init__(self, config):
        self.config = config

    def interpret(self, env, m_project, model_checking_info):
        raise NotImplementedError()


