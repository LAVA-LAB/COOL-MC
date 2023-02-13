import mlflow
from common.utilities.helper import *


if __name__ == '__main__':
    print("=======================")
    args = get_arguments()
    set_random_seed(args['seed'])

    if args['task'] == 'safe_training':
        mlflow.run(
            "safe_training",
            env_manager="local",
            parameters=dict(args)
        )
    elif args['task'] == 'rl_model_checking':
        mlflow.run(
            "rl_model_checking",
            env_manager="local",
            parameters=dict(args)
        )
