"""
Command-line flags for ``main.py``.

Some flags are algorithm-specific: the parser first peeks at ``--algo-name``,
then registers the extra arguments that only CPO/CPOMeta (``--max-kl``,
``--anneal``, ...) or SafeMeta/MAML_constraint (``--meta-lambda``, ...) accept.
"""

import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")

def parse_all_arguments():
        
    parser = argparse.ArgumentParser() #description='Running {}'.format(algo_name))
    
    # Basic agruments 
    parser.add_argument('--algo-name', default="SafeMeta", metavar='G', #MAML_constraint, SafeMeta, CPOMeta, CPO
                       help='algorithm name')

    parser.add_argument('--env-name', default="Hopper", metavar='G',  #HalfCheetah, Swimmer, Humanoid, Hopper
                        help='name of the environment to run')
    parser.add_argument('--env-num', type=int, default=10, metavar='G',
                        help='number of environments')
    
    # update with prev parameters 
    parser.add_argument('--model-path', metavar='G', default="./assets/learned_models/SafeMeta/2024-05-11-exp-SafeMeta-Hopper/",
                        help='path of pre-trained model')
    parser.add_argument('--is-meta-test', type=str2bool, nargs='?', const=True, default=True,
                        help='path of pre-trained model')
    parser.add_argument('--update-iter-num', metavar='N', type=int, default=0, 
                        help='path of pre-trained model') 

    # Learning rates and regularizations
    parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                        help='log std for the policy (default: -0.0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                        help='gae (default: 0.95)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization of value function (default: 1e-3)')
    parser.add_argument('--learning-rate', type=float, default=1e-4, metavar='G',
                        help='gae (default: 3e-4)')
    
    # GPU index, multi-threading and seeding
    parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
    parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                        help='number of threads for agent (default: 4)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 1)')
    
    # batch size and iteration number
    parser.add_argument('--min-batch-size', type=int, default=8000, metavar='N',
                        help='minimal batch size per PPO update (default: 2048)')
    parser.add_argument('--max-batch-size', type=int, default=8000, metavar='N',
                        help='maximum batch size per PPO update (default: 2000)')
    parser.add_argument('--time-horizon', type=int, default=200, metavar='N',
                        help='maximum batch size per PPO update (default: 2000)')
    parser.add_argument('--max-iter-num', type=int, default=300, metavar='N',
                        help='maximal number of main iterations (default: 300)')
    parser.add_argument('--meta-iter-num', type=int, default=20, metavar='N',
                        help='maximal number of main iterations (default: 100)') 
    
    # logging and saving models
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')
    #parser.add_argument('--save-model-interval', type=int, default=50, metavar='N',
    #                    help="interval between saving model (default: 0, means don't save)")
    parser.add_argument('--save-intermediate-model', type=int, default=100, metavar='N',
                        help="intermediate model saving interval (default: 0, means don't save)")
       
    preliminary_args, _ = parser.parse_known_args()
    print(preliminary_args.algo_name)
    parser.add_argument('--max-constraint', type=float, default=5, metavar='G',  #halfcheetah: 10, swimmer: 5, humanoid: 20, hopper: 5
                        help='max constraint value (default: 1e-2)')
    parser.add_argument('--use-cover-set-tasks', type=str2bool, nargs='?', const=True, default=False,
                        help='Use fixed Hopper task set from cover_set file instead of sampling task distribution')
    parser.add_argument('--cover-set-path', type=str, default='assets/cover_set.json',
                        help='Path to JSON list of fixed Hopper goal velocities')
    
    if preliminary_args.algo_name == "CPOMeta" or preliminary_args.algo_name =="CPO":
        parser.add_argument('--exp-num', default="1", metavar='G',
                        help='Experiment number for today (default: 1)')
        parser.add_argument('--exp-name', default="Exp-1", metavar='G',
                        help='Experiment name')
        parser.add_argument('--local-num', type=int, default=1, metavar='N',
                        help='maximal number of main iterations (default: 3)')
        parser.add_argument('--max-kl', type=float, default=1e-3, metavar='G',
                        help='max kl value (default: 1e-2)')
        parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
                        help='damping (default: 1e-2)')
        parser.add_argument('--annealing_factor', type=float, default=1e-2, metavar='G',
                        help='annealing factor of constraint (default: 1e-2)')
        parser.add_argument('--anneal', type=str2bool, nargs='?', const=True, default=True,
                        help='Should the constraint be annealed or not')
        parser.add_argument('--grad-norm', type=str2bool, nargs='?', const=True, default=True,
                        help='Should the norm of policy gradient be taken (default: False)')
    elif preliminary_args.algo_name == "SafeMeta" or preliminary_args.algo_name == "MAML_constraint":
        parser.add_argument('--max-kl', type=float, default=1e-4, metavar='G',
                        help='max kl value for TRPO (default: 1e-2)')
        parser.add_argument('--damping', type=float, default=0e-2, metavar='G',
                        help='damping (default: 1e-2)')
        parser.add_argument('--meta-lambda', type=float, default=1.0, metavar='G', #halfcheetah: 1.0, swimmer: 0.2, humanoid: 5.0, hopper: 1.0
                        help='meta-lambda (default: 0.5)')
        
    
    return parser.parse_args()

if __name__ == "__main__":
    print("Parsing arguments. . .")