"""
Training / meta-testing entry point.

Selects one of four algorithms via ``--algo-name`` (SafeMeta, MAML_constraint,
CPOMeta, CPO), builds the policy / value / cost networks, and dispatches to the
matching ``train_*`` routine. ``--is-meta-test`` switches between meta-training
from scratch and adapting a checkpoint loaded from ``--model-path``.

See ``utils/argument_parsing.py`` for the full flag list.
"""

import argparse
import os
import sys
import pickle

# Call utilities
from utils import *

# Call models
from models.continuous_policy import Policy
from models.critic import Value
from models.discrete_policy import DiscretePolicy

# Call algorithms
from algos.CPO import CPO
from algos.CPOMeta import CPOMeta
from algos.SafeMeta import SafeMeta
from algos.MAML_constraint import MAML_constraint

# Call tensorboard for logging
from torch.utils.tensorboard import SummaryWriter

# Returns the current local date
from datetime import date

def main_loop():
    today = date.today()
    print("Today date is: ", today)

    # Parse arguments 
    args = parse_all_arguments()
    print("Arguments: ",args)

    """Data type and compute device"""
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print('using gpu')
        torch.cuda.set_device(args.gpu_index)

    """environment"""
    env,env_parameter_list = create_sigle_envs(args)    

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print('\n')
    print('state dim: ', state_dim)
    print('action dim: ', action_dim)
    print('\n')

    is_disc_action = len(env.action_space.shape) == 0
    running_state = ZFilter((state_dim,), clip=5)

    """seeding"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    """create all the paths to save learned models/data"""
    save_info_obj = save_info(assets_dir(), args.algo_name, args.algo_name, args.env_name) #model saving object
    save_info_obj.create_all_paths() # create all paths
    writer = SummaryWriter(os.path.join(assets_dir(), save_info_obj.saving_path, 'runs/')) #tensorboard summary
    print('Saving path: ', save_info_obj.saving_path)

    """define actor and critic"""
    model_file = None
    if args.model_path is not None:
        model_file = os.path.join(args.model_path, 'model.p')
        if not os.path.exists(model_file):
            model_last = os.path.join(args.model_path, 'model_last.p')
            if os.path.exists(model_last):
                model_file = model_last

    if model_file is None or not os.path.exists(model_file):
        if model_file is not None:
            print(f"Model file not found at {model_file}. Starting from scratch.")
        if is_disc_action:
            policy_net = DiscretePolicy(state_dim, action_dim)
        else:
            policy_net = Policy(state_dim, action_dim, log_std=args.log_std)
        value_net = Value(state_dim)
        cost_net = Value(state_dim)
    else:
        print('TRAINING FROM PREVIOUS PARAMETERS. . .', args)
        cli_args = args
        policy_net, value_net, cost_net, running_state, prev_args = pickle.load(open(model_file, "rb"))
        for key, value in vars(cli_args).items():
            setattr(prev_args, key, value)
        args = prev_args

    policy_net.to(device)
    value_net.to(device)
    cost_net.to(device)
    if args.is_meta_test:
        print('meta testing')
        if args.algo_name == 'CPOMeta':
            algo = CPOMeta(env, policy_net, value_net, cost_net, args, dtype, device,
                        running_state=running_state, num_threads=args.num_threads)
            algo.train_CPOMeta(writer, save_info_obj)
        elif args.algo_name == 'CPO':
            algo = CPO([env], policy_net, value_net, cost_net, args, dtype, device,
                        running_state=running_state, num_threads=args.num_threads)
            algo.train_CPO(writer, save_info_obj)
        elif args.algo_name == 'SafeMeta':
            algo = SafeMeta(env, policy_net, value_net, cost_net, args, dtype, device,
                            running_state=running_state, num_threads=args.num_threads)
            algo.train_SafeMeta(writer, save_info_obj)
        elif args.algo_name == 'MAML_constraint':
            algo = MAML_constraint(env, policy_net, value_net, cost_net, args, dtype, device,
                                   running_state=running_state, num_threads=args.num_threads)
            algo.train_MAML_constraint(writer, save_info_obj)
            
    else:
        """create agent"""
        if args.algo_name == 'CPO':
            algo = CPO([env], policy_net, value_net, cost_net, args, dtype, device,
                        running_state=running_state, num_threads=args.num_threads)
            algo.train_CPO(writer, save_info_obj)

        elif args.algo_name == 'CPOMeta':
            algo = CPOMeta(env, policy_net, value_net, cost_net, args, dtype, device,
                            running_state=running_state, num_threads=args.num_threads)
            algo.train_CPOMeta(writer, save_info_obj)

        elif args.algo_name == 'SafeMeta':
            algo = SafeMeta(env, policy_net, value_net, cost_net, args, dtype, device,
                            running_state=running_state, num_threads=args.num_threads)
            algo.train_SafeMeta(writer, save_info_obj)
        elif args.algo_name == 'MAML_constraint':
            algo = MAML_constraint(env, policy_net, value_net, cost_net, args, dtype, device,
                                   running_state=running_state, num_threads=args.num_threads)
            algo.train_MAML_constraint(writer, save_info_obj)

if __name__ == '__main__':
    main_loop()


