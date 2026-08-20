"""
Safe meta-RL via dual-method-based policy adaptation -- the method of the paper.

``task_specific_update`` performs the constrained inner (task) step; the
Lagrangian weight on the cost is controlled by ``--meta-lambda``.
``meta_update`` differentiates through that inner step to update the meta
policy while keeping the constraint satisfied.
"""

from torch.optim import LBFGS
import cvxpy as cp
import numpy as np
import multiprocessing
import math
import time
import csv
import os

from utils.tools import *
from utils.torch import *
from utils.replay_memory import Memory
from utils.argument_parsing import parse_all_arguments
from utils.replay_memory import Memory
from core.common import estimate_advantages, estimate_constraint_value

import copy
from algos.trpo import *


#summarizing using tensorboard
from torch.utils.tensorboard import SummaryWriter

def collect_trajectory(pid, queue, env, env_parameter, env_name, policy, 
                       mean_action, running_state, min_batch_size, horizon, seed):
    torch.randn(pid)
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    env_total_reward = 0
    env_total_cost = 0
    num_episodes = 0
    reward_episode_list = []
    env_reward_episode_list = []

    """
    #randomize environment seed
    seed = np.random.randint(1,1000)
    env.seed(seed)
    torch.manual_seed(seed)
    """
    while num_steps < min_batch_size:
        state, info = env.reset(seed=seed)
        if running_state is not None:
            state = running_state(state)

        reward_episode = 0
        env_reward_episode = 0
        env_cost_episode = 0
        reward_episode_list_1 = []
        env_reward_episode_list_1 = []
        
        for t in range(horizon):
            state_var = tensor(state).unsqueeze(0)

            with torch.no_grad():
                if mean_action:
                    action = policy(state_var)[0][0].numpy()
                else:
                    # Stochastic action
                    action = policy.select_action(state_var)[0].numpy()
    

            action = int(action) if policy.is_disc_action else action.astype(np.float64)
            next_state, reward1, done, truncated, info = env.step(action)
            reward, cost = compute_task_reward_cost(env_name, env_parameter, reward1, info)

            env_reward_episode += reward
            env_cost_episode += cost
            env_reward_episode_list_1.append(reward)

            if running_state is not None:
                next_state = running_state(next_state)

            mask = 0 if done else 1

            memory.push(state, action, mask, next_state, reward, cost)

            if done or truncated:
                break

            state = next_state

        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        env_reward_episode_list.append(env_reward_episode)
        env_total_reward += env_reward_episode
        env_total_cost += env_cost_episode
        min_reward = min(min_reward, env_reward_episode)
        max_reward = max(max_reward, env_reward_episode)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['env_total_reward'] = env_total_reward
    log['env_avg_reward'] = env_total_reward / num_episodes
    log['env_total_cost'] = env_total_cost
    log['env_avg_cost'] = env_total_cost / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    log['env_reward_ep_list'] = env_reward_episode_list

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log
    

def merge_log(log_list):
    log = dict()
    total_rewards_episodes = []

    # merge env reward 
    log['env_total_reward'] = sum([x['env_total_reward'] for x in log_list])
    log['env_total_cost'] = sum([x['env_total_cost'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    log['env_avg_reward'] = log['env_total_reward'] / log['num_episodes']
    log['env_avg_cost'] = log['env_total_cost'] / log['num_episodes']
    for x in log_list:
        b = x['env_reward_ep_list']
        total_rewards_episodes += b
    log['total_rewards_episodes'] = total_rewards_episodes

    """std deviation of env rewards in one iteration"""
    reward_episode_list_array = np.array(total_rewards_episodes) - log['env_avg_reward']
    reward_episode_list_array = np.square(reward_episode_list_array)
    reward_episode_list_sum = np.sum(reward_episode_list_array)
    reward_episode_list_variance = reward_episode_list_sum / log['num_episodes']
    reward_episode_list_std = np.sqrt(reward_episode_list_variance)
    log['std_reward']  = reward_episode_list_std
    
    return log



class SafeMeta:
    def __init__(self, env, policy_net, value_net, cost_net, args, dtype, 
                 device, mean_action=False, running_state=None, num_threads=1):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.running_state = running_state
        self.mean_action = mean_action

        self.args = args

        self.min_batch = args.min_batch_size
        self.num_threads = args.num_threads

        self.dtype = dtype
        self.device = device

        self.meta_policy = policy_net
        self.local_policy = copy.deepcopy(policy_net)
        self.value_net = value_net
        self.cost_net = cost_net

        self.param_size = sum(p.numel() for p in self.meta_policy.parameters())

        self.lenth_constant=0.0
        for i in range(self.args.time_horizon):
            self.lenth_constant += self.args.gamma**i
        
    def collect_samples(self, env, env_parameter, env_name, policy, seed):
        t_start = time.time()
        to_device(torch.device('cpu'), policy)  
        thread_batch_size = int(math.floor(self.min_batch / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads-1):
            worker_args = (i+1, queue, env, env_parameter, env_name, policy, self.mean_action, 
                            self.running_state, thread_batch_size, self.args.time_horizon, seed)
            workers.append(multiprocessing.Process(target=collect_trajectory, args=worker_args))
        for worker in workers:
            worker.start()
        #for worker in workers:
        #    worker.join()

        memory, log = collect_trajectory(0, None, env, env_parameter, env_name,  policy,
                                         self.mean_action, self.running_state, thread_batch_size, 
                                         self.args.time_horizon, seed)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        to_device(self.device, policy)
        t_end = time.time()
        log['sample_time'] = t_end - t_start

        return memory, log
    
    def task_specific_update(self, batch, batch1,policy=None):
        
        """
        RETURN: gradient by finding loss and etc.
        """
        states = torch.from_numpy(np.stack(batch.state)[:self.args.max_batch_size]).to(self.dtype).to(self.device) #[:args.batch_size]
        costs = torch.from_numpy(np.stack(batch.cost)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
        actions = torch.from_numpy(np.stack(batch.action)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(np.stack(batch.reward)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
        masks = torch.from_numpy(np.stack(batch.mask)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
        
        """Update Critic"""
        for i in range(5): 

            r_optim = torch.optim.LBFGS(self.value_net.parameters(), lr=0.03, max_iter=30)
            c_optim = torch.optim.LBFGS(self.cost_net.parameters(), lr=0.03, max_iter=30)
            #r_optim = torch.optim.Adam(self.value_net.parameters(), lr=0.001)
            #c_optim = torch.optim.Adam(self.cost_net.parameters(), lr=0.001)

            get_value_loss = torch.nn.MSELoss()

            with torch.no_grad():
                reward_values = self.value_net(states)

            with torch.no_grad():
                cost_values = self.cost_net(states)

            _, returns = estimate_advantages(rewards, masks, reward_values, self.args.gamma, self.args.tau, self.device)
            _, cost_returns = estimate_advantages(costs, masks, cost_values, self.args.gamma, self.args.tau, self.device)
            #J_value = estimate_constraint_value(rewards, masks, self.args.gamma, self.device)
            #constraint_value = estimate_constraint_value(costs, masks, self.args.gamma, self.device)

            def r_closure():
                r_optim.zero_grad()
                r_pred = self.value_net(states)
                v_loss = get_value_loss(r_pred, returns)
                for param in self.value_net.parameters():
                    v_loss += param.pow(2).sum() * self.args.l2_reg
                v_loss.backward()
                return v_loss

            def c_closure():
                c_optim.zero_grad()
                c_pred = self.cost_net(states)
                c_loss = get_value_loss(c_pred, cost_returns)
                for param in self.cost_net.parameters():
                    c_loss += param.pow(2).sum() * self.args.l2_reg
                c_loss.backward()
                return c_loss
            
            r_optim.step(r_closure)
            c_optim.step(c_closure)

        '''Update Advantage'''

        states = torch.from_numpy(np.stack(batch1.state)[:self.args.max_batch_size]).to(self.dtype).to(self.device) #[:args.batch_size]
        costs = torch.from_numpy(np.stack(batch1.cost)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
        actions = torch.from_numpy(np.stack(batch1.action)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(np.stack(batch1.reward)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
        masks = torch.from_numpy(np.stack(batch1.mask)[:self.args.max_batch_size]).to(self.dtype).to(self.device)

        with torch.no_grad():
            reward_values = self.value_net(states)

        with torch.no_grad():
            cost_values = self.cost_net(states)

        reward_advantages, returns = estimate_advantages(rewards, masks, reward_values, self.args.gamma, self.args.tau, self.device)
        cost_advantages, cost_returns = estimate_advantages(costs, masks, cost_values, self.args.gamma, self.args.tau, self.device)
        J_value = estimate_constraint_value(rewards, masks, self.args.gamma, self.device)
        constraint_value = estimate_constraint_value(costs, masks, self.args.gamma, self.device)

        print("constraint value before: ", constraint_value[0].numpy())
        print("J_value before: ", J_value[0].numpy())

        bbbb= ( -constraint_value[0] + self.args.max_constraint)/self.lenth_constant
        
        """update policy"""
        if policy is None:
            this_local_policy = copy.deepcopy(self.meta_policy)
        else:
            this_local_policy = copy.deepcopy(policy)
        
        print(reward_advantages.std())
        print(cost_advantages.std())
        
        fixed_log_probs = this_local_policy.get_log_prob(states, actions).detach().clone().data

        #one_step_trpo_constraint_2(this_local_policy, self.args.meta_lambda, states, actions, fixed_log_probs, cost_advantages, reward_advantages, bbbb)
        
        def get_loss():
            
            log_prob = this_local_policy.get_log_prob(states, actions)
            aaaa=torch.exp(log_prob - Variable(fixed_log_probs))
            action_loss = -reward_advantages *  torch.special.expit(2.0*aaaa-2.0)*2 
            #action_loss = -Variable(q_values) * aaaa
            return action_loss.mean() 
        
        def get_constraint():
            
            log_prob = this_local_policy.get_log_prob(states, actions)
            aaaa=torch.exp(log_prob - Variable(fixed_log_probs))
            action_loss = Variable(cost_advantages) *  torch.special.expit(2.0*aaaa-2.0)*2 
            #action_loss = Variable(q_values) * aaaa
            return action_loss.mean()  

        mean1, log_std1, std1 = this_local_policy.forward(states)
        mean0 = mean1.detach().clone().data
        log_std0 = log_std1.clone().detach().data
        std0 = std1.clone().detach().data
        
        def get_kl():
            mean1, log_std1, std1 = this_local_policy.forward(states)
            kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
            return kl.sum(1, keepdim=True)
        
        one_step_trpo_constraint(this_local_policy, get_loss, get_constraint, bbbb, get_kl,self.args.meta_lambda,lower_opt="Adam") 

        return this_local_policy,bbbb

    def meta_update(self, batch_list, batch_list1,constraint_smaller_than_max=True):
        advantages_list=[]
        costs_list=[]
        states_list=[]
        action_list=[]
        average_constraint = 0.0

        for batch, batch1 in zip(batch_list,batch_list1):
            states = torch.from_numpy(np.stack(batch.state)[:self.args.max_batch_size]).to(self.dtype).to(self.device) #[:args.batch_size]
            costs = torch.from_numpy(np.stack(batch.cost)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
            actions = torch.from_numpy(np.stack(batch.action)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
            rewards = torch.from_numpy(np.stack(batch.reward)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
            masks = torch.from_numpy(np.stack(batch.mask)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
            
            """Update Critic"""
            for i in range(5):

                r_optim = torch.optim.LBFGS(self.value_net.parameters(), lr=0.1, max_iter=20)
                c_optim = torch.optim.LBFGS(self.cost_net.parameters(), lr=0.1, max_iter=20)
                #r_optim = torch.optim.Adam(self.value_net.parameters(), lr=0.001)
                #c_optim = torch.optim.Adam(self.cost_net.parameters(), lr=0.001)

                get_value_loss = torch.nn.MSELoss()

                with torch.no_grad():
                    reward_values = self.value_net(states)

                with torch.no_grad():
                    cost_values = self.cost_net(states)

                _, returns = estimate_advantages(rewards, masks, reward_values, self.args.gamma, self.args.tau, self.device)
                _, cost_returns = estimate_advantages(costs, masks, cost_values, self.args.gamma, self.args.tau, self.device)
                #J_value = estimate_constraint_value(rewards, masks, self.args.gamma, self.device)
                #constraint_value = estimate_constraint_value(costs, masks, self.args.gamma, self.device)

                def r_closure():
                    r_optim.zero_grad()
                    r_pred = self.value_net(states)
                    v_loss = get_value_loss(r_pred, returns)
                    for param in self.value_net.parameters():
                        v_loss += param.pow(2).sum() * self.args.l2_reg
                    v_loss.backward()
                    return v_loss

                def c_closure():
                    c_optim.zero_grad()
                    c_pred = self.cost_net(states)
                    c_loss = get_value_loss(c_pred, cost_returns)
                    for param in self.cost_net.parameters():
                        c_loss += param.pow(2).sum() * self.args.l2_reg
                    c_loss.backward()
                    return c_loss
                
                r_optim.step(r_closure)
                c_optim.step(c_closure)

            '''Update Advantage'''

            states = torch.from_numpy(np.stack(batch1.state)[:self.args.max_batch_size]).to(self.dtype).to(self.device) #[:args.batch_size]
            costs = torch.from_numpy(np.stack(batch1.cost)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
            actions = torch.from_numpy(np.stack(batch1.action)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
            rewards = torch.from_numpy(np.stack(batch1.reward)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
            masks = torch.from_numpy(np.stack(batch1.mask)[:self.args.max_batch_size]).to(self.dtype).to(self.device)

            with torch.no_grad():
                reward_values = self.value_net(states)

            with torch.no_grad():
                cost_values = self.cost_net(states)

            reward_advantages, returns = estimate_advantages(rewards, masks, reward_values, self.args.gamma, self.args.tau, self.device)
            cost_advantages, cost_returns = estimate_advantages(costs, masks, cost_values, self.args.gamma, self.args.tau, self.device)
            J_value = estimate_constraint_value(rewards, masks, self.args.gamma, self.device)
            constraint_value = estimate_constraint_value(costs, masks, self.args.gamma, self.device)

            print("constraint value after: ", constraint_value[0].numpy())
            print("J_value after: ", J_value[0].numpy())
            average_constraint += constraint_value[0].numpy()

            advantages_list.append(reward_advantages)
            states_list.append(states)
            action_list.append(actions)
            costs_list.append(cost_advantages) 
        
        """define the loss function for TRPO"""
        fixed_log_probs_list=[]
        for states, actions in zip(states_list, action_list):
            fixed_log_probs1 = self.meta_policy.get_log_prob(states, actions).detach().clone().data
            fixed_log_probs_list.append(fixed_log_probs1)

        def get_loss(volatile=False):
            overall_loss=0.0
            for states, actions, reward_advantages,fixed_log_probs11 in zip(states_list, action_list, advantages_list,fixed_log_probs_list):
                log_probs = self.meta_policy.get_log_prob(states, actions) 
                action_loss = -reward_advantages * torch.exp(log_probs - fixed_log_probs11)
                overall_loss+=action_loss.mean()/self.args.env_num
            return overall_loss
        
        def get_constraint_loss(volatile=False):
            overall_constraint_loss=0.0
            for states, actions, cost_advantages,fixed_log_probs11 in zip(states_list, action_list, costs_list,fixed_log_probs_list):
                log_probs = self.meta_policy.get_log_prob(states, actions) 
                action_constraint_loss = cost_advantages * torch.exp(log_probs - fixed_log_probs11)
                overall_constraint_loss+=action_constraint_loss.mean()/self.args.env_num
            return overall_constraint_loss

        states_all=torch.cat(states_list)
        mean1, log_std1, std1 = self.meta_policy.forward(states_all)
        mean0 = mean1.detach().clone().data
        log_std0 = log_std1.clone().detach().data
        std0 = std1.clone().detach().data

        def get_kl():
            mean1, log_std1, std1 = self.meta_policy.forward(states_all)
            kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
            return kl.sum(1, keepdim=True)
        
        if constraint_smaller_than_max:
            print("constraint is less than max constraint")
            trpo_step(self.meta_policy, get_loss, get_kl, self.args.max_kl, self.args.damping)
        else:
            print("constraint is greater than max constraint")
            trpo_step(self.meta_policy, get_constraint_loss, get_kl, self.args.max_kl, self.args.damping)

        return 
            

    def meta_test(self, writer,save_info_obj):
        print("Start meta-testing")

        env_parameter_list = create_meta_test_task_list(self.args)

        sample_time_list=np.zeros([len(env_parameter_list),self.args.meta_iter_num])
        update_time_list=np.zeros([len(env_parameter_list),self.args.meta_iter_num])
        meta_avg_cost1_list=np.zeros([len(env_parameter_list),self.args.meta_iter_num])
        meta_avg_cost2_list=np.zeros([len(env_parameter_list),self.args.meta_iter_num])
        reward_list1=np.zeros([len(env_parameter_list),self.args.meta_iter_num])
        reward_list2=np.zeros([len(env_parameter_list),self.args.meta_iter_num])
        
        for env_parameter_index in range(len(env_parameter_list)):
            meta_avg_cost = []
            self.local_policy=copy.deepcopy(self.meta_policy)

            for m_iter in range(self.args.meta_iter_num):
                sample_time = 0
                seed = deterministic_rollout_seed(self.args.seed, env_parameter_index, m_iter, 0)
                batch1, log = self.collect_samples(self.env, env_parameter_list[env_parameter_index], self.args.env_name, self.local_policy, seed)
                batch2, log2 = self.collect_samples(
                    self.env,
                    env_parameter_list[env_parameter_index],
                    self.args.env_name,
                    self.local_policy,
                    deterministic_rollout_seed(self.args.seed, env_parameter_index, m_iter, 1),
                )
                
                sample_time += log['sample_time']+log2['sample_time']

                t1 = time.time()
                self.local_policy,_=self.task_specific_update(batch1.sample(),batch2.sample(),self.local_policy)
                t2 = time.time()

                eval_seed = deterministic_rollout_seed(self.args.seed, env_parameter_index, m_iter, 2)
                eval_reward, eval_cost = evaluate_policy_on_task(
                    self.env,
                    self.local_policy,
                    self.running_state,
                    self.args.env_name,
                    env_parameter_list[env_parameter_index],
                    self.args.gamma,
                    self.args.time_horizon,
                    num_trajectories=100,
                    base_seed=eval_seed,
                )

                meta_avg_cost.append(eval_cost)

                print('{}\tT_sample {:.4f}  T_update {:.4f}\tC_avg/iter {:.2f}  Test_C_avg {:.2f}\tR_avg {:.2f}\tTest_R_avg {:.2f}'.format( 
                m_iter, sample_time, t2-t1, np.average(meta_avg_cost), eval_cost, log['env_avg_reward'], eval_reward))   

                sample_time_list[env_parameter_index,m_iter]=sample_time
                update_time_list[env_parameter_index,m_iter]=t2-t1
                meta_avg_cost1_list[env_parameter_index,m_iter]=np.average(meta_avg_cost)
                meta_avg_cost2_list[env_parameter_index,m_iter]=eval_cost
                reward_list1[env_parameter_index,m_iter]=log['env_avg_reward']
                reward_list2[env_parameter_index,m_iter]=eval_reward

                writer.add_scalar('meta_rewards', eval_reward, m_iter)
                writer.add_scalar('meta_costs', eval_cost, m_iter) 

                """clean up gpu memory"""
                torch.cuda.empty_cache()

        cvs_file=os.path.join(assets_dir(), save_info_obj.saving_path, 'test_log2.csv')

        sample_time_list=sample_time_list.mean(axis=0)
        update_time_list=update_time_list.mean(axis=0)
        meta_avg_cost1_list=meta_avg_cost1_list.mean(axis=0)
        meta_avg_cost2_list=meta_avg_cost2_list.mean(axis=0)
        reward_list1=reward_list1.mean(axis=0)
        reward_list2=reward_list2.mean(axis=0)

        for m_iter in range(self.args.meta_iter_num):
            with open(cvs_file, 'a+') as file:
                writer_csv = csv.writer(file)
                writer_csv.writerow([m_iter, sample_time_list[m_iter], update_time_list[m_iter], meta_avg_cost1_list[m_iter], meta_avg_cost2_list[m_iter], reward_list1[m_iter], reward_list2[m_iter]])

        writer.close()
        return 

    def train_SafeMeta(self, writer, save_info_obj):
        # lists for dumping plotting data for agent
        env_avg_reward = []
        env_avg_cost = []

        # lists for dumping plotting data for mean agent
        iter_for_best_avg_reward = None

        # for saving the best model
        best_avg_reward = -9999999

        if self.args.is_meta_test:
            meta_avg_cost = self.meta_test(writer,save_info_obj)
            return 
        
        if self.args.model_path is not None:
            total_iterations = self.args.max_iter_num - self.args.update_iter_num
            print('total iterations: ',self.args.max_iter_num,
               'updated iteration: ', self.args.update_iter_num,
               'remaining iteration: ', total_iterations)
        else: 
            self.args.update_iter_num = 0
        
        hline()
        print('Training has begun')
        hline()
        for i_iter in range(self.args.update_iter_num, self.args.max_iter_num):
            """Define meta parameters"""
            sample_time = 0
            update_time = 0

            avg_reward = 0
            avg_cost = 0
            
            batch_list = []
            batch_list1 = []

            batch_list_prev = []
            batch_list1_prev = []

            bbbbbbbb_average=0.0
            env_parameter_list=create_env_parameter_list(self.args)
            local_task_count = len(env_parameter_list)
            if local_task_count == 0:
                raise RuntimeError("No training tasks available. Check task sampling or cover set configuration.")

            # Collect samples
            for local_iter in range(local_task_count):
                t1 = time.time()
                memory, _ = self.collect_samples(self.env, env_parameter_list[local_iter],self.args.env_name, self.meta_policy, seed=self.args.seed)
                memory2, _ = self.collect_samples(self.env, env_parameter_list[local_iter],self.args.env_name, self.meta_policy, seed=self.args.seed+1)
                t2 = time.time()

                sample_time += t2 - t1
                this_local_policy,bbbbbbbb = self.task_specific_update(memory.sample(),memory2.sample())
                bbbbbbbb_average+=bbbbbbbb/local_task_count
                batch_list_prev.append(memory.sample())
                batch_list1_prev.append(memory2.sample())
                t3 = time.time(); update_time += t3 - t2
                print("----------------------")

                t1 = time.time()
                local_memory, _ = self.collect_samples(self.env, env_parameter_list[local_iter],self.args.env_name, this_local_policy, seed=local_iter)
                local_memory2, _ = self.collect_samples(self.env, env_parameter_list[local_iter],self.args.env_name, this_local_policy, seed=local_iter+1)
                t2 = time.time(); sample_time += t2 - t1
                batch_list.append(local_memory.sample())
                batch_list1.append(local_memory2.sample())

                # compute avg reward and cost for log
                rewards =  np.stack(local_memory.sample().reward)
                costs =  torch.from_numpy(np.stack(local_memory.sample().cost)).to(self.dtype).to(self.device)
                masks = torch.from_numpy(np.stack(local_memory.sample().mask)).to(self.dtype).to(self.device)
                
                num_episode = self.args.min_batch_size/self.args.time_horizon
                avg_reward += (np.sum(rewards) / num_episode) / local_task_count
                avg_cost += estimate_constraint_value(costs, masks, self.args.gamma, self.device)[0].to(torch.device('cpu')) / local_task_count

            #if bbbbbbbb_average<0.0:
            recent_avg_cost = float(np.mean(env_avg_cost[-10:])) if len(env_avg_cost) > 0 else float(avg_cost)
            if recent_avg_cost > self.args.max_constraint and avg_cost > self.args.max_constraint:
                print("constraint is greater than max constraint")
                constraint_smaller_than_max=False
                self.meta_update(batch_list_prev, batch_list1_prev,constraint_smaller_than_max)
                self.meta_update(batch_list, batch_list1,True)
            else:
                print("constraint is less than max constraint")
                constraint_smaller_than_max=True
                self.meta_update(batch_list, batch_list1,constraint_smaller_than_max)
            print("----------------------")
        
            env_avg_reward.append(avg_reward)
            env_avg_cost.append(avg_cost)

            # update tensorboard summaries
            writer.add_scalar('rewards', avg_reward, i_iter)  
            writer.add_scalar('costs', avg_cost, i_iter)  

            print('{}  T_sample {:.4f}  T_update {:.4f}  C_avg {:.2f}  Test_C_avg {:.2f}  R_avg {:.2f}  Test_R_avg {:.2f}'.format( 
            i_iter, sample_time, update_time, np.mean(env_avg_cost[-10:]), avg_cost, np.mean(env_avg_reward[-10:]), avg_reward))   
            cvs_file=os.path.join(assets_dir(), save_info_obj.saving_path, 'training_log.csv')

            with open(cvs_file, 'a+') as file:
                writer_csv = csv.writer(file)
                writer_csv.writerow([i_iter, sample_time, update_time, np.mean(env_avg_cost[-10:]), avg_cost.numpy(), np.mean(env_avg_reward[-10:]), avg_reward])

            # save the best model
            if avg_reward >= best_avg_reward and avg_cost.numpy()<=self.args.max_constraint:
                print('Saving new best model !!!!')
                to_device(torch.device('cpu'), self.meta_policy, self.value_net, self.cost_net)
                save_info_obj.save_models(self.meta_policy, self.value_net, self.cost_net, self.running_state, self.args)
                to_device(self.device, self.meta_policy, self.value_net, self.cost_net)
                best_avg_reward = avg_reward
                iter_for_best_avg_reward = i_iter+1

            # save some intermediate models to sample trajectories from
            if self.args.save_intermediate_model > 0 and (i_iter+1) % self.args.save_intermediate_model == 0:
                to_device(torch.device('cpu'), self.meta_policy, self.value_net)
                save_info_obj.save_intermediate_models(self.meta_policy, self.value_net, self.cost_net, self.running_state, self.args, i_iter)
                to_device(self.device, self.meta_policy, self.value_net)

            """clean up gpu memory"""
            torch.cuda.empty_cache()

        if not os.path.exists(save_info_obj.model_saving_path):
            print('Saving final model (no best-model checkpoint found).')
            to_device(torch.device('cpu'), self.meta_policy, self.value_net, self.cost_net)
            save_info_obj.save_models(self.meta_policy, self.value_net, self.cost_net, self.running_state, self.args)
            to_device(self.device, self.meta_policy, self.value_net, self.cost_net)

        print('Saving final model snapshot (model_last.p).')
        to_device(torch.device('cpu'), self.meta_policy, self.value_net, self.cost_net)
        save_info_obj.save_models_last(self.meta_policy, self.value_net, self.cost_net, self.running_state, self.args)
        to_device(self.device, self.meta_policy, self.value_net, self.cost_net)

        print(iter_for_best_avg_reward, 'Best eval R:', best_avg_reward)

if __name__ == '__main__':
    print('SafeMeta')