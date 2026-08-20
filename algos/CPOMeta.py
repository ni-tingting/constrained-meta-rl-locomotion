"""
CPO-based meta-RL baseline.

Runs a CPO step per sampled task, then aggregates the task steps into a meta
update, projecting the result back into the constraint-feasible set
(``project_step``).
"""

from torch.optim import LBFGS
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
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


def cpo_problem(param_size, dtype):
    '''
    CPO implementation using DCO (Differentiable Convex Optimization)
    '''
    x = cp.Variable(param_size) # update we are to find
    g = cp.Parameter(param_size) # reward advantage gradient

    max_kl = cp.Parameter() 

    a = cp.Parameter(param_size) # cost advantage gradient
    max_constraint = cp.Parameter()

    objective = cp.Minimize(g.T @ x)
    constraints = [0.5 * cp.sum_squares(x) <= max_kl,
                   max_constraint + a.T @ x <= 0]

    problem = cp.Problem(objective, constraints)
    cvxpylayer = CvxpyLayer(problem, parameters=[g, max_kl, a, max_constraint], variables=[x])

    return cvxpylayer

def trpo_problem(param_size, dtype):
    '''
    TRPO implementation using DCO (Differentiable Convex Optimization)
    '''
    x = cp.Variable(param_size)

    g = cp.Parameter(param_size)
    max_kl = cp.Parameter()

    objective = cp.Minimize(g.T @ x)
    constraints = [0.5 * cp.sum_squares(x) <= max_kl]

    problem = cp.Problem(objective, constraints)
    cvxpylayer = CvxpyLayer(problem, parameters=[g, max_kl], variables=[x])

    return cvxpylayer

def projection(param_size, dtype):
    '''
    DCO implementation for projection onto the safety set
    In other words, projection step seeks nearest set that satisfy the constraint.
    '''
    prev_step = cp.Parameter(param_size)

    a = cp.Parameter(param_size)
    max_constraint = cp.Parameter()

    step = cp.Variable(param_size)

    objective = cp.Minimize(0.5 * cp.sum_squares(step - prev_step))
    constraint = [a.T @ step + max_constraint <= 0]

    problem = cp.Problem(objective, constraint)
    cvxpylayer = CvxpyLayer(problem, parameters=[a, max_constraint, prev_step], variables=[step])
    
    return cvxpylayer

class CPOMeta:
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
        #self.local_policy = self.meta_policy # Initialize with meta_policy
        self.local_policy=copy.deepcopy(self.meta_policy)
        self.value_net = value_net
        self.cost_net = cost_net

        self.param_size = sum(p.numel() for p in self.meta_policy.parameters())
        self.cpo_problem = cpo_problem(self.param_size, self.dtype)
        self.trpo_problem = trpo_problem(self.param_size, self.dtype)
        self.projection = projection(self.param_size, self.dtype)

    def line_search(self, model, reward_f, cost_f, x, fullstep, expected_reward_improve_full, expected_cost_improve_full, max_backtracks=30, accept_ratio=0.1):
        reward_fval = reward_f(True).item()
        cost_fval = cost_f(True).item()
        
        for stepfrac in [.5**x for x in range(max_backtracks)]:
            x_new = x + stepfrac * fullstep
            set_flat_params_to(model, x_new)

            reward_fval_new = reward_f(True).item()
            cost_fval_new = cost_f(True).item()

            actual_reward_improve = reward_fval - reward_fval_new  
            actual_cost_improve = cost_fval - cost_fval_new


            #expected_reward_improve = expected_reward_improve_full * stepfrac
            #expected_cost_improve = expected_cost_improve_full * stepfrac

            #print('actual: ', actual_reward_improve, 'expected: ', expected_reward_improve)
            #print('actual: ', actual_cost_improve, 'expected: ', expected_cost_improve)

            #r_ratio = actual_reward_improve / expected_reward_improve
            #c_ratio = actual_cost_improve / expected_cost_improve
            #if r_ratio > accept_ratio and c_ratio > accept_ratio:
            if torch.norm(stepfrac * fullstep) <= self.args.max_kl  and actual_reward_improve > 0 and actual_cost_improve > 0:
                print('step mean: ', abs(stepfrac * fullstep).mean())
                return True, x_new
        print('Line search Failed')    
        return False, x

    def meta_line_search(self, x, batch, fullstep, max_backtracks = 20, accept_ratio=0.1):
        states = torch.from_numpy(np.stack(batch.state)).to(self.dtype).to(self.device)
        costs = torch.from_numpy(np.stack(batch.cost)).to(self.dtype).to(self.device)
        actions = torch.from_numpy(np.stack(batch.action)).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(np.stack(batch.reward)).to(self.dtype).to(self.device)
        masks = torch.from_numpy(np.stack(batch.mask)).to(self.dtype).to(self.device)

        with torch.no_grad():
            values = self.value_net(states)
        
        with torch.no_grad():
            cost_values = self.cost_net(states)

        advantages, _ = estimate_advantages(rewards, masks, values, self.args.gamma, self.args.tau, self.device)
        cost_advantages, _ = estimate_advantages(costs, masks, cost_values, self.args.gamma, self.args.tau, self.device)
        #constraint_value = estimate_constraint_value(costs, masks, self.args.gamma, self.device)

        with torch.no_grad():
            fixed_log_probs = self.meta_policy.get_log_prob(states, actions)

        def get_reward_loss(volatile=False):
            with torch.set_grad_enabled(not volatile):
                log_probs = self.meta_policy.get_log_prob(states, actions)
                action_loss = -advantages * torch.exp(log_probs - fixed_log_probs)
                return action_loss.mean()
            
        def get_cost_loss(volatile=False):
            with torch.set_grad_enabled(not volatile):
                log_probs = self.meta_policy.get_log_prob(states, actions)
                cost_loss = cost_advantages * torch.exp(log_probs - fixed_log_probs)
                return cost_loss.mean()
            
        reward_loss = get_reward_loss()
        cost_loss = get_cost_loss()

        loss_reward_grad = torch.autograd.grad(reward_loss, self.meta_policy.parameters())
        grads = torch.cat([grad.view(-1) for grad in loss_reward_grad])
        g = grads.clone().detach()

        loss_cost_grad = torch.autograd.grad(cost_loss, self.meta_policy.parameters())
        grads = torch.cat([grad.view(-1) for grad in loss_cost_grad])
        a = grads.clone().detach()
        
        reward_fval = get_reward_loss(True).item()
        cost_fval = get_cost_loss(True).item()

        for stepfrac in [0.5**x for x in range(max_backtracks)]:
            x_new = x + stepfrac * fullstep
            set_flat_params_to(self.meta_policy, x_new)

            reward_fval_new = get_reward_loss(True).item()
            cost_fval_new = get_reward_loss(True).item()

            actual_reward_improve = reward_fval - reward_fval_new
            actual_cost_improve = cost_fval - cost_fval_new

            if torch.norm(stepfrac * fullstep) <= self.args.max_kl and actual_cost_improve > 0:
                return True, x_new, stepfrac * fullstep
            
        return False, x, torch.zeros(fullstep.numel()).to(self.device)
        

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

        memory, log = collect_trajectory(0, None, env, env_parameter, env_name, policy,
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
    
    def update(self, batch, meta_update=False, step_only=False):
        """
        RETURN: gradient by finding loss and etc.
        """
        states = torch.from_numpy(np.stack(batch.state)[:self.args.max_batch_size]).to(self.dtype).to(self.device) #[:args.batch_size]
        costs = torch.from_numpy(np.stack(batch.cost)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
        actions = torch.from_numpy(np.stack(batch.action)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(np.stack(batch.reward)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
        masks = torch.from_numpy(np.stack(batch.mask)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
        
        """Update Critic"""
        r_optim = torch.optim.LBFGS(self.value_net.parameters(), lr=0.1, max_iter=20)
        c_optim = torch.optim.LBFGS(self.cost_net.parameters(), lr=0.1, max_iter=20)

        get_value_loss = torch.nn.MSELoss()

        with torch.no_grad():
            reward_values = self.value_net(states)

        with torch.no_grad():
            cost_values = self.cost_net(states)

        reward_advantages, returns = estimate_advantages(rewards, masks, reward_values, self.args.gamma, self.args.tau, self.device)
        cost_advantages, cost_returns = estimate_advantages(costs, masks, cost_values, self.args.gamma, self.args.tau, self.device)
        constraint_value = estimate_constraint_value(costs, masks, self.args.gamma, self.device)

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

        reward_value_loss = get_value_loss(reward_values, returns)
        cost_value_loss = get_value_loss(cost_values, cost_returns)

        """update policy"""
        policy = self.meta_policy if meta_update else self.local_policy
        with torch.no_grad():
            fixed_log_probs = policy.get_log_prob(states, actions)

        """define the loss function for TRPO"""
        def get_loss(volatile=False, advantage=False):
            with torch.set_grad_enabled(not volatile):
                log_probs = policy.get_log_prob(states, actions)
                action_loss = -advantage * torch.exp(log_probs - fixed_log_probs)
                return action_loss.mean()

        reward_loss = get_loss(advantage=reward_advantages)
        cost_loss = get_loss(advantage=-cost_advantages)

        """Compute gradient of loss"""
        grads = torch.autograd.grad(reward_loss, policy.parameters(), retain_graph=True, create_graph=True)
        loss_reward_grad = torch.cat([grad.view(-1) for grad in grads])
        loss_reward_grad = loss_reward_grad / torch.norm(loss_reward_grad)
        g = loss_reward_grad.clone().detach().requires_grad_(True)

        grads = torch.autograd.grad(cost_loss, policy.parameters(), retain_graph=True, create_graph=True)            
        loss_cost_grad = torch.cat([grad.view(-1) for grad in grads])
        loss_cost_grad = loss_cost_grad / torch.norm(loss_cost_grad)
        a = loss_cost_grad.clone().detach().requires_grad_(True)

        """Define Differential Programming Parameter"""
        max_kl = torch.tensor(self.args.max_kl, dtype=self.dtype).to(self.device)
        max_constraint = torch.tensor(self.args.max_constraint, dtype=self.dtype).to(self.device)
        b = (constraint_value[0] - max_constraint).to(self.device)

        is_infeasible = False
        try:
            # solve CPO using DCO
            step, = self.cpo_problem(g, max_kl, a, b)
        except:
            # Recover step; solely minimizing cost
            is_infeasible = True
            step, = self.trpo_problem(a, max_kl)

        if step_only:
            return step

        x_gradients = torch.zeros((self.param_size,)).to(self.device)
        loss_gradients = torch.zeros((self.param_size,)).to(self.device)

        for i in range(self.param_size):
            if not is_infeasible:
                step[i].backward()
                x_gradients[i] = g.grad[i]
                g_grad = torch.autograd.grad(loss_reward_grad[i], policy.parameters(), retain_graph=True)
                g_grad_flat = torch.cat([grad.view(-1) for grad in g_grad])
                loss_gradients[i] = g_grad_flat[i]
            else:
                step[i].backward()
                x_gradients[i] = a.grad[i]
                a_grad = torch.autograd.grad(loss_cost_grad[i], policy.parameters(), retain_graph=True)
                a_grad_flat = torch.cat([grad.view(-1) for grad in a_grad])
                loss_gradients[i] = a_grad_flat[i]

        meta_correction = 1 + x_gradients * loss_gradients

        prev_params = get_flat_params_from(policy)
        new_params = prev_params + step

        set_flat_params_to(self.local_policy, new_params)
        
        return meta_correction
    
    def cpo_step(self, batch, local_policy, use_fim=False):
        states = torch.from_numpy(np.stack(batch.state)[:self.args.max_batch_size]).to(self.dtype).to(self.device) #[:args.batch_size]
        costs =  torch.from_numpy(np.stack(batch.cost)[:self.args.max_batch_size]).to(self.dtype).to(self.device) #[:args.batch_size]
        actions = torch.from_numpy(np.stack(batch.action)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(np.stack(batch.reward)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
        masks = torch.from_numpy(np.stack(batch.mask)[:self.args.max_batch_size]).to(self.dtype).to(self.device)
        #c_states = states[:,12:]

        with torch.no_grad():
            reward_values = self.value_net(states)

        with torch.no_grad():
            cost_values = self.cost_net(states)

        """get advantage estimation from the trajectories"""
        reward_advantages, reward_returns = estimate_advantages(rewards, masks, reward_values, self.args.gamma, self.args.tau, self.device)
        cost_advantages, cost_returns = estimate_advantages(costs, masks, cost_values, self.args.gamma, self.args.tau, self.device)
        constraint_value = estimate_constraint_value(costs, masks, self.args.gamma, self.device)
        
        """update policy"""
        with torch.no_grad():
            fixed_log_probs = local_policy.get_log_prob(states, actions)

        def get_reward_loss(volatile=False):
            with torch.set_grad_enabled(not volatile):
                log_probs = local_policy.get_log_prob(states, actions)
                action_loss = -reward_advantages * torch.exp(log_probs - fixed_log_probs)
                return action_loss.mean()
            
        """define the cost loss function for TRPO"""
        def get_cost_loss(volatile=False):
            with torch.set_grad_enabled(not volatile):
                log_probs = local_policy.get_log_prob(states, actions)
                action_loss = cost_advantages * torch.exp(log_probs - fixed_log_probs)
                return action_loss.mean()
        
        r_optim = torch.optim.LBFGS(self.value_net.parameters(), lr=0.1, max_iter=20)
        c_optim = torch.optim.LBFGS(self.cost_net.parameters(), lr=0.1, max_iter=20)

        get_value_loss = torch.nn.MSELoss()

        #for i in range(10):    
        def r_closure():
            r_optim.zero_grad()
            r_pred = self.value_net(states)
            #print('r comparison: ', r_pred[0][0], reward_returns[0][0])
            r_loss = get_value_loss(r_pred, reward_returns)
            for param in self.value_net.parameters():
                r_loss += param.pow(2).sum() * self.args.l2_reg
            r_loss.backward()
            return r_loss
        
        r_optim.step(r_closure)

        #for i in range(10):    
        def c_closure():
            c_optim.zero_grad()
            c_pred = self.cost_net(states)
            #print('c comparison: ', c_pred[0][0], cost_returns[0][0])
            c_loss = get_value_loss(c_pred, cost_returns)
            for param in self.cost_net.parameters():
                c_loss += param.pow(2).sum() * self.args.l2_reg
            c_loss.backward()
            return c_loss
        
        c_optim.step(c_closure)

        r_value_loss = get_value_loss(reward_values, reward_returns)
        c_value_loss = get_value_loss(cost_values, cost_returns)

        reward_loss = get_reward_loss()
        cost_loss = get_cost_loss()

        """define the loss function for TRPO"""
        reward_grads = torch.autograd.grad(reward_loss, local_policy.parameters())
        loss_reward_grad = torch.cat([grad.view(-1) for grad in reward_grads])

        cost_grads = torch.autograd.grad(cost_loss, local_policy.parameters())
        loss_cost_grad = torch.cat([grad.view(-1) for grad in cost_grads])

        """Gradient normalization"""
        if self.args.grad_norm:
            loss_reward_grad = loss_reward_grad / torch.norm(loss_reward_grad)
            loss_cost_grad = loss_cost_grad / torch.norm(loss_cost_grad)   

        """DP"""
        g = loss_reward_grad.clone().detach()
        a = loss_cost_grad.clone().detach()

        max_kl = torch.tensor(self.args.max_kl, dtype=self.dtype).to(self.device)
        max_constraint = torch.tensor(self.args.max_constraint, dtype=self.dtype)

        b = (constraint_value[0] - max_constraint).to(self.device)
        print('b: ', b)
        
        try:
            step, = self.cpo_problem(g, max_kl, a, b)
            #print('step mean: ', abs(step).mean())
            prev_params = get_flat_params_from(local_policy)
            expected_reward_improve = -loss_reward_grad.dot(step)
            expected_cost_improve = -loss_cost_grad.dot(step)
            _, new_params = self.line_search(local_policy, get_reward_loss, get_cost_loss, prev_params, step,
                                              expected_reward_improve, expected_cost_improve)

            set_flat_params_to(local_policy, new_params)
        except:
            step, = self.trpo_problem(a, max_kl)
            print('recover mean: ', abs(step).mean())

            prev_params = get_flat_params_from(local_policy)
            expected_reward_improve = -loss_reward_grad.dot(step)
            expected_cost_improve = -loss_cost_grad.dot(step)
            _, new_params = self.line_search(local_policy, get_reward_loss, get_cost_loss, prev_params, step,
                                              expected_reward_improve, expected_cost_improve)
            set_flat_params_to(local_policy, new_params)

            return r_value_loss, c_value_loss, reward_loss, cost_loss

        return r_value_loss, c_value_loss, reward_loss, cost_loss
            
    def project_step(self, batch, meta_step):
        states = torch.from_numpy(np.stack(batch.state)).to(self.dtype).to(self.device) #[:args.batch_size]
        costs = torch.from_numpy(np.stack(batch.cost)).to(self.dtype).to(self.device)
        actions = torch.from_numpy(np.stack(batch.action)).to(self.dtype).to(self.device)
        masks = torch.from_numpy(np.stack(batch.mask)).to(self.dtype).to(self.device)
        
        with torch.no_grad():
            cost_values = self.cost_net(states)

        cost_advantages, _ = estimate_advantages(costs, masks, cost_values, self.args.gamma, self.args.tau, self.device)
        constraint_value = estimate_constraint_value(costs, masks, self.args.gamma, self.device)
        
        with torch.no_grad():
            fixed_log_probs = self.meta_policy.get_log_prob(states, actions)

        def get_cost_loss(volatile=False):
            with torch.set_grad_enabled(not volatile):
                log_probs = self.meta_policy.get_log_prob(states, actions)
                cost_loss = cost_advantages * torch.exp(log_probs - fixed_log_probs)
                return cost_loss.mean()   
        
        cost_loss = get_cost_loss()

        grads = torch.autograd.grad(cost_loss, self.meta_policy.parameters())            
        loss_cost_grad = torch.cat([grad.view(-1) for grad in grads])
        loss_cost_grad = loss_cost_grad / torch.norm(loss_cost_grad)
        a = loss_cost_grad.clone().detach()

        max_constraint = torch.tensor(self.args.max_constraint, dtype=self.dtype).to(self.device)
        b = (constraint_value[0] - max_constraint).to(self.device)
        
        try:
            step, = self.projection(a, b, meta_step)
        except:
            return torch.zeros(a.numel())
        print('projecting...  b: ', b, 'projection mean: ', abs(step).mean(), 'projection norm: ', torch.norm(step))

        return step
    
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
            self.local_policy = self.meta_policy
            for m_iter in range(self.args.meta_iter_num):
                sample_time = 0
                seed = deterministic_rollout_seed(self.args.seed, env_parameter_index, m_iter, 0)
                batch, log = self.collect_samples(self.env, env_parameter_list[env_parameter_index], self.args.env_name, self.local_policy, seed)
                batch=batch.sample()
                
                sample_time += log['sample_time']

                t1 = time.time()
                _, _, _, _ = self.cpo_step(batch,self.local_policy)
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

                writer.add_scalar('meta_rewards',  eval_reward, m_iter)
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
        return meta_avg_cost

    def train_CPOMeta(self, writer, save_info_obj):
        # lists for dumping plotting data for agent
        env_avg_reward = []
        env_avg_cost = []

        # lists for dumping plotting data for mean agent
        iter_for_best_avg_reward = None

        # for saving the best model
        best_avg_reward = -99999999

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
            meta_update_sum = 0
            sample_time = 0
            update_time = 0

            avg_reward = 0
            avg_cost = 0
            
            env_parameter_list=create_env_parameter_list(self.args)
            # Collect samples
            for local_iter in range(self.args.env_num):
                t1 = time.time()
                memory, _ = self.collect_samples(self.env, env_parameter_list[local_iter],self.args.env_name, self.meta_policy, seed=self.args.seed)
                t2 = time.time()
                sample_time += t2 - t1
                meta_correction = self.update(memory.sample(), meta_update=True)
                t3 = time.time(); update_time += t3 - t2

                """local-updating"""
                for k_iter in range(self.args.local_num):
                    print(f'Updating iter: {i_iter} \t local_num: {local_iter}/{self.args.env_num-1} | k_num {k_iter}/{self.args.local_num-1}', end="\r")
                    t1 = time.time()
                    local_memory, _ = self.collect_samples(self.env, env_parameter_list[local_iter],self.args.env_name, self.local_policy, seed=local_iter)
                    t2 = time.time(); sample_time += t2 - t1
                    memory.append(local_memory)
                    if k_iter+1 == self.args.local_num:
                        local_correction = self.update(memory.sample(), step_only=True)
                    else:
                        local_correction = self.update(memory.sample())
                    t3 = time.time(); update_time += t3 - t2
                    meta_correction *= local_correction

                meta_update_sum += meta_correction

                # compute avg reward and cost for log
                rewards =  np.stack(local_memory.sample().reward)
                costs =  torch.from_numpy(np.stack(local_memory.sample().cost)).to(self.dtype).to(self.device)
                masks = torch.from_numpy(np.stack(local_memory.sample().mask)).to(self.dtype).to(self.device)
                
                num_episode = self.args.min_batch_size/self.args.time_horizon
                avg_reward += (np.sum(rewards) / num_episode) / self.args.env_num
                avg_cost += estimate_constraint_value(costs, masks, self.args.gamma, self.device)[0].to(torch.device('cpu')) / self.args.env_num

            prev_params = get_flat_params_from(self.meta_policy)            
            new_params = prev_params + meta_update_sum/self.args.env_num
            set_flat_params_to(self.meta_policy, new_params)
        
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

        print(iter_for_best_avg_reward, 'Best eval R:', best_avg_reward)

if __name__ == '__main__':
    print('CPOMeta')