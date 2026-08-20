"""
Advantage and constraint-value estimation shared by all four algorithms.
"""

import torch
from utils import to_device
import numpy as np

def estimate_advantages(rewards, masks, values, gamma, tau, device):
    """
    Generalised advantage estimation, walking the batch backwards.

    ``masks[i]`` is 0 at a terminal transition, which resets the recursion.
    Advantages are mean-centred but deliberately not divided by their std, so
    that the scale of the reward and cost advantages stays comparable -- the
    constrained steps in ``algos/`` rely on that.

    Returns ``(advantages, returns)`` on ``device``.
    """
    rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) #/ advantages.std() #should be modified
    advantages, returns = to_device(device, advantages, returns)
    return advantages, returns



def estimate_constraint_value(costs, masks, gamma, device):
    """
    Average discounted cost per trajectory, i.e. the constraint value J_C(pi).

    This is the quantity compared against ``--max-constraint``.
    """
    costs, masks = to_device(torch.device('cpu'), costs, masks)
    tensor_type = type(costs)
    constraint_value = torch.tensor(0)

    j = 1
    traj_num = 1
    for i in range(costs.size(0)):
        constraint_value = constraint_value + costs[i] * gamma**(j-1)

        if masks[i] == 0:
            j = 1 #reset
            traj_num = traj_num + 1
        else: 
            j = j+1
            
    constraint_value = constraint_value/traj_num
    constraint_value = to_device(device, constraint_value)
    return constraint_value