"""
Trust-region building blocks.

``conjugate_gradients`` and ``linesearch`` back the standard TRPO step; the
``one_step_trpo*`` variants implement the single-inner-step, constraint-aware
updates that SafeMeta and MAML_constraint use for task adaptation.
"""

import numpy as np

import torch
from torch.autograd import Variable
from utils.torch import *


def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def linesearch(model,
               f,
               x,
               fullstep,
               expected_improve_rate,
               max_backtracks=10,
               accept_ratio=.1):
    fval = f(True).data
    print("fval before", fval.item())
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            print("fval after", newfval.item())
            return True, xnew
    return False, x


def trpo_step(model, get_loss, get_kl, max_kl, damping):
    loss = get_loss()
    grads = torch.autograd.grad(loss, model.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

    def Fvp(v):
        kl = get_kl()
        kl = kl.mean()

        grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, model.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + v * damping

    stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

    shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]

    neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
    print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))

    prev_params = get_flat_params_from(model)
    success, new_params = linesearch(model, get_loss, prev_params, fullstep,
                                     neggdotstepdir / lm[0])
    set_flat_params_to(model, new_params)

    return loss
    
def one_step_trpo(model, get_loss, get_kl,meta_lambda,lower_opt="Adam"):
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    if lower_opt=="Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        print("Adam")
    elif lower_opt=="adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)
        print("Adagrad")
    elif lower_opt=="rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0003)
        print("RMSprop")
    elif lower_opt=="sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
        print("SGD")
    
    for i in range(50):
        optimizer.zero_grad()
        loss = get_loss()*1.0/meta_lambda+get_kl().mean()
        #print("total_loss ", loss)
        #print("get_kl ",get_kl().mean())
        if get_kl().mean().clone().detach().numpy()>3.0:
            break
        loss.backward()
        optimizer.step()

    print("get_kl ",get_kl().mean())
    #print(model.action_log_std)

    return


def one_step_trpo_constraint(model, get_loss, get_constraint, bbbb, get_kl,meta_lambda,lower_opt="Adam"):
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    if lower_opt=="Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        print("Adam")
    elif lower_opt=="adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)
        print("Adagrad")
    elif lower_opt=="rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0003)
        print("RMSprop")
    elif lower_opt=="sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
        print("SGD")
    
    u=0.0
    u_lr=1.0
    for i in range(100):
        optimizer.zero_grad()
        loss = (get_loss()+u*get_constraint())*1.0/meta_lambda+get_kl().mean()
        #print("total_loss ", loss)
        #print("get_kl ",get_kl().mean())
        if get_kl().mean().clone().detach().numpy()>3.0:
            break
        loss.backward()
        optimizer.step()
        grad_u=-get_constraint().clone().detach().numpy()+bbbb
        if u - u_lr* grad_u < 0.0:
            u=0.0
        elif u - u_lr* grad_u > 8.0:
            u=8.0
        else:
            u = u - u_lr* grad_u

    print("get_kl ",get_kl().mean())
    print("multiplier: ", u)
    #print(model.action_log_std)

    return

def one_step_trpo_constraint_2(this_local_policy, meta_lambda, states, actions, fixed_log_probs, cost_advantages, reward_advantages, bbbb):
    u=0.0
    u_lr=0.1
    optim11 = torch.optim.SGD(this_local_policy.parameters(), lr=3e-3)
    for i in range(100):
        log_probs = this_local_policy.get_log_prob(states, actions)
        log_probs1=log_probs.clone().detach().data
        grad_u=-(cost_advantages * torch.exp(log_probs1-fixed_log_probs)).mean() + bbbb
        if u - u_lr* grad_u < 0.0:
            u=0.0
        else:
            u = u - u_lr* grad_u
        #print(u)
        #u = 0.0
        #std=(reward_advantages-u*cost_advantages).std()
        loss_inter=(torch.clamp(log_probs-fixed_log_probs,-2.0,2.0)-1.0/meta_lambda*(reward_advantages-u*cost_advantages))*torch.special.expit(2.0*torch.exp(log_probs - fixed_log_probs)-2.0)*2
        loss11=loss_inter.mean()
        optim11.zero_grad()
        loss11.backward()
        optim11.step()
        if np.abs(loss11.data.numpy())>0.5:
            print("gg")
            break
    #print("multiplier_grad: ", -(cost_advantages * torch.exp(log_probs1-fixed_log_probs)).mean())
    print("loss: ", loss11)
    print("multiplier: ", u)