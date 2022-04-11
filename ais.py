import numpy as np
from tqdm.auto import tqdm

import torch
from torch.autograd import grad as torchgrad
from third.BDMC import hmc
from third.BDMC import utils
import pdb


def log_sum_weighted_exp(val1, val2, weight1, weight2):
    """This is like log sum exp but with two weighted arguments

    Args:
        val1: first value inside log sum exp
        val2: second value inside log sum exp
        weight1: weight of first value
        weight2: weight of second value

    Returns:
        float: log of (weight1 * exp(val1) + weight2 * exp(val2))
    """
    val_max = torch.where(val1 > val2, val1, val2)
    val1_exp = weight1 * torch.exp(val1 - val_max)
    val2_exp = weight2 * torch.exp(val2 - val_max)
    lse = val_max + torch.log(val1_exp + val2_exp)
    return lse


def log_f_i_geometric(model, z, x, y, beta, loglik_scale=1.0):
    """Compute log unnormalised density for intermediate distribution `f_i`
        using a geometric mixture:
        f_i = p(z)^(1-beta) p(x,z)^(beta) = p(z) p(x|z)^(scale*beta)
        => log f_i = log p(z) + beta * scale * log p(x|z)

    Args:
        model: model that can invoke log prior and log likelihood
        z: latent value at which to evaluate log density
        data: data to compute the log likelihood
        beta: mixture weight
        scale: log lik scale
    Returns:
        log unnormalised density
    """

    log_prior = model.log_prior(z)
    log_likelihood = loglik_scale * model.log_likelihood(z, x, y)
    log_joint = log_prior + log_likelihood.mul_(beta)
    return log_joint


def log_f_i_qpath(model, z, x, y, beta, q, loglik_scale=1.0):
    """Compute log unnormalised density for intermediate distribution `f_i`
        using a q-mixture:
        f_i = ( (1-beta) p(z)^(1-q) + beta p(x,z)^(1-q) )^(1/(1-q))

    Args:
        model: model that can invoke log prior and log likelihood
        z: latent value at which to evaluate log density
        data: data to compute the log likelihood
        beta: mixture weight
        q: q value for the mixture
        use_cuda: whether using cuda or not

    Returns:
        log unnormalised density
    """
    log_prior = model.log_prior(z)
    log_likelihood = loglik_scale * model.log_likelihood(z, x, y)
    log_a = log_prior
    log_b = log_prior + log_likelihood
    if beta == 0.0:
        log_joint = log_a
    elif beta == 1.0:
        log_joint = log_b
    else:
        log_a = (1 - q) * log_a
        log_b = (1 - q) * log_b
        lse = log_sum_weighted_exp(log_a, log_b, 1 - beta, beta)
        log_joint = lse / (1 - q)
    return log_joint


def log_f_i(model, z, x, y, beta, use_qpath=False, q=0.8, loglik_scale=1.0):
    """Wrapper function, return log_f_i_geometric or log_f_i_qpath"""
    if use_qpath:
        return log_f_i_qpath(model, z, x, y, beta, q, loglik_scale)
    else:
        return log_f_i_geometric(model, z, x, y, beta, loglik_scale)


def ais_trajectory_mini_batch(
    model,
    x,
    y,
    schedule=np.linspace(0.0, 1.0, 500),
    n_sample=100,
    use_cuda=False,
    L=10,
    batch_size=100,
    no_ar=False,
    step_size=0.01,
    loglik_scale=1.0,
):
    start_idx = 0
    idx = 0
    done = False
    while not done:
        end_idx = start_idx + batch_size
        if end_idx >= n_sample:
            end_idx = n_sample
            done = True

        print("running ais, samples %d to %d..." % (start_idx, end_idx))
        lw, s = ais_trajectory(
            model,
            x,
            y,
            schedule,
            end_idx - start_idx,
            use_cuda,
            L,
            no_ar,
            step_size,
            loglik_scale,
        )
        if idx == 0:
            lws = lw
            samples = s
        else:
            lws = torch.cat((lws, lw))
            samples = torch.cat((samples, s), 0)

        start_idx = end_idx
        idx += 1
    const_term = torch.log(torch.Tensor([n_sample]))
    lml_estimate = torch.logsumexp(lws.cpu(), dim=0) - const_term
    return lml_estimate, samples


def ais_trajectory(
    model,
    x,
    y,
    schedule=np.linspace(0.0, 1.0, 500),
    n_sample=100,
    use_cuda=False,
    L=10,
    no_ar=False,
    step_size=0.01,
    loglik_scale=1.0,
):
    K = n_sample

    with torch.no_grad():
        if use_cuda:
            epsilon = torch.ones(K).cuda().mul_(step_size)
            accept_hist = torch.zeros(K).cuda()
            logw = torch.zeros(K).cuda()
        else:
            epsilon = torch.ones(K).mul_(step_size)
            accept_hist = torch.zeros(K)
            logw = torch.zeros(K)

    # current_z = torch.randn(K, model.param_dim)
    current_z = model.get_prior_samples(K)
    if use_cuda:
        current_z = current_z.cuda()
    current_z = current_z.requires_grad_()

    for j, (t0, t1) in tqdm(enumerate(zip(schedule[:-1], schedule[1:]), 1)):
        # update log importance weight
        log_int_1 = log_f_i(
            model, current_z, x, y, t0, loglik_scale=loglik_scale
        )
        log_int_2 = log_f_i(
            model, current_z, x, y, t1, loglik_scale=loglik_scale
        )
        log_diff = log_int_2 - log_int_1
        logw += log_diff.detach()
        # resample velocity
        current_v = torch.randn(current_z.size())
        if use_cuda:
            current_v = current_v.cuda()

        def U(z):
            return -log_f_i(model, z, x, y, t1, loglik_scale=loglik_scale)

        def grad_U(z):
            # grad w.r.t. outputs; mandatory in this case
            grad_outputs = torch.ones(K)
            if use_cuda:
                grad_outputs = grad_outputs.cuda()
            # torch.autograd.grad default returns volatile
            grad = torchgrad(U(z), z, grad_outputs=grad_outputs)[0]
            # clip by norm
            max_ = K * model.param_dim * 100.0
            grad = torch.clamp(grad, -max_, max_)
            grad.requires_grad_()
            return grad

        def normalized_kinetic(v):
            zeros = torch.zeros(K, model.param_dim)
            if use_cuda:
                zeros = zeros.cuda()
            return -utils.log_normal(v, zeros, zeros)

        z, v = hmc.hmc_trajectory(current_z, current_v, grad_U, epsilon, L=L)
        if not no_ar:
            current_z, epsilon, accept_hist = hmc.accept_reject(
                current_z,
                current_v,
                z,
                v,
                epsilon,
                accept_hist,
                j,
                U,
                K=normalized_kinetic,
            )
        current_z = current_z.detach()
        current_z = current_z.requires_grad_()
    return logw, current_z.detach()
