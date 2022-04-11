from matplotlib.pyplot import plot
from bnn_reg import run_exp
import numpy as np


real_noise_std = 0.4
no_train_points = 50
no_test_points = 50
network_size = [1, 50, 1]
ais_chain_length = 2000
ais_no_samples = 100
ais_step_size = 0.01
ais_no_leapfrog = 10
vi_no_eval_samples = 5000
vi_no_train_samples = 5
vi_no_train_epochs = 5000
vi_learning_rate = 0.01

act = "relu"
seed = 0
dataset = "sin"

# optimal values given by AIS log marginal likelihood
ais_noise_std = 0.4
ais_prior_std = 1.1

# optimal values given by MFVI log marginal likelihood
vi_noise_std = 0.75
vi_prior_std = 0.13


print(
    "we run AIS and VI using AIS opt hypers, noise std %.3f, prior std %.3f"
    % (ais_noise_std, ais_prior_std)
)
run_exp(
    no_train_points=no_train_points,
    no_test_points=no_test_points,
    dataset=dataset,
    network_size=network_size,
    real_noise_std=real_noise_std,
    eval_noise_std=ais_noise_std,
    eval_prior_std=ais_prior_std,
    seed=seed,
    ais_chain_length=ais_chain_length,
    ais_no_samples=ais_no_samples,
    ais_step_size=ais_step_size,
    ais_no_leapfrog=ais_no_leapfrog,
    method="ais",
    plot=True,
)

run_exp(
    no_train_points=no_train_points,
    no_test_points=no_test_points,
    dataset=dataset,
    network_size=network_size,
    real_noise_std=real_noise_std,
    eval_noise_std=ais_noise_std,
    eval_prior_std=ais_prior_std,
    seed=seed,
    vi_no_eval_samples=vi_no_eval_samples,
    vi_no_train_samples=vi_no_train_samples,
    vi_no_train_epochs=vi_no_train_epochs,
    vi_learning_rate=vi_learning_rate,
    method="vi",
    plot=True,
)


print(
    "we now run AIS and VI using VI opt hypers, noise std %.3f, prior std %.3f"
    % (vi_noise_std, vi_prior_std)
)
run_exp(
    no_train_points=no_train_points,
    no_test_points=no_test_points,
    dataset=dataset,
    network_size=network_size,
    real_noise_std=real_noise_std,
    eval_noise_std=vi_noise_std,
    eval_prior_std=vi_prior_std,
    seed=seed,
    ais_chain_length=ais_chain_length,
    ais_no_samples=ais_no_samples,
    ais_step_size=ais_step_size,
    ais_no_leapfrog=ais_no_leapfrog,
    method="ais",
    plot=True,
)

run_exp(
    no_train_points=no_train_points,
    no_test_points=no_test_points,
    dataset=dataset,
    network_size=network_size,
    real_noise_std=real_noise_std,
    eval_noise_std=vi_noise_std,
    eval_prior_std=vi_prior_std,
    seed=seed,
    vi_no_eval_samples=vi_no_eval_samples,
    vi_no_train_samples=vi_no_train_samples,
    vi_no_train_epochs=vi_no_train_epochs,
    vi_learning_rate=vi_learning_rate,
    method="vi",
    plot=True,
)
