import torch
import torch.nn as nn
import matplotlib.pylab as plt
import numpy as np

from bnn import BNN, GaussianLikelihood
from ais import ais_trajectory_mini_batch


def get_reg_data(dataset, N, real_noise_std):
    if dataset == "sin":
        x = torch.randn(N, 1) * 2
        x = torch.where(x < 0, x - 4, x)
        x = torch.where(x < 0, x, x + 0.5)
        y = torch.sin(x) + torch.randn(N, 1) * real_noise_std
        # x = (x - x.mean()) / x.std()
        x = (x + 1) / 3.0
    elif dataset == "linear":
        x = torch.randn(N, 1) * 2
        x = torch.where(x < 0, x - 4, x)
        x = torch.where(x < 0, x, x + 0.5)
        y = x / 5 + torch.randn(N, 1) * real_noise_std
        x = (x - x.mean()) / x.std()
    elif dataset == "cubic":
        x = torch.randn(N, 1) * 2
        x = torch.where(x < 0, x - 4, x)
        x = torch.where(x < 0, x, x + 0.5)
        y = x**3
        # print(x.mean(), x.std())
        # print(y.mean(), y.std())
        # x = (x - x.mean()) / x.std()
        # y = (y - y.mean()) / y.std()
        x = (x + 1) / 3
        y = (y + 75) / 140
        y += torch.randn(N, 1) * real_noise_std
    elif dataset == "cluster":
        x = torch.cat(
            [
                -1.0 + torch.randn(N // 2, 1) * 0.1,
                1.0 + torch.randn(N // 2, 1) * 0.1,
            ],
            0,
        )
        y = torch.cat(
            [
                -2 + torch.randn(N // 2, 1) * real_noise_std,
                2 + torch.randn(N // 2, 1) * real_noise_std,
            ],
            0,
        )
    else:
        raise Exception("unknown dataset name")
    return x, y


def compute_mse_and_ll(truth, pred, loss_fn, log_w):
    if log_w is not None:
        pred_mod = pred * log_w.exp().unsqueeze(-1).unsqueeze(-1)
    else:
        pred_mod = pred
    mse = torch.mean((pred_mod - truth) ** 2)
    neg_loss = -loss_fn(pred, truth, reduction=False, use_cuda=False)
    if log_w is not None:
        neg_loss += log_w.unsqueeze(-1)
    ll = torch.logsumexp(neg_loss, 0).mean()
    if log_w is None:
        ll -= np.log(neg_loss.shape[0])
    return mse, ll


def run_exp(
    no_train_points=50,
    no_test_points=50,
    dataset="sin",
    network_size=[1, 50, 1],
    real_noise_std=0.4,
    eval_noise_std=0.4,
    eval_prior_std=1.0,
    seed=0,
    ais_chain_length=1000,
    ais_no_samples=100,
    ais_step_size=0.01,
    ais_no_leapfrog=10,
    vi_no_eval_samples=100,
    vi_no_train_samples=5,
    vi_no_train_epochs=5000,
    vi_learning_rate=0.01,
    method="vi",
    plot=False,
    act_func="relu",
    path="./tmp",
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    use_cuda = torch.cuda.is_available()

    # create dataset
    x_train, y_train = get_reg_data(dataset, no_train_points, real_noise_std)
    x_test, y_test = get_reg_data(dataset, no_test_points, real_noise_std)
    x_train = x_train.cuda() if use_cuda else x_train
    y_train = y_train.cuda() if use_cuda else y_train
    x_test = x_test.cuda() if use_cuda else x_test
    y_test = y_test.cuda() if use_cuda else y_test
    output_dim = y_train.shape[1]

    # create a model
    likelihood = GaussianLikelihood(output_dim, eval_noise_std, use_cuda)
    act = nn.functional.relu if act_func == "relu" else nn.functional.tanh
    model = BNN(
        network_size,
        likelihood,
        prior_mean=0,
        prior_std=eval_prior_std,
        act_function=act,
        use_cuda=use_cuda,
    )
    if use_cuda:
        model = model.cuda()
    post_log_weights = None
    if method == "vi":
        # train the model using VI
        model.train_vi(
            x_train,
            y_train,
            no_epochs=vi_no_train_epochs,
            no_samples=vi_no_train_samples,
            lrate=vi_learning_rate,
        )
        lml_estimate = model.compute_vi_objective(
            x_train, y_train, no_samples=vi_no_eval_samples
        )
        post_samples = model.get_posterior_samples_vi(vi_no_eval_samples)
    elif method == "ais":
        # run AIS with a linearly scheduled geometric path
        beta_schedule = np.linspace(0.0, 1.0, ais_chain_length)
        lml_estimate, post_samples = ais_trajectory_mini_batch(
            model,
            x_train,
            y_train,
            schedule=beta_schedule,
            n_sample=ais_no_samples,
            batch_size=ais_no_samples,
            step_size=ais_step_size,
            L=ais_no_leapfrog,
            use_cuda=use_cuda,
        )
    else:
        print("method %s not known" % method)

    if post_log_weights is not None:
        post_log_weights = post_log_weights.cpu()
    train_pred = model.forward(post_samples, x_train).detach().cpu()
    test_pred = model.forward(post_samples, x_test).detach().cpu()
    train_mse, train_ll = compute_mse_and_ll(
        y_train.cpu(), train_pred, model.cpu().likelihood.loss, post_log_weights
    )
    test_mse, test_ll = compute_mse_and_ll(
        y_test.cpu(), test_pred, model.cpu().likelihood.loss, post_log_weights
    )

    res = [lml_estimate, train_mse, train_ll, test_mse, test_ll]

    print("%s stochastic lower bound %.4f" % (method, lml_estimate))
    print("train mse %.3f, ll %.3f " % (train_mse, train_ll))
    print("test mse %.3f, ll %.3f " % (test_mse, test_ll))

    fname = "%s/dataset_%s_no_train_%d_no_test_%d_network_size_%s_act_%s" % (
        path,
        dataset,
        no_train_points,
        no_test_points,
        network_size,
        act_func,
    )
    fname += (
        "_seed_%d_method_%s_real_noise_std_%.3f_eval_noise_std_%.3f_eval_prior_std_%.3f"
        % (seed, method, real_noise_std, eval_noise_std, eval_prior_std)
    )

    if method == "ais":
        fname += "_step_size_%.3f_no_leapfrog_%d" % (
            ais_step_size,
            ais_no_leapfrog,
        )
        fname += "_chain_length_%d_no_samples_%d" % (
            ais_chain_length,
            ais_no_samples,
        )

    if plot:
        Nplot = 200
        xplot = torch.linspace(-3, 3, Nplot).reshape([Nplot, 1])
        if use_cuda:
            xplot_cuda = xplot.cuda()
        else:
            xplot_cuda = xplot
        plot_samples = model.forward(post_samples, xplot_cuda).detach().cpu()
        if post_log_weights is not None:
            post_weights = post_log_weights.exp().unsqueeze(-1).unsqueeze(-1)
            plot_mean = (plot_samples * post_weights).sum(0)
            plot_std = (
                (((plot_samples - plot_mean) ** 2) * post_weights).sum(0).sqrt()
            )
        else:
            plot_mean = plot_samples.mean(0)
            plot_std = plot_samples.std(0)

        plt.figure()
        plt.plot(xplot, plot_mean, "-k", linewidth=2)
        plt.fill_between(
            xplot[:, 0],
            plot_mean[:, 0]
            + 2 * np.sqrt(plot_std[:, 0] ** 2 + eval_noise_std**2),
            plot_mean[:, 0]
            - 2 * np.sqrt(plot_std[:, 0] ** 2 + eval_noise_std**2),
            color="k",
            alpha=0.3,
        )
        for i in range(10):
            plt.plot(
                xplot,
                plot_samples[i, :, :] + 0.01 * np.random.randn(),
                "-k",
                alpha=0.5
                if post_log_weights is None
                else float(post_weights[i].numpy()[0]),
            )

        plt.plot(x_train.cpu(), y_train.cpu(), "+", color="k", markersize=8)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim([-3, 3])
        plt.ylim([-3.5, 4])
        # plt.show()
        plt.savefig(
            fname + "_prediction.pdf",
            bbox_inches="tight",
            pad_inches=0,
        )

    return res


if __name__ == "__main__":
    import fire

    fire.Fire(run_exp)
