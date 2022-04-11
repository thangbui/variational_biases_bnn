import torch
import torch.nn as nn
from torch.distributions import Independent, Normal, MultivariateNormal
from torch.distributions.kl import kl_divergence
import torch.distributions as dist

from tqdm.auto import tqdm
import pdb
import math


class BNN(nn.Module):
    def __init__(
        self,
        network_size,
        likelihood,
        prior_mean=0.0,
        prior_std=1.0,
        act_function=nn.functional.relu,
        use_cuda=False,
    ):
        nn.Module.__init__(self)

        self.network_size = network_size
        self.no_weight_layers = len(network_size) - 1
        self.param_dim = param_dim = self.compute_no_params(network_size)
        S = torch.Size([param_dim])
        prior_mean = torch.ones(S) * prior_mean
        if isinstance(prior_std, list):
            prior_std = self.set_prior_std(prior_std, network_size)
        else:
            prior_std = torch.ones(S) * prior_std
        if use_cuda:
            prior_mean = prior_mean.cuda()
            prior_std = prior_std.cuda()
        self.prior_dist = Independent(
            Normal(loc=prior_mean, scale=prior_std), 1
        )
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.act_function = act_function
        self.likelihood = likelihood

        self.q_mean = nn.Parameter(torch.Tensor(param_dim).normal_(0.0, 0.1))
        self.q_log_std = nn.Parameter(
            torch.log(torch.Tensor([0.01])) * torch.ones([param_dim])
        )
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    def set_prior_std(self, prior_std, network_size):
        no_layers = len(network_size) - 1
        for i in range(no_layers):
            din = network_size[i]
            dout = network_size[i + 1]
            no_weights = din * dout
            no_biases = dout
            prior_std_ws = torch.ones(no_weights) * prior_std[i]
            prior_std_bs = torch.ones(no_biases)
            if i == 0:
                prior_std_vec = torch.cat((prior_std_ws, prior_std_bs), 0)
            else:
                prior_std_vec = torch.cat(
                    (prior_std_vec, prior_std_ws, prior_std_bs), 0
                )
        return prior_std_vec

    def compute_no_params(self, network_size):
        no_layers = len(network_size) - 1
        no_params = 0
        for i in range(no_layers):
            din = network_size[i]
            dout = network_size[i + 1]
            no_params += din * dout + dout
        return no_params

    def forward(self, params, x):
        start_idx = 0
        inputs = x.unsqueeze(0).expand(params.size(0), -1, -1)
        for i in range(self.no_weight_layers):
            din = self.network_size[i]
            dout = self.network_size[i + 1]
            end_idx = start_idx + din * dout
            wi = params[:, start_idx:end_idx]
            wi = wi.reshape([wi.shape[0], din, dout])
            start_idx = end_idx
            end_idx = start_idx + dout
            bi = params[:, start_idx:end_idx].unsqueeze(1)
            a = torch.einsum("kio,kni->kno", wi, inputs) + bi
            inputs = self.act_function(a)

            start_idx = end_idx

        return a

    def log_prior(self, theta):
        return self.prior_dist.log_prob(theta)

    def log_likelihood(self, theta, x, y, reduction=True):
        output = self(theta, x)
        lp = -self.likelihood.loss(output, y, reduction=reduction)
        return lp

    def log_joint(self, theta, x, y):
        return self.log_prior(theta) + self.log_likelihood(theta, x, y)

    def loss_mfvi(self, x, y, no_samples):
        # compute analytic kl term
        q_mean = self.q_mean
        q_std = torch.exp(self.q_log_std)
        q_dist = Independent(Normal(loc=q_mean, scale=q_std), 1)
        kl_term = kl_divergence(q_dist, self.prior_dist)

        # reparameterisation trick
        q_samples = q_dist.rsample(torch.Size([no_samples]))
        lik = self.log_likelihood(q_samples, x, y).mean()
        return kl_term, lik

    def log_normaliser_diag_normal(self, mean, std):
        ind_terms = 0.5 * mean**2 / std**2 + 0.5 * torch.log(std)
        return torch.sum(ind_terms)

    def compute_stochastic_cavity(
        self, post_mean, post_std, prior_mean, prior_std, no_datapoints, alpha
    ):
        post_var = post_std**2
        prior_var = prior_std**2
        post_eta_1, post_eta_2 = post_mean / post_var, 1.0 / post_var
        prior_eta_1, prior_eta_2 = prior_mean / prior_var, 1.0 / prior_var
        sub_fraction = alpha / no_datapoints
        cav_eta_1 = post_eta_1 - sub_fraction * (post_eta_1 - prior_eta_1)
        cav_eta_2 = post_eta_2 - sub_fraction * (post_eta_2 - prior_eta_2)
        cav_mean, cav_var = cav_eta_1 / cav_eta_2, 1.0 / cav_eta_2
        return cav_mean, cav_var.sqrt()

    def loss_bb_alpha(self, x, y, no_samples, alpha):
        N = x.shape[0]
        # compute analytic log normaliser terms
        # posterior
        q_mean = self.q_mean
        q_std = torch.exp(self.q_log_std)
        q_logz = self.log_normaliser_diag_normal(q_mean, q_std)
        # prior
        prior_mean, prior_std = self.prior_mean, self.prior_std
        prior_logz = self.log_normaliser_diag_normal(prior_mean, prior_std)
        # cavity
        cav_mean, cav_std = self.compute_stochastic_cavity(
            q_mean, q_std, prior_mean, prior_std, x.shape[0], alpha
        )
        cav_logz = self.log_normaliser_diag_normal(cav_mean, cav_std)
        # compute log tilted term
        cav_dist = Independent(Normal(loc=cav_mean, scale=cav_std), 1)
        # reparameterisation trick
        q_samples = cav_dist.rsample(torch.Size([no_samples]))
        log_lik_terms = self.log_likelihood(q_samples, x, y, reduction=False)
        lse = torch.logsumexp(alpha * log_lik_terms, dim=0)
        log_tilted = (lse - math.log(no_samples)).sum()
        return q_logz, prior_logz, cav_logz, log_tilted

    def train_vi(
        self,
        x,
        y,
        no_epochs=500,
        no_samples=10,
        lrate=0.01,
        print_cadence=500,
        batch_size=None,
    ):
        optim = torch.optim.Adam(self.parameters(), lr=lrate)
        Nall = x.shape[0]
        if batch_size is None:
            batch_size = Nall
        count = 0
        pbar = tqdm(range(no_epochs))
        for e in pbar:
            start_ind = 0
            while start_ind < Nall:
                end_ind = start_ind + batch_size
                if end_ind > Nall:
                    end_ind = Nall
                optim.zero_grad()
                x_batch = x[start_ind:end_ind, :]
                y_batch = y[start_ind:end_ind]
                n_batch = x_batch.shape[0]
                kl, lik = self.loss_mfvi(x_batch, y_batch, no_samples)
                loss = kl / Nall - lik / n_batch
                loss.backward()
                optim.step()
                if count % print_cadence == 0:
                    # tqdm.write(
                    #     "Epoch {}/{} \t loss {:.4f}".format(e, no_epochs, loss)
                    # )
                    pbar.set_postfix(
                        {"epoch": e, "total": no_epochs, "loss": loss.detach()}
                    )
                count += 1
                start_ind = end_ind

    def compute_vi_objective(self, x, y, no_samples=500, batch_size=None):
        loss = 0
        no_batches = 0
        Nall = x.shape[0]
        if batch_size is None:
            batch_size = Nall
        start_ind = 0
        while start_ind < Nall:
            end_ind = start_ind + batch_size
            if end_ind > Nall:
                end_ind = Nall
            x_batch = x[start_ind:end_ind, :]
            y_batch = y[start_ind:end_ind]
            n_batch = x_batch.shape[0]
            kl, lik = self.loss_mfvi(x_batch, y_batch, no_samples)
            loss += (kl - lik / n_batch * Nall).detach()
            start_ind = end_ind
            no_batches += 1
        loss /= no_batches
        return loss.detach()

    def predict_vi(self, x, no_samples):
        q_samples = self.get_posterior_samples_vi(no_samples)
        return self.forward(q_samples, x)

    def get_posterior_samples_vi(self, no_samples):
        q_mean = self.q_mean
        q_std = torch.exp(self.q_log_std)
        q_dist = Independent(Normal(loc=q_mean, scale=q_std), 1)
        q_samples = q_dist.rsample(torch.Size([no_samples]))
        return q_samples

    def get_prior_samples(self, no_samples):
        return self.prior_dist.sample(torch.Size([no_samples]))


class GaussianLikelihood(nn.Module):
    """
    Independent multi-output Gaussian likelihood.
    """

    def __init__(self, out_size, noise_std, use_cuda=False):
        super().__init__()
        self.noise_std = noise_std * torch.ones(out_size)
        if use_cuda:
            self.noise_std = self.noise_std.cuda()
        self.use_cuda = use_cuda

    def forward(self, mu):
        """
        Arguments:
            mu: no_samples x batch_size x out_size

        Returns:
            observation mean and variance

            obs_mu: no_samples x batch_size x out_size
            obs_var: no_samples x batch_size x out_size
        """
        obs_mu = mu
        zero_vec = torch.zeros_like(obs_mu)
        if self.use_cuda:
            zero_vec = zero_vec.cuda()
        obs_var = zero_vec + torch.square(self.noise_std)
        return obs_mu, obs_var

    def loss(self, f_samples, y, reduction=True, use_cuda=True):
        """
        Arguments:
            f_samples: batch_size x out_size
            y: batch_size x out_size

        Returns:
            Total loss scalar value.
        """
        obs_mean, obs_var = self(f_samples)

        if not use_cuda:
            obs_mean = obs_mean.cpu()
            obs_var = obs_var.cpu()

        y_dist = dist.Normal(obs_mean, obs_var.sqrt())
        log_prob = y_dist.log_prob(y)

        nll = -log_prob.sum(dim=-1)
        if reduction:
            nll = nll.sum(dim=-1)
        return nll

    def predict(self, f_samples):
        return f_samples
