#! /usr/bin/python3

import torch
import numpy as np
import matplotlib.pylab as plt
# plt.style.use('icml')


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optbnn.gp.models.gpr import GPR
from optbnn.gp import kernels, mean_functions
from optbnn.bnn.reparam_nets import GaussianMLPReparameterization
from optbnn.bnn.nets.mlp import MLP
from optbnn.bnn.likelihoods import LikGaussian
from optbnn.bnn.priors import FixedGaussianPrior, OptimGaussianPrior
from optbnn.prior_mappers.wasserstein_mapper import MapperWasserstein, WassersteinDistance
from optbnn.utils.rand_generators import MeasureSetGenerator, GridGenerator
from optbnn.utils.normalization import normalize_data
from optbnn.utils import plot_utils
from optbnn.sgmcmc_bayes_net.regression_net import RegressionNet
from optbnn.datasets import synthdata

model_seed = 3424121
torch.manual_seed(0)
np.random.seed(model_seed)

PLOT_DATA = 0
PLOT_PRIOR = True
figsize = np.array(plt.rcParams.get('figure.figsize').copy())

scale_figsize = lambda x: figsize * np.array(x)

N = 64
M = 500
a, b = -3, 5

sn2 = 0.01
leng = 1
ampl = 1 

width, depth = 50, 6
transfer_fn = 'cos'
kernel_fn = 'rbf'

Nsamples = 1000
mapper_num_iters=20
sampling_configs = {
    "batch_size": min(32, N),
    "num_samples": 100,
    "n_discarded": 20,
    "num_burn_in_steps": 5000,
    "keep_every": 100,
    "lr": 0.01,
    "num_chains": 1,
    "mdecay": 0.01,
    "print_every_n_samples": 10
}


# suffix = transfer_fn + "{:d}x{:d}".format(width,depth)
suffix = kernel_fn + transfer_fn + "{:d}x{:d}".format(width,depth)
out_dir = './exp/synthetic_' + str(model_seed) + '_' + suffix


def make_data(N, M):
    # X = np.arange(-3, 3, 1/N,)[:, None] #np.random.rand(N, 1) * (b-a) + a
    X = np.random.rand(N, 1) * 20 - 10
    synthdata.make_random_gap(X, gap_ratio=0.4)
    y = synthdata.gp_sample(X, ampl=1.6, leng=1.8, sn2=sn2)
    X = ((X + 10)/ 20) * 4 - 1

    # X = np.random.rand(100, 1) * 6
    # y = 0.5 * np.cos(0.4 * np.pi * X) + np.sin(0.8 * np.pi * X) + 0.2 * np.cos(1.4 * np.pi * X + 0.5) + np.random.randn(*X.shape) * np.sqrt(sn2)

    # X, y = get_snelson()

    Xtest = np.linspace(-3, 5, M).reshape(-1, 1)

    return X, y, Xtest

def plot_dataset(X, y):
    fig, ax = plt.subplots()
    plot_utils.plot_data(X, y, ax)
    ax.set_title('Dataset')
    if PLOT_DATA:
        plt.show()

def build_gpmodel(X, y, leng, ampl, sn2):
    kernels_kv = {'rbf': kernels.RBF, 'matern32': kernels.Matern32}
    kernel = kernels_kv[kernel_fn](input_dim=1, ARD=False,
        lengthscales=torch.tensor([leng], dtype=torch.double),
        variance=torch.tensor([ampl**2], dtype=torch.double))
    gpmodel = GPR(X=torch.from_numpy(X), Y=torch.from_numpy(y).reshape([-1, 1]),
             kern=kernel, mean_function=mean_functions.Zero())
    gpmodel.likelihood.variance.set(sn2)
    return gpmodel

def optimize_bnn_prior():
    # data_generator = MeasureSetGenerator(X, x_min=np.array([a]), x_max=np.array([b]))
    data_generator = GridGenerator(x_min=a, x_max=b)
    mapper = MapperWasserstein(gpmodel, mlp_reparam, data_generator,
                               out_dir, 
                               wasserstein_steps=(0, 400),
                               wasserstein_lr=0.02,
                               n_data=500)

    ckpt_path = os.path.join(out_dir, "ckpts/it-{}.ckpt".format(mapper_num_iters))
    if not os.path.exists(ckpt_path):
        w_hist = mapper.optimize(num_iters=mapper_num_iters, n_samples=100, lr=0.05, 
                save_ckpt_every=10, print_every=1, debug=True)
        path = os.path.join(out_dir, "wsr_values.log")
        np.savetxt(path, w_hist, fmt='%.6e')        
    else:
        print('Prior loaded from:', ckpt_path)
    
    return ckpt_path

def plot_wasserstein_optimization():
    ## Plot Wasserstein distance estimates (ie allof them, including intermediate)
    ## ===========================================================================

    wsr_intermediate_path = os.path.join(out_dir, 'wsr_intermediate_values.log')
    wsr_path = os.path.join(out_dir, 'wsr_values.log')
    if os.path.exists(wsr_path) and os.path.exists(wsr_intermediate_path):
        wsr_vals = np.loadtxt(wsr_intermediate_path)
        fig, axs = plt.subplots(2, 1, sharey=False, figsize=scale_figsize([1.5, 2]))

        ax = axs[0]
        iters = np.linspace(0,mapper_num_iters,  (len(wsr_vals)))
        ax.plot(iters, wsr_vals)
        ax.set_title('Intermediate Wasserstein estimates')
        ax.set_ylabel('Intermediate\nWasserstein value')
        ax.set_xlabel('Iteration')

        ax = axs[1]
        wsr_vals = np.loadtxt(wsr_path)
        ax.plot(np.arange(1, len(wsr_vals)+1), wsr_vals, '-ok')
        ax.set_title('Prior optimization')
        ax.set_ylabel(r'$W({NN}, {GP})$')
        ax.set_xlabel('Iteration')

        axs[0].set_ylim(axs[1].get_ylim()) ## makes axs[0] more readable
        plt.tight_layout()
        plt.savefig(f'wsr_intermediate_values_{suffix}.png', bbox_inches='tight')
        # plt.show()

def sample_priors(gpmodel, mlp_prior, mlp_reparam, ckpt_path, compute_wasserstein=True):

    gp_samples = gpmodel.sample_functions(torch.FloatTensor(Xtest),Nsamples)
    std_samples = mlp_prior.sample_functions(torch.FloatTensor(Xtest),Nsamples)
    mlp_reparam.load_state_dict(torch.load(ckpt_path))
    opt_samples = mlp_reparam.sample_functions(torch.FloatTensor(Xtest),Nsamples)

    if compute_wasserstein:
        # # Compute distance between GP and BNN (with fixed prior)
        distance_gp_to_bnnfixed = WassersteinDistance(mlp_prior, gpmodel, len(Xtest), 1)
        distance_gp_to_bnnfixed.wasserstein_optimisation(torch.FloatTensor(Xtest), 1000, n_steps=400)
        distance_gp_to_bnnfixed = distance_gp_to_bnnfixed.calculate(std_samples, gp_samples)
        # # Compute distance between GP and BNN (with matched prior)
        distance_gp_to_bnnmatched = WassersteinDistance(mlp_reparam, gpmodel, len(Xtest), 1)
        distance_gp_to_bnnmatched.wasserstein_optimisation(torch.FloatTensor(Xtest), 1000, n_steps=400)
        distance_gp_to_bnnmatched = distance_gp_to_bnnmatched.calculate(opt_samples, gp_samples)

        return gp_samples, std_samples, opt_samples, distance_gp_to_bnnfixed, distance_gp_to_bnnmatched

    return gp_samples, std_samples, opt_samples, np.nan, np.nan

def plot_prior(Xtest, gp_samples, std_samples, opt_samples):
    fig, axs = plt.subplots(1, 3, figsize=scale_figsize([3, 1]), sharey=True)

    ax = axs[0]
    plot_utils.plot_samples(Xtest, gp_samples, color='tab:blue', ax=ax)
    ax.set_title('GP prior')

    ax = axs[1]
    plot_utils.plot_samples(Xtest, std_samples, color='tab:green', ax=ax)
    ax.set_title('BNN prior (fixed)')
    # ax.text(.98, 0.02, r'\textbf{Distance} = %.2f' % distance_gp_to_std, horizontalalignment='right',
    #   verticalalignment='bottom', transform=ax.transAxes)

    ax = axs[2]
    plot_utils.plot_samples(Xtest, opt_samples, color='tab:orange')
    ax.set_title('BNN prior (GP-matched)')
    # ax.text(.98, 0.02, r'\textbf{Distance} = %.2f' % distance_gp_to_opt, horizontalalignment='right',
    #   verticalalignment='bottom', transform=ax.transAxes)
    ax.set_ylim(-4.5, 4.5)

    plt.tight_layout()
    plt.savefig(f'result_priors_{suffix}.png', bbox_inches='tight')
    # plt.show(); 

def plot_covariances(gp_samples, std_samples, opt_samples):
    fig, axs = plt.subplots(1, 3, figsize=scale_figsize([3, 1]), sharey=True)

    ax = axs[0]
    ax.imshow(np.cov(gp_samples.squeeze().detach()))
    ax.set_title('GP prior')

    ax = axs[1]
    ax.imshow(np.cov(std_samples.squeeze().detach()))
    ax.set_title('BNN prior (fixed)')

    ax = axs[2]
    ax.imshow(np.cov(opt_samples.squeeze().detach()))
    ax.set_title('BNN prior (GP-matched)')

    plt.tight_layout()
    plt.savefig(f'result_priors_cov_{suffix}.png', bbox_inches='tight')

def run_posterior_inference(X, y, Xtest):
    gp_fmu, gp_fs2 = gpmodel.predict_f(torch.from_numpy(Xtest).double(), full_cov=True)

    saved_dir = ''
    posterior_std_path = os.path.join(out_dir, 'posterior_std.npz')
    if not os.path.exists(posterior_std_path):
        print('Posterior for fixed prior not found. Starting sampling now.')
        std_net = RegressionNet(net, likelihood, std_prior, saved_dir, n_gpu=0)
        std_net.sample_multi_chains(X, y, **sampling_configs)
        _fmu, std_pred_var, std_samples = std_net.predict(Xtest, True)
        std_samples = std_samples.transpose(1,0,2)
        np.savez(posterior_std_path, std_pred_var=std_pred_var, std_samples=std_samples)
        print(f'Posterior for fixed prior saved in {posterior_std_path}')
    else:
        print(f'Loading posterior from {posterior_std_path}')
        _data = np.load(posterior_std_path)
        std_pred_var, std_samples = _data['std_pred_var'], _data['std_samples']


    posterior_opt_path = os.path.join(out_dir, 'posterior_opt.npz')
    if not os.path.exists(posterior_opt_path):
        print('Posterior for GP-matched prior not found. Starting sampling now.')
        opt_net = RegressionNet(net, likelihood, optim_prior, saved_dir, n_gpu=0)
        opt_net.sample_multi_chains(X, y, **sampling_configs)
        _fmu, opt_pred_var, opt_samples = opt_net.predict(Xtest, True)
        opt_samples = opt_samples.transpose(1,0,2)
        np.savez(posterior_opt_path, opt_pred_var=opt_pred_var, opt_samples=opt_samples)
        print(f'Posterior for GP-matched prior saved in {posterior_opt_path}')
    else:
        print(f'Loading posterior from {posterior_opt_path}')
        _data = np.load(posterior_opt_path)
        opt_pred_var, opt_samples = _data['opt_pred_var'], _data['opt_samples']

    return gp_fmu, gp_fs2, std_samples, opt_samples


def plot_posterior(Xtest, gp_fmu, gp_fs2, std_samples, opt_samples):
    fig, axs = plt.subplots(1, 3, figsize=scale_figsize([3, 1]), sharey=True)

    ax = axs[0]
    plot_utils.plot_data(X, y, ax)
    # # plt.plot(X, y, 'ok')
    plot_utils.plot_gp(Xtest, gp_fmu, gp_fs2, color='xkcd:bluish', ax=ax)
    ax.set_title('GP posterior')

    ax = axs[1]
    plot_utils.plot_data(X, y, ax)
    # ax.plot(X, y, 'ok', ms=2, zorder=5)
    plot_utils.plot_samples(Xtest, std_samples,  color='tab:green', ax=ax)
    ax.set_title('BNN posterior (standard)')

    ax = axs[2]
    plot_utils.plot_data(X, y, ax)
    # ax.plot(X, y, 'ok', ms=2, zorder=5)
    plot_utils.plot_samples(Xtest, opt_samples, color='tab:orange', ax=ax)
    ax.set_title('BNN posterior (GP-matched)')

    plt.tight_layout()
    plt.savefig(f'result_posteriors_{suffix}.png', bbox_inches='tight')




if __name__ == '__main__':
    X, y, Xtest = make_data(N, M)
    plot_dataset(X,y)

    gpmodel = build_gpmodel(X, y, leng, ampl, sn2)

    mlp_prior = GaussianMLPReparameterization(input_dim=1, output_dim=1, activation_fn=transfer_fn, 
        hidden_dims=[width]*depth)

    mlp_reparam = GaussianMLPReparameterization(input_dim=1, output_dim=1, activation_fn=transfer_fn, 
        hidden_dims=[width]*depth)


    ckpt_path = optimize_bnn_prior()

    net = MLP(input_dim=1, output_dim=1, hidden_dims=[width]*depth, activation_fn=transfer_fn)
    likelihood = LikGaussian(sn2)
    optim_prior = OptimGaussianPrior(ckpt_path)
    std_prior = FixedGaussianPrior(1.0)

    plot_wasserstein_optimization()

    gp_samples, std_samples, opt_samples, distance_gp_to_std, distance_gp_to_opt = sample_priors(gpmodel, mlp_prior, mlp_reparam, ckpt_path, False)

    plot_prior(Xtest, gp_samples, std_samples, opt_samples)

    plot_covariances(gp_samples, std_samples, opt_samples)

    gp_fmu, gp_fs2, std_samples, opt_samples = run_posterior_inference(X, y, Xtest)

    plot_posterior(Xtest, gp_fmu, gp_fs2, std_samples, opt_samples)

    plt.show()


