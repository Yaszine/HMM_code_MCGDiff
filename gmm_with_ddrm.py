import sys

import numpy as np

import torch



from torch import tensor, ones, eye, randn_like, randn, vstack, \
    manual_seed
from torch.distributions import MixtureSameFamily, MultivariateNormal, Categorical
from ddrm.denoising import efficient_generalized_steps
from ddrm.svd_replacement import GeneralH
from generative_models import ScoreModel, NetReparametrized, build_extended_svd, get_optimal_timesteps_from_singular_values, gaussian_posterior
from utils import EpsilonNetFromScore, Expandednet
from particle_filter import particle_filter
import torch


#Resampling part of the weights that correspond to pseudo-code Algorithm 1 
def ou_mixt(alpha_t, means, dim, weights):
    cat = Categorical(weights)

    ou_norm = MultivariateNormal(
        vstack(tuple((alpha_t**.5) * m for m in means)),
        eye(dim).repeat(len(means), 1, 1))
    return MixtureSameFamily(cat, ou_norm)





# posterior phi exactly calcultated with Bayes formula
def get_posterior(obs, prior, A, Sigma_y):
    modified_means = []
    modified_covars = []
    weights = []
    precision = torch.linalg.inv(Sigma_y)
    for loc, cov, weight in zip(prior.component_distribution.loc,
                                prior.component_distribution.covariance_matrix,
                                prior.mixture_distribution.probs):
        new_dist = gaussian_posterior(obs,
                                      A,
                                      torch.zeros_like(obs),
                                      precision,
                                      loc,
                                      cov)
        modified_means.append(new_dist.loc)
        modified_covars.append(new_dist.covariance_matrix)
        prior_x = MultivariateNormal(loc=loc, covariance_matrix=cov)
        residue = obs - A @ new_dist.loc
        log_constant = -(residue[None, :] @ precision @ residue[:, None]) / 2 + \
                       prior_x.log_prob(new_dist.loc) - \
                       new_dist.log_prob(new_dist.loc)
        weights.append(torch.log(weight).item() + log_constant)
    weights = torch.tensor(weights)  #weight of the mixture model
    weights = weights - torch.logsumexp(weights, dim=0)
    cat = Categorical(logits=weights)
    ou_norm = MultivariateNormal(loc=torch.stack(modified_means, dim=0),
                                 covariance_matrix=torch.stack(modified_covars, dim=0))
    return MixtureSameFamily(cat, ou_norm)

# generate the inverse problem parameters and the posterior distribution where build_extend is a function that returns the SVD of a matrix and x_ the unobserved data
def generate_measurement_equations(dim, dim_y, device, mixt):
    A = torch.randn((dim_y, dim))

    u, diag, coordinate_mask, v = build_extended_svd(A)
    diag = torch.sort(torch.rand_like(diag), descending=True).values

    A = u @ (torch.diag(diag) @ v[coordinate_mask == 1, :])
    init_sample = mixt.sample()
    std = (torch.rand((1,)))[0] * torch.ones(len(diag)) * max(diag)
    var_observations = std**2

    init_obs = A @ init_sample
    init_obs += randn_like(init_obs) * (var_observations**.5)
    Sigma_y = torch.diag(var_observations)
    posterior = get_posterior(init_obs, mixt, A, Sigma_y)
    return A, Sigma_y, u, diag, coordinate_mask, v, var_observations, posterior, init_obs


if __name__ == '__main__':
    use_gibbs = False
    n_samples = 3000
    device = torch.device('cpu')
    n_particles_mcg_diff = 1000
    T = 10
    delta = 0.01
    steps = int(T / delta)
    # Argument ligne de commande 
    save_folder = './steps'
    
    print(1)
    for ind_increase, (n_steps, eta) in enumerate(zip([20, 100], [.6, .85])): #[20, 100] [.6, .85]
        for ind_dim, dim in enumerate([80]): #[800, 80, 8] # marche pas avec 8 pb de dimension pour calcul mask svd 
            # setup of the inverse problem
            means = []
            for i in range(-2, 3):
                means += [torch.tensor([-8.*i, -8.*j]*(dim//2)) for j in range(-2, 3)]  #Annexe B.3 Choice of the mean of the prior
            weights = torch.randn(len(means))**2
            weights = weights / weights.sum()

            #Change pour pas avoir jax partial
            mixt = ou_mixt(1, means, dim, weights)
            def ou_mixt_fun(alpha_t):
                return ou_mixt(alpha_t, means, dim, weights)
            target_samples = mixt.sample((n_samples,)).cpu()
            for ind_ptg, dim_y in enumerate([4]): #[1, 2 ,4]

                for i in range(1): #20 for seed
                    #seed_num_inv_problem = (2**(ind_dim))*(3**(ind_ptg)*(5**(ind_increase))) + i 
                    seed = 3 # 4, 7, 8
                    manual_seed(seed) #seed_num_inv_problem
                    
                    try:
                        # Initialize the inverse problem
                        A, Sigma_y, u, diag, coordinate_mask, v, var_observations, posterior, init_obs = generate_measurement_equations(dim, dim_y, device, mixt)
                    except ValueError:
                        seed += 1
                        manual_seed(seed)
                        A, Sigma_y, u, diag, coordinate_mask, v, var_observations, posterior, init_obs = generate_measurement_equations(dim,
                                                                                                                                        dim_y,
                                                                                                                                        device,
                                                                                                                                        mixt)

                   
                
                    sigma_y = var_observations[0].item()**.5                    
                  
                    # setting up parameters for MCGDIFF where alphas and betas are used to compute samples with normal distribution 2.3 and 2.4 in the paper
                    betas = torch.linspace(.02, 1e-4, steps=999, device=device)
                    alphas_cumprod = torch.cumprod(tensor([1,] + [1 - beta for beta in betas]), dim=0) #they all ad alphas cumprod = 1 in the beginning
                    timesteps = torch.linspace(0, steps-1, n_steps, device=device).long()
                    adapted_timesteps = get_optimal_timesteps_from_singular_values(alphas_cumprod=alphas_cumprod,
                                                                                   singular_value=diag,
                                                                                   n_timesteps=n_steps,
                                                                                   var=var_observations[0].item(),
                                                                                   mode='else')
                    
                    
                    # score model
                    score_model = ScoreModel(NetReparametrized(base_score_module=EpsilonNetFromScore(ou_dist=ou_mixt_fun,
                                                                                                     alphas_cumprod=alphas_cumprod),
                                                               orthogonal_transformation=v),
                                             alphas_cumprod=alphas_cumprod,
                                             device=device)
                    score_model.net.device = device


                    # Getting posterior samples form nuts
                    posterior_samples = posterior.sample((n_samples,)).to(device)

                    n_particles = n_samples
                    initial_particles = randn(n_particles, dim).to(device)

                    H_funcs = GeneralH(H=A)
                    particles_ddrm = efficient_generalized_steps(x=torch.randn(n_samples, 1, 1, dim).to(device),
                                                                        b=betas.to(device),
                                                                        seq=adapted_timesteps[:-1].tolist(),
                                                                        model=Expandednet(base_net=score_model.net.base_score_module,
                                                                                          expanded_size=(1, 1, dim)),
                                                                        y_0=init_obs[None, :].to(device),
                                                                        H_funcs=H_funcs,
                                                                        sigma_0=var_observations[0].item()**.5,
                                                                        etaB=1,
                                                                        etaA=.85,
                                                                        etaC=1,
                                                                        classes=None,
                                                                        cls_fn=None)[0][-1].flatten(1, 3).cpu()

                    
                    particles_mcg_diff = []
                    n_batches_mcg_diff = n_samples // n_particles_mcg_diff
                    for batch_initial_particles in initial_particles.reshape(n_batches_mcg_diff,
                                                                             n_particles_mcg_diff,
                                                                             -1):
                        # MCGDIFF
                        particle_cloud = particle_filter(
                            initial_particles=batch_initial_particles,
                            observation=(u.T @ init_obs).to(device),
                            score_model=score_model,
                            likelihood_diagonal=diag,
                            coordinates_mask=coordinate_mask,
                            var_observation=var_observations[0].item(),
                            timesteps=adapted_timesteps,
                            eta=eta
                        )
                        particles_mcg_diff.append((v.T @ particle_cloud.T).T.cpu())
                    particles_mcg_diff = torch.cat(particles_mcg_diff, dim=0)
                    
                    
                    data = {
                        "seed": seed,
                        "sigma_y": sigma_y,
                        "D_X": dim,
                        "D_Y": dim_y,
                        "prior": mixt.sample((n_samples,)).cpu().numpy(),
                        "posterior": posterior_samples.cpu().numpy(),
                        "DDRM": particles_ddrm.cpu().numpy(),
                        "MCG_DIFF": particles_mcg_diff.cpu().numpy()
                    }
                    np.savez(f'{save_folder}/{dim}_{dim_y}_{dim_y}_{25}_{seed}_{n_steps}.npz',
                             **data)
