import torch
from torch.distributions import MultivariateNormal, Normal


class ScoreModel():
    #net: torch.nn.Module
    #alphas_cumprod: torch.tensor
    
    def __init__(self, net, alphas_cumprod, device):
        self.net = net
        self.alphas_cumprod = alphas_cumprod
        self.device = device
        
    def to(self, device):
        self.model = self.net.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.device=device

    def cpu(self, device):
        self.to('cpu')
        self.device = device('cpu')

    def cuda(self, device):
        self.to('cuda:0')
        self.device = device('cuda:0')


class NetReparametrized(torch.nn.Module):

    def __init__(self,
                 base_score_module: torch.nn.Module,
                 orthogonal_transformation: torch.Tensor):
        super().__init__()
        self.base_score_module = base_score_module
        self.orthogonal_transformation = orthogonal_transformation

    def forward(self, x, *args):
        x_back_to_basis = (self.orthogonal_transformation.T @ x.T).T
        score = self.base_score_module(x_back_to_basis, *args)
        return (self.orthogonal_transformation @ score.T).T


def build_extended_svd(A: torch.tensor):
    U, d, V = torch.linalg.svd(A, full_matrices=True)
    coordinate_mask = torch.ones_like(V[0])
    coordinate_mask[len(d):] = 0
    return U, d, coordinate_mask, V


def gaussian_posterior_batch_diagonal(y,
                                      likelihood_A,
                                      likelihood_bias,
                                      likelihood_precision_diag,
                                      prior_loc,
                                      prior_covar_diag):
    prior_precision_diag = 1 / prior_covar_diag
    posterior_precision_diag = prior_precision_diag.clone()
    posterior_precision_diag[likelihood_A != 0] += (likelihood_A[likelihood_A != 0] ** 2) * likelihood_precision_diag
    posterior_covariance_diag = 1 / posterior_precision_diag
    mean_residue = y - likelihood_bias
    mean_projected_residue = torch.zeros_like(prior_loc[0])
    mean_projected_residue[likelihood_A != 0] = likelihood_A[
                                                    likelihood_A != 0] * likelihood_precision_diag * mean_residue
    mean_prior = prior_precision_diag[None, :] * prior_loc
    posterior_mean = posterior_covariance_diag[None, :] * (mean_projected_residue[None, :] + mean_prior)
    return Normal(loc=posterior_mean,
                  scale=posterior_covariance_diag.unsqueeze(0).repeat(posterior_mean.shape[0], 1) ** .5)


def get_optimal_timesteps_from_singular_values(alphas_cumprod, singular_value, n_timesteps, var, jump=1, mode='equal'):
    distances = torch.unique(
        var * alphas_cumprod[None, :] - (1 - alphas_cumprod)[None, :] * singular_value[:, None] ** 2)
    optimal_distances = sorted(list(set((distances.abs().argmin(dim=-1, keepdims=True)).tolist())), key=lambda x: x)
    if 0 == optimal_distances[0]:
        optimal_distances = optimal_distances[1:]
    timesteps = [0]
    start_index = 0
    start_cumprod = alphas_cumprod[0] ** .5
    end = torch.where(alphas_cumprod ** .5 < 1e-2)[0][0].item()
    target_increase = (alphas_cumprod[start_index] ** .5 - alphas_cumprod[end] ** .5) / (
            n_timesteps - 1 - len(optimal_distances))
    last_value = start_cumprod
    for i in range(start_index + 1, end):
        if last_value - alphas_cumprod[i] ** .5 >= target_increase:
            timesteps.append(i)
            last_value = alphas_cumprod[i] ** .5
        elif i in optimal_distances:
            timesteps.append(i)
            last_value = alphas_cumprod[i] ** .5
    timesteps += torch.ceil(
        torch.linspace(timesteps[-1], len(alphas_cumprod) - 1, n_timesteps - len(timesteps) + 1)).tolist()[1:]
    return torch.tensor(timesteps).long()


# --------- Pour particle_filter


def generate_coefficients_ddim(
        alphas_cumprod,
        time_step,
        prev_time_step,
        eta):
    alphas_cumprod_t_1 = alphas_cumprod[prev_time_step] if prev_time_step >= 0 else 1
    alphas_cumprod_t = alphas_cumprod[time_step]

    noise = eta * (
                ((1 - alphas_cumprod_t_1) / (1 - alphas_cumprod_t)) * (1 - alphas_cumprod_t / alphas_cumprod_t_1)) ** .5

    coeff_sample = (alphas_cumprod_t_1 / alphas_cumprod_t) ** .5
    coeff_score = ((1 - alphas_cumprod_t_1 - noise ** 2) ** .5) - coeff_sample * ((1 - alphas_cumprod_t) ** .5)

    return noise, coeff_sample, coeff_score


def get_taus_from_singular_values(alphas_cumprod, timesteps, singular_values, var):
    distances = (var * alphas_cumprod[None, timesteps] - (1 - alphas_cumprod)[None, timesteps] * singular_values[:,
                                                                                                 None] ** 2)
    taus_indices = distances.abs().argmin(dim=1)
    taus = timesteps[taus_indices]
    return taus, taus_indices

####### de inverse_problem_utils.py

def gaussian_posterior(y,
                       likelihood_A,
                       likelihood_bias,
                       likelihood_precision,
                       prior_loc,
                       prior_covar):
    prior_precision_matrix = torch.linalg.inv(prior_covar)
    posterior_precision_matrix = prior_precision_matrix + likelihood_A.T @ likelihood_precision @ likelihood_A
    posterior_covariance_matrix = torch.linalg.inv(posterior_precision_matrix)
    posterior_mean = posterior_covariance_matrix @ (likelihood_A.T @ likelihood_precision @ (y - likelihood_bias) + prior_precision_matrix @ prior_loc)
    try:
        posterior_covariance_matrix = (posterior_covariance_matrix + posterior_covariance_matrix.T) / 2
        return MultivariateNormal(loc=posterior_mean, covariance_matrix=posterior_covariance_matrix)
    except ValueError:
        u, s, v = torch.linalg.svd(posterior_covariance_matrix, full_matrices=False)
        s = s.clip(1e-12, 1e6).real
        posterior_covariance_matrix = u.real @ torch.diag_embed(s) @ v.real
        posterior_covariance_matrix = (posterior_covariance_matrix + posterior_covariance_matrix.T) / 2
        return MultivariateNormal(loc=posterior_mean, covariance_matrix=posterior_covariance_matrix)
