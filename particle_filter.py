from typing import Tuple, List

import numpy as np
import torch
import tqdm
from torch.distributions import Categorical

from generative_models import ScoreModel, generate_coefficients_ddim, get_taus_from_singular_values, generate_coefficients_ddim

def predict(score_model: ScoreModel, particles: torch.Tensor, t: float, t_prev: float, eta: float,
            n_samples_per_gpu: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Generate coefficients for the generative model
    noise, coeff_sample, coeff_score = generate_coefficients_ddim(
        alphas_cumprod=score_model.alphas_cumprod.to(particles.device),
        time_step=t,
        prev_time_step=t_prev,
        eta=eta
    )

    # Compute epsilon_predicted using the neural network model
    if hasattr(score_model.net, 'device_ids'):
        batch_size = n_samples_per_gpu * len(score_model.net.device_ids)
        epsilon_predicted = []
        for batch_idx in range(0, particles.shape[0], batch_size):
            batch_particles = particles[batch_idx:batch_idx + batch_size].to(particles.device)
            epsilon_batch = score_model.net(batch_particles, t).cpu()
            epsilon_predicted.append(epsilon_batch)
        epsilon_predicted = torch.cat(epsilon_predicted, dim=0).to(particles.device)
    else:
        epsilon_predicted = score_model.net(particles, t).to(particles.device)

    # Compute the mean using the generated coefficients
    mean = coeff_sample * particles + coeff_score * epsilon_predicted

    return mean, noise, epsilon_predicted

def particle_filter(initial_particles: torch.Tensor,
                    observation: torch.Tensor,
                    score_model: ScoreModel,
                    coordinates_mask: torch.Tensor,
                    likelihood_diagonal: torch.Tensor,
                    var_observation: torch.Tensor,
                    timesteps: List[int],
                    eta: float = 1,
                    n_samples_per_gpu_inference: int = 16,
                    gaussian_var: float = 1e-2):
    n_particles, dim = initial_particles.shape
    alphas_cumprod = score_model.alphas_cumprod.to(initial_particles.device)

    log_weights = torch.zeros((n_particles,), device=initial_particles.device)
    particles = initial_particles

    # Compute scaling factors (taus) based on singular values and time steps
    taus, taus_indices = get_taus_from_singular_values(alphas_cumprod=alphas_cumprod,
                                                       timesteps=timesteps,
                                                       singular_values=likelihood_diagonal,
                                                       var=var_observation)

    coordinates_in_state = torch.where(coordinates_mask == 1)[0]
    always_free_coordinates = torch.where(coordinates_mask == 0)[0]
    rescaled_observation = (alphas_cumprod[taus]**.5) * observation / likelihood_diagonal

    # Iterate over time steps in reverse order using tqdm for progress visualization
    pbar = tqdm.tqdm(enumerate(zip(timesteps.tolist()[1:][::-1],
                                   timesteps.tolist()[:-1][::-1])),
                     desc='Particle Filter')

    for i, (t, t_prev) in pbar:
        # Generate prediction using the generative model
        predicted_mean, predicted_noise, eps = predict(score_model=score_model,
                                                       particles=particles,
                                                       t=t,
                                                       t_prev=t_prev,
                                                       eta=eta,
                                                       n_samples_per_gpu=n_samples_per_gpu_inference)

        # Sample ancestors based on log weights
        ancestors = Categorical(logits=log_weights).sample((n_particles,)) #Sampling ancestor indices for particle approximation given by equation (2.6) in the article
        new_particles = torch.empty_like(particles)
        new_log_weights = torch.zeros_like(log_weights)
        coordinates_to_filter = coordinates_in_state[taus < t_prev]
        exactly_observed_coordinates = coordinates_in_state[taus == t_prev]
        free_coordinates = torch.cat((coordinates_in_state[taus > t_prev], always_free_coordinates), dim=0)

        if len(coordinates_to_filter) > 0:
            # Rescale alphas for observed coordinates
            alpha_t_prev = alphas_cumprod[t_prev] / alphas_cumprod[taus[taus < t_prev]]
            diffused_observation = rescaled_observation[taus < t_prev] * (alpha_t_prev ** .5)
            observation_std = ((1 - (1 - gaussian_var) * alpha_t_prev) ** .5)
            top_predicted_mean = predicted_mean[ancestors, :][:, coordinates_to_filter]

            # Compute posterior for observed coordinates
            posterior_precision = (1 / (predicted_noise ** 2)) + (1 / (observation_std ** 2))
            posterior_mean = (1 / posterior_precision)[None, :] * (
                    diffused_observation[None, :] / (observation_std ** 2) + (
                    top_predicted_mean / (predicted_noise ** 2)))

            noise_top = torch.randn_like(posterior_mean)
            top_samples = posterior_mean + noise_top * ((1 / posterior_precision) ** .5)

            # Compute log weights
            log_integration_constant = -.5 * torch.linalg.norm(
                (top_predicted_mean - diffused_observation[None, :]) / (
                        (predicted_noise ** 2 + observation_std ** 2)[None, :] ** .5), dim=-1) ** 2

            alpha_t = alphas_cumprod[t] / alphas_cumprod[taus[taus < t_prev]]
            top_previous_particles = particles[ancestors][:, coordinates_to_filter].clone()
            previous_residue = (top_previous_particles - rescaled_observation[None, taus < t_prev] * (
                    alpha_t[None, :] ** .5)) / ((1 - (1 - gaussian_var) * alpha_t[None, :]) ** .5)
            log_forward_previous_likelihood = -.5 * torch.linalg.norm(previous_residue, dim=-1) ** 2

            new_log_weights += log_integration_constant - log_forward_previous_likelihood
            new_particles[:, coordinates_to_filter] = top_samples

        # ... (similar processing for other cases)

        log_weights = new_log_weights.clone()
        particles = new_particles.clone()

    return particles
