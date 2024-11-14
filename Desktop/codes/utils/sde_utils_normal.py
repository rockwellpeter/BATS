import enum
import math
import random
import numpy as np
import torch
import abc
from tqdm import tqdm
import torchvision.utils as tvutils
import os
from scipy import integrate
import matplotlib.pyplot as plt
from torchvision.transforms import transforms



class SDE(abc.ABC):
    def __init__(self, T, device=None):
        self.T = T
        self.dt = 1 / T
        self.device = device

    @abc.abstractmethod
    def drift(self, x, t):
        pass

    @abc.abstractmethod
    def dispersion(self, x, t):
        pass

    @abc.abstractmethod
    def sde_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def ode_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def score_fn(self, x, t):
        pass

    ################################################################################

    def forward_step(self, x, t):
        return x + self.drift(x, t) + self.dispersion(x, t)

    def reverse_sde_step_mean(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t)

    def reverse_sde_step_mean_contractive(self, x, score, t):
        return x - self.sde_reverse_drift_contractive(x, score, t)

    def reverse_sde_step(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t) - self.dispersion(x, t)

    def reverse_sde_step_contractive(self, x, score, t):
        return x - self.sde_reverse_drift_contractive(x, score, t) - self.dispersion(x, t)

    def reverse_ode_step(self, x, score, t):
        return x - self.ode_reverse_drift(x, score, t)

    def forward(self, x0, T=-1):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

        return x

    def reverse_sde(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_sde_step(x, score, t)

        return x

    def reverse_ode(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_ode_step(x, score, t)

        return x


#############################################################################


class  IRSDE(SDE):
    '''
    Let timestep t start from 1 to T, state t=0 is never used
    '''
    def __init__(self, max_sigma, T=100, schedule='cosine', eps=0.01,  device=None):
        super().__init__(T, device)
        self.max_sigma = max_sigma / 255 if max_sigma > 1 else max_sigma
        self.max_sigma = max_sigma / 255 if max_sigma > 1 else max_sigma
        self._initialize(self.max_sigma, T, schedule, eps)

    def _initialize(self, max_sigma, T, schedule, eps=0.01):

        def constant_theta_schedule(timesteps, v=1.):
            """
            constant schedule
            """
            print('constant schedule')
            timesteps = timesteps + 1 # T from 1 to 100
            return torch.ones(timesteps, dtype=torch.float32)

        def linear_theta_schedule(timesteps):
            """
            linear schedule
            """
            print('linear schedule')
            timesteps = timesteps + 1 # T from 1 to 100
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


        def cosine_theta_schedule(timesteps, s = 0.008):
            """
            cosine schedule
            """
            print('cosine schedule')
            timesteps = timesteps + 2 # for truncating from 1 to -1
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:-1]
            return betas


        def cosine_theta_schedule_contractive(timesteps, s = 0.008):
            """
            cosine schedule contractive
            """
            print('cosine schedule')
            timesteps = timesteps + 2 # for truncating from 1 to -1
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            # reverse
            alphas_cumprod = torch.flip(alphas_cumprod, dims=[0])
            
            betas = 1 - alphas_cumprod[1:-1]
            return betas


        def get_thetas_cumsum(thetas):
            return torch.cumsum(thetas, dim=0)

        def get_sigmas(thetas):
            return torch.sqrt(max_sigma**2 * 2 * thetas)

        def get_sigma_bars(thetas_cumsum):
            return torch.sqrt(max_sigma**2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))
            
        if schedule == 'cosine':
            thetas = cosine_theta_schedule(T)
        elif schedule == 'linear':
            thetas = linear_theta_schedule(T)
        elif schedule == 'constant':
            thetas = constant_theta_schedule(T)
        elif schedule == 'cosine-contractive':
            thetas = cosine_theta_schedule_contractive(T)
        else:
            print('Not implemented such schedule yet!!!')

        sigmas = get_sigmas(thetas)
        thetas_cumsum = get_thetas_cumsum(thetas) - thetas[0] # for that thetas[0] is not 0
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = get_sigma_bars(thetas_cumsum)
        
        self.thetas = thetas.to(self.device)
        self.sigmas = sigmas.to(self.device)
        self.thetas_cumsum = thetas_cumsum.to(self.device)
        self.sigma_bars = sigma_bars.to(self.device)

        self.mu = 0.
        self.model = None


        # epsilon scaling

        # linear schedule
        # start = 1.011 - self.eps_scaler * ((self.T - 1) / 2)
        # self.sampling_scaler = [(start + i * self.eps_scaler) for i in range(0, self.T)]
        # start = 1.0041
        # self.eps_scaler = 0.0002
        # self.sampling_scaler = [(start + i * self.eps_scaler) for i in range(0, self.T)]


        # uniform schedule
        # self.eps_scaler = 1.006
        # self.eps_scaler = 1
        self.eps_scaler = 1.1
        self.sampling_scaler = [self.eps_scaler for i in range(0, self.T)]

    #####################################

    # set mu for different cases
    def set_mu(self, mu):
        self.mu = mu
        # self.mu = torch.zeros_like(mu)

    # set score model for reverse process
    def set_model(self, model):
        self.model = model

    #####################################

    def mu_bar(self, x0, t):
        return  x0 * torch.exp(-self.thetas_cumsum[t] * self.dt)
        # return self.mu + (x0 - self.mu) * torch.exp(-self.thetas_cumsum[t] * self.dt)

    
    def mu_bar_contractive(self, x0, t):
        return  x0 * torch.exp(self.thetas_cumsum[t] * self.dt)
        # return self.mu + (x0 - self.mu) * torch.exp(-self.thetas_cumsum[t] * self.dt)



    def sigma_bar(self, t):
        return self.sigma_bars[t]

    def drift(self, x, t):
        return self.thetas[t] * (self.mu - x) * self.dt

    # def sde_reverse_drift(self, x, score, t):
    #     return (self.thetas[t] * (self.mu - x) - self.sigmas[t]**2 * score) * self.dt

    def sde_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (-x) - self.sigmas[t]**2 * score) * self.dt
        
    def sde_reverse_drift_contractive(self, x, score, t):
        return (-self.thetas[t] * (-x) - self.sigmas[t]**2 * score) * self.dt

    def ode_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (self.mu - x) - 0.5 * self.sigmas[t]**2 * score) * self.dt

    def dispersion(self, x, t):
        return self.sigmas[t] * (torch.randn_like(x) * math.sqrt(self.dt)).to(self.device)

    def get_score_from_noise(self, noise, t):
        return -noise / self.sigma_bar(t)

    def score_fn(self, x, t, **kwargs):
        # need to pre-set mu and score_model
        noise = self.model(x, self.mu, t, **kwargs)
        return self.get_score_from_noise(noise, t)
    
    # def score_fn_epsilon_scaling(self, x, t, **kwargs):
    #     # need to pre-set mu and score_model
    #     # noise = self.model(x, self.mu, t, **kwargs)
    #     noise = self.noise_fn_epsilon_scaling(x, t, **kwargs)
    #     return self.get_score_from_noise(noise, t)
    
    def score_fn_epsilon_scaling(self, x, t, **kwargs):
        # need to pre-set mu and score_model
        noise = self.model(x, self.mu, t, **kwargs)
        score = self.get_score_from_noise(noise, t)
        score = score/self.sampling_scaler[t-1]
        return score

    def noise_fn(self, x, t, **kwargs):
        # need to pre-set mu and score_model
        return self.model(x, self.mu, t, **kwargs)

    def noise_fn_epsilon_scaling(self, x, t, **kwargs):
        # need to pre-set mu and score_model
        noise = self.model(x, self.mu, t, **kwargs)
        ind = t
        noise = noise/self.sampling_scaler[ind-1]
        print(self.sampling_scaler[ind-1])
        return noise

    # optimum x_{t-1}
    def reverse_optimum_step(self, xt, x0, t):
        A = torch.exp(-self.thetas[t] * self.dt)
        B = torch.exp(-self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-self.thetas_cumsum[t-1] * self.dt)

        term1 = A * (1 - C**2) / (1 - B**2)
        term2 = C * (1 - A**2) / (1 - B**2)

        # return term1 * (xt - self.mu) + term2 * (x0 - self.mu) + self.mu
        return term1 * (xt) + term2 * (x0)

    def reverse_optimum_std(self, t):
        A = torch.exp(-2*self.thetas[t] * self.dt)
        B = torch.exp(-2*self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-2*self.thetas_cumsum[t-1] * self.dt)

        posterior_var = (1 - A) * (1 - C) / (1 - B)
        # return torch.sqrt(posterior_var)

        min_value = (1e-20 * self.dt).to(self.device)
        log_posterior_var = torch.log(torch.clamp(posterior_var, min=min_value))
        return (0.5 * log_posterior_var).exp() * self.max_sigma

    def reverse_posterior_step(self, xt, noise, t):
        x0 = self.get_init_state_from_noise(xt, noise, t)
        mean = self.reverse_optimum_step(xt, x0, t)
        std = self.reverse_optimum_std(t)
        return mean + std * torch.randn_like(xt)

    def sigma(self, t):
        return self.sigmas[t]

    def theta(self, t):
        return self.thetas[t]

    def get_real_noise(self, xt, x0, t):
        return (xt - self.mu_bar(x0, t)) / self.sigma_bar(t)

    def get_real_score(self, xt, x0, t):
        return -(xt - self.mu_bar(x0, t)) / self.sigma_bar(t)**2

    def get_init_state_from_noise(self, xt, noise, t):
        A = torch.exp(self.thetas_cumsum[t] * self.dt)
        # print(self.thetas_cumsum)
        # return (xt - self.mu - self.sigma_bar(t) * noise) * A + self.mu
        return (xt - self.sigma_bar(t) * noise) * A

    # mu =0 for vanilla ddpm
    def get_noise_from_init_state(self, state, xt, t):
        A = torch.exp(self.thetas_cumsum[t] * self.dt) # 计算 A
        sigma_bar_t = self.sigma_bar(t) # 计算 sigma_bar(t)

        # 根据推导公式反推 noise
        # noise = (xt - self.mu - (state - self.mu) / A) / sigma_bar_t
        noise = (xt - (state) / A) / sigma_bar_t

        return noise


    # forward process to get x(T) from x(0)
    def forward(self, x0, T=-1, save_dir='forward_state'):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{t}.png', normalize=False)
        return x    
    
    def reverse_sde_visual(self, xt, current_step, T=-1, save_states=False, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t, **kwargs)
            x = self.reverse_sde_step(x, score, t)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x

    def reverse_sde_visual_epsilon_scaling(self, xt, current_step, T=-1, save_states=False, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn_epsilon_scaling(x, t, **kwargs)
            x = self.reverse_sde_step(x, score, t)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x

    def reverse_sde_visual_contractive(self, xt, current_step, T=-1, save_states=False, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t, **kwargs)
            x = self.reverse_sde_step_contractive(x, score, t)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x

    def reverse_sde_visual_x0(self, xt, current_step, T=-1, save_states=False, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            #noise from init_state
            # noise_fn() actually predicts x0 at present
            state =self.noise_fn(x, t, **kwargs)
            noise = self.get_noise_from_init_state(state, x, t)
            score = self.get_score_from_noise(noise, t)
            # score = self.score_fn(x, t, **kwargs)
            x = self.reverse_sde_step(x, score, t)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x
    
    def reverse_sde_visual_x0_es(self, xt, current_step, T=-1, save_states=False, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            #noise from init_state
            # noise_fn() actually predicts x0 at present
            state =self.noise_fn(x, t, **kwargs)
            # state = state/self.sampling_scaler[t-1]
            noise = self.get_noise_from_init_state(state, x, t)
            noise = noise/self.sampling_scaler[t-1]
            score = self.get_score_from_noise(noise, t)
            # score = self.score_fn(x, t, **kwargs)
            x = self.reverse_sde_step(x, score, t)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x


    def reverse_sde_visual_x0_c(self, xt, current_step, T=-1, save_states=False, condition=None, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            #noise from init_state
            # noise_fn() actually predicts x0 at present
            state_and_c =self.noise_fn(x, t, **kwargs)
            state = state_and_c * 2 - condition
            noise = self.get_noise_from_init_state(state, x, t)
            score = self.get_score_from_noise(noise, t)
            # score = self.score_fn(x, t, **kwargs)
            x = self.reverse_sde_step(x, score, t)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x


    def reverse_sde_visual_traditional_reg(self, xt, current_step=-1, T=-1, save_states=False, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(T, T + 1))):
            #noise from init_state
            # noise_fn() actually predicts x0 at present
            state =self.noise_fn(x, t, **kwargs)
            x = state
            # noise = self.get_noise_from_init_state(state, xt, t)
            # score = self.get_score_from_noise(noise, t)
            # # score = self.score_fn(x, t, **kwargs)
            # x = self.reverse_sde_step(x, score, t)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x

    def reverse_sde_visual_traditional_reg_c(self, xt, current_step=-1, T=-1, save_states=False, condition=None, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(T, T + 1))):
            #noise from init_state
            # noise_fn() actually predicts x0 at present
            state_and_c =self.noise_fn(x, t, **kwargs)
            state = state_and_c * 2 -condition
            x = state
            # noise = self.get_noise_from_init_state(state, xt, t)
            # score = self.get_score_from_noise(noise, t)
            # # score = self.score_fn(x, t, **kwargs)
            # x = self.reverse_sde_step(x, score, t)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x

        

    # def reverse_sde(self, xt, T=-1, save_states=False, save_dir='sde_state', **kwargs):
    # # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
    #     T = self.T if T < 0 else T
    #     x = xt.clone()
    #     for t in tqdm(reversed(range(1, T + 1))):
    #         score = self.score_fn(x, t, **kwargs)
    #         x = self.reverse_sde_step(x, score, t)

    #         if save_states: # only consider to save 100 images
    #             interval = self.T // 100
    #             if t % interval == 0:
    #                 idx = t // interval
    #                 os.makedirs(save_dir, exist_ok=True)
    #                 tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

    #     return x

    def reverse_ode(self, xt, T=-1, save_states=False, save_dir='ode_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t, **kwargs)
            x = self.reverse_ode_step(x, score, t)

            if save_states: # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x

    def reverse_posterior(self, xt, T=-1, save_states=False, save_dir='posterior_state', **kwargs):
        T = self.T if T < 0 else T

        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            noise = self.noise_fn(x, t, **kwargs)
            x = self.reverse_posterior_step(x, noise, t)

            if save_states: # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x


    # sample ode using Black-box ODE solver (not used)
    def ode_sampler(self, xt, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3,):
        shape = xt.shape

        def to_flattened_numpy(x):
          """Flatten a torch tensor `x` and convert it to numpy."""
          return x.detach().cpu().numpy().reshape((-1,))

        def from_flattened_numpy(x, shape):
          """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
          return torch.from_numpy(x.reshape(shape))

        def ode_func(t, x):
            t = int(t)
            x = from_flattened_numpy(x, shape).to(self.device).type(torch.float32)
            score = self.score_fn(x, t)
            drift = self.ode_reverse_drift(x, score, t)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(ode_func, (self.T, eps), to_flattened_numpy(xt),
                                     rtol=rtol, atol=atol, method=method)

        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(self.device).type(torch.float32)

        return x

    def optimal_reverse(self, xt, x0, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            x = self.reverse_optimum_step(x, x0, t)

        return x

    ################################################################

    def weights(self, t):
        return torch.exp(-self.thetas_cumsum[t] * self.dt)

    # sample states for training
    # def generate_random_states(self, x0, mu):
    #     x0 = x0.to(self.device)
    #     mu = mu.to(self.device)

    #     self.set_mu(mu)

    #     batch = x0.shape[0]

    #     timesteps = torch.randint(1, self.T + 1, (batch, 1, 1, 1)).long()

    #     state_mean = self.mu_bar(x0, timesteps)
    #     noises = torch.randn_like(state_mean)
    #     noise_level = self.sigma_bar(timesteps)
    #     noisy_states = noises * noise_level + state_mean

    #     # return timesteps, noisy_states.to(torch.float32)
    #     return timesteps, noisy_states.to(torch.float32), noises

    def generate_random_states(self, x0, mu):
        x0 = x0.to(self.device)
        mu = mu.to(self.device)

        self.set_mu(mu)

        batch = x0.shape[0]

        timesteps = torch.randint(1, self.T + 1, (batch, 1, 1, 1)).long()

        # test
        # timesteps = 30
        # timesteps_fix = 70 * torch.ones_like(timesteps).long()
        # timesteps_fix = (self.T) * torch.ones_like(timesteps).long()
        # timesteps_fix = (2) * torch.ones_like(timesteps).long()
        # state_mean = self.mu_bar(x0, timesteps_fix)
        # noises = torch.randn_like(state_mean)
        # noise_level = self.sigma_bar(timesteps_fix)
        # for i in range(1,101):
        #     irsde_noise_level = []
        #     timesteps_fix = (i) * torch.ones_like(timesteps).long()
        #     state_mean = self.mu_bar(x0, timesteps_fix)
        #     noises = torch.randn_like(state_mean)
        #     noise_level = self.sigma_bar(timesteps_fix)
        #     print(noise_level[0, 0, 0, 0].item())
        #     irsde_noise_level.append(noise_level[0, 0, 0, 0].item())

        # # 保存列表到 .txt 文件
        # with open('irsde_noise_level.txt', 'w') as f:
        #     for level in irsde_noise_level:
        #         f.write(f"{level}\n")  # 每个 noise level 写一行
        # exit(0)
        
        noisy_states = noises * noise_level + state_mean
        
        import numpy as np
        from torchvision.utils import make_grid
        def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
            """
            Converts a torch Tensor into an image Numpy array
            Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
            Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
            """
            tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
            tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
            n_dim = tensor.dim()
            if n_dim == 4:
                n_img = len(tensor)
                img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
                img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
            elif n_dim == 3:
                img_np = tensor.numpy()
                img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
            elif n_dim == 2:
                img_np = tensor.numpy()
            else:
                raise TypeError(
                    "Only support 4D, 3D and 2D tensor. But received with dimension: {:d}".format(
                        n_dim
                    )
                )
            if out_type == np.uint8:
                img_np = (img_np * 255.0).round()
                # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
            return img_np.astype(out_type)
        
        # import cv2
        # noisy_states = tensor2img(noisy_states)
        # x0 = tensor2img(x0)
        # cv2.imwrite('/home/proj/image-restoration-sde-main/codes/temp/noisy_maxsigma30.png', noisy_states)
        # cv2.imwrite('/home/proj/image-restoration-sde-main/codes/temp/x0.png', x0)
        
        # exit()

        state_mean = self.mu_bar(x0, timesteps)
        noises = torch.randn_like(state_mean)
        noise_level = self.sigma_bar(timesteps)
        noisy_states = noises * noise_level + state_mean

        return timesteps, noisy_states.to(torch.float32), noises


    def generate_random_states_ddpm_ip(self, x0, mu):
        self.input_pertub = 0.1
        x0 = x0.to(self.device)
        mu = mu.to(self.device)

        self.set_mu(mu)

        batch = x0.shape[0]

        timesteps = torch.randint(1, self.T + 1, (batch, 1, 1, 1)).long()

      #######################################
        state_mean = self.mu_bar(x0, timesteps)
        # if noises is None:
        #     noises = torch.randn_like(state_mean)
        # noises = noises + self.input_pertub * torch.randn_like(noises)
        noises = torch.randn_like(state_mean)
        noises = noises + self.input_pertub * torch.randn_like(state_mean)


        noise_level = self.sigma_bar(timesteps)
        noisy_states = noises * noise_level + state_mean
    


        return timesteps, noisy_states.to(torch.float32), noises


    def generate_random_states_contractive(self, x0, mu):
        x0 = x0.to(self.device)
        mu = mu.to(self.device)

        self.set_mu(mu)

        batch = x0.shape[0]

        timesteps = torch.randint(1, self.T + 1, (batch, 1, 1, 1)).long()

        timesteps_fix = (self.T) * torch.ones_like(timesteps).long()
        # state_mean = self.mu_bar_contractive(x0, timesteps_fix)
        state_mean = self.mu_bar(x0, timesteps_fix)
        noises = torch.randn_like(state_mean)
        noise_level = self.sigma_bar(timesteps_fix)
        noisy_states = noises * noise_level + state_mean

        return timesteps, noisy_states.to(torch.float32), noises


    # sample states for training
    def generate_random_states_traditional_reg(self, x0, mu):
        x0 = x0.to(self.device)
        mu = mu.to(self.device)

        self.set_mu(mu)

        batch = x0.shape[0]

        timesteps = torch.randint(self.T, self.T + 1, (batch, 1, 1, 1)).long()

        state_mean = self.mu_bar(x0, timesteps)
        noises = torch.randn_like(state_mean)
        noise_level = self.sigma_bar(timesteps)
        noisy_states = noises * noise_level + state_mean
    
        # return timesteps, noisy_states.to(torch.float32)
        return timesteps, noisy_states.to(torch.float32), noises

    # def noise_state(self, tensor):
    #     return tensor + torch.randn_like(tensor) * self.max_sigma


    def save_noisy_images(self, noisy_states, filename='noisy_image.png'):
        import torch
        import matplotlib.pyplot as plt
        import numpy as np
        # 如果 noisy_states 是 GPU 张量，先将其转到 CPU
        noisy_states = noisy_states.cpu()

        # 将 noisy_states 转换为 NumPy 数组
        noisy_states_np = noisy_states.squeeze().numpy()  # 去掉 batch 维度（假设是单张图）

        # 如果图像是多通道 (如 RGB)，调整维度 (C, H, W) -> (H, W, C)
        if noisy_states_np.ndim == 3 and noisy_states_np.shape[0] == 3:
            noisy_states_np = np.transpose(noisy_states_np, (1, 2, 0))

        # 创建绘图
        plt.imshow(noisy_states_np, cmap='gray')  # 对于灰度图像使用 cmap='gray'
        plt.axis('off')  # 不显示坐标轴

        # 保存图像
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()


    def noise_state(self, tensor):
        # return torch.randn_like(tensor) * self.max_sigma
        return tensor + torch.randn_like(tensor) * self.max_sigma
        # return tensor + torch.ones_like(tensor) - (torch.randn_like(tensor) * self.max_sigma)
    


    def noise_t_state(self, tensor, t):
        batch = tensor.shape[0]
        # timesteps = torch.full((batch, 1, 1, 1), self.T).long()
        timesteps = torch.full((batch, 1, 1, 1), t).long()
        tensor = tensor.to(self.device)
        noise_level = self.sigma_bar(timesteps).to(self.device)
        alpha_bar = torch.exp(-self.thetas_cumsum[t] * self.dt).to(self.device)
        
        return tensor * alpha_bar + torch.randn_like(tensor) * noise_level


    def inference_multi_steploss(self, xt, x_gt, name, begin=1, end=100, **kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        multi_step_mask = []
        multi_step_unmask = []

        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)
        print(self.mu.shape)
        import matplotlib.pyplot as plt

        mask = torch.any(x_gt.float() != self.mu.float(), dim=1).unsqueeze(1) #[1, 1, 256, 256]

        # 通过 expand 扩展到 [1, 3, 256, 256]
        mask = mask.expand(-1, 3, -1, -1)

        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t, **kwargs)
            x = self.reverse_sde_step(x, score, t)
            x0_bar = self.get_init_state_from_noise(x, self.model(x, self.mu, t, **kwargs), t)  
            loss_multi_mask = loss_fn(x0_bar[mask], x_gt[mask], reduction='none')
            loss_multi_mask = einops.reduce(loss_multi_mask, 'b ... -> b (...)', 'mean')
            multi_step_mask.append(loss_multi_mask.mean().item())
            print(loss_multi_mask.mean().item())
            loss_multi_unmask = loss_fn(x0_bar[~mask], x_gt[~mask], reduction='none')
            loss_multi_unmask = einops.reduce(loss_multi_unmask, 'b ... -> b (...)', 'mean')
            multi_step_unmask.append(loss_multi_unmask.mean().item())
            print(loss_multi_unmask.mean().item())   
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        # 绘制各条曲线
        plt.plot(multi_step_mask[begin:end], label='Mask Loss', color='green', linestyle='-', linewidth=1.5)  # 第一条曲线
        plt.plot(multi_step_unmask[begin:end], label='Unmask Loss', color='blue', linestyle='-', linewidth=1.5)  # 第二条曲线
        plt.plot(multi_step[begin:end], label='Total Loss', color='red', linestyle='-', linewidth=1.5)  # 第三条曲线

        # 添加图例
        plt.legend()
        
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Multi-step Loss Curve')
        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp_seperate/multi_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突
        exit()
        #     mask = torch.any(x_gt != self.mu, dim=-1)    
        #     loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
        #     loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
        #     multi_step.append(loss_multi.mean().item())
        #     print(loss_multi.mean().item())
        # # 画图部分
        # plt.plot(multi_step[begin:end])
        # plt.xlabel('Time Step')
        # plt.ylabel('Loss')
        # plt.title('Multi-step Loss Curve')
        # plt.grid(True)
        # plt.show()
        # # 保存图像
        # plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp/multi_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        # plt.close()  # 关闭图形，防止后续绘图冲突
        # exit()
        # return x
    
    def inference_multi_steploss_epsilon_scaling(self, xt, x_gt, name, begin=0, end=100, **kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t, **kwargs)
            x = self.reverse_sde_step(x, score, t)
            x0_bar = self.get_init_state_from_noise(x, self.model(x, self.mu, t, **kwargs), t)
            
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        # 画图部分
        plt.plot(multi_step[begin:end])
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Multi-step Loss Curve')
        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp/multi_step_{name}_linear_{self.eps_scaler}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突
        exit()
        return x

    def inference_single_steploss(self, xt, x_gt, name, begin=70, end=100, **kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        multi_step_mask = []
        multi_step_unmask = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)

        mask = torch.any(x_gt.float() != self.mu.float(), dim=1).unsqueeze(1) #[1, 1, 256, 256]

        # 通过 expand 扩展到 [1, 3, 256, 256]
        mask = mask.expand(-1, 3, -1, -1)

        for t in tqdm(reversed(range(1, T + 1))):
            
            xt_temp = self.noise_t_state(x_gt, t)
            # score = self.score_fn(x, t, **kwargs)
            # x = self.reverse_sde_step(x, score, t)
            x0_bar = self.get_init_state_from_noise(xt_temp, self.model(xt_temp, self.mu, t, **kwargs), t)
            
            loss_multi_mask = loss_fn(x0_bar[mask], x_gt[mask], reduction='none')
            loss_multi_mask = einops.reduce(loss_multi_mask, 'b ... -> b (...)', 'mean')
            multi_step_mask.append(loss_multi_mask.mean().item())
            print(loss_multi_mask.mean().item())
            loss_multi_unmask = loss_fn(x0_bar[~mask], x_gt[~mask], reduction='none')
            loss_multi_unmask = einops.reduce(loss_multi_unmask, 'b ... -> b (...)', 'mean')
            multi_step_unmask.append(loss_multi_unmask.mean().item())
            print(loss_multi_unmask.mean().item())   
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        # 绘制各条曲线
        plt.plot(multi_step_mask[begin:end], label='Mask Loss', color='green', linestyle='-', linewidth=1.5)  # 第一条曲线
        plt.plot(multi_step_unmask[begin:end], label='Unmask Loss', color='blue', linestyle='-', linewidth=1.5)  # 第二条曲线
        plt.plot(multi_step[begin:end], label='Total Loss', color='red', linestyle='-', linewidth=1.5)  # 第三条曲线

        # 添加图例
        plt.legend()
        
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Single-step Loss Curve')
        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp_seperate/single_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突
        exit()

    def inference_single_steploss_x0(self, xt, x_gt, name, begin=70, end=100,**kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        multi_step_mask = []
        multi_step_unmask = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)

        mask = torch.any(x_gt.float() != self.mu.float(), dim=1).unsqueeze(1) #[1, 1, 256, 256]

        # 通过 expand 扩展到 [1, 3, 256, 256]
        mask = mask.expand(-1, 3, -1, -1)

        for t in tqdm(reversed(range(1, T + 1))):     
            xt_temp = self.noise_t_state(x_gt, t)
            x0_bar = self.noise_fn(xt_temp, t, **kwargs)
            # score = self.score_fn(x, t, **kwargs)
            # # x = self.reverse_sde_step(x, score, t)
            # x0_bar = self.get_init_state_from_noise(xt_temp, self.model(xt_temp, self.mu, t, **kwargs), t)
            
            loss_multi_mask = loss_fn(x0_bar[mask], x_gt[mask], reduction='none')
            loss_multi_mask = einops.reduce(loss_multi_mask, 'b ... -> b (...)', 'mean')
            multi_step_mask.append(loss_multi_mask.mean().item())
            print(loss_multi_mask.mean().item())
            loss_multi_unmask = loss_fn(x0_bar[~mask], x_gt[~mask], reduction='none')
            loss_multi_unmask = einops.reduce(loss_multi_unmask, 'b ... -> b (...)', 'mean')
            multi_step_unmask.append(loss_multi_unmask.mean().item())
            print(loss_multi_unmask.mean().item())   
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        # 绘制各条曲线
        plt.plot(multi_step_mask[begin:end], label='Mask Loss', color='green', linestyle='-', linewidth=1.5)  # 第一条曲线
        plt.plot(multi_step_unmask[begin:end], label='Unmask Loss', color='blue', linestyle='-', linewidth=1.5)  # 第二条曲线
        plt.plot(multi_step[begin:end], label='Total Loss', color='red', linestyle='-', linewidth=1.5)  # 第三条曲线

        # 添加图例
        plt.legend()
        
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Single-step Loss Curve')
        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp_seperate/single_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突
        exit()

    def inference_single_steploss_x0_c(self, xt, x_gt, name, condition=None, begin=70, end=100,**kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)
        for t in tqdm(reversed(range(1, T + 1))):     
            xt_temp = self.noise_t_state(x_gt, t)
            x0_bar_and_c = self.noise_fn(xt_temp, t, **kwargs)
            x0_bar = x0_bar_and_c * 2 - condition
            # score = self.score_fn(x, t, **kwargs)
            # # x = self.reverse_sde_step(x, score, t)
            # x0_bar = self.get_init_state_from_noise(xt_temp, self.model(xt_temp, self.mu, t, **kwargs), t)
            
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        
        # 画图部分
        plt.plot(multi_step[begin:end])
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Single-step Loss Curve')
        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp/single_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突
        exit()
        return x
    

    def inference_multi_steploss_traditional_reg(self, xt, x_gt, name, begin=1, end=100, **kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)
        for t in tqdm(reversed(range(T, T + 1))):
            # 此处直接用conditionalUnet预测x0_bar即可，无需转换为score
            x0_bar = self.noise_fn(x, t, **kwargs)
            # score = self.score_fn(x, t, **kwargs)
            # x = self.reverse_sde_step(x, score, t)
            # x0_bar = self.get_init_state_from_noise(x, self.model(x, self.mu, t, **kwargs), t)
            
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        # 画图部分

        # 将判别式模型的单步结果复制为多步
        multi_step = multi_step * (end-begin-1)
        plt.plot(multi_step[begin:end])
        # plt.plot (multi_step[begin:end])
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Multi-step Loss Curve')
        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp/multi_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突

        # 保存 xt 和 x_gt 的图像
        # x0_bar = 1-(x0_bar - x0_bar.min()) / (x0_bar.max() - x0_bar.min())
        # # x0_bar = (x0_bar + 1) / 2 
        self.save_image(x0_bar , f'/home/proj/image-restoration-sde-main/codes/temp/x0_bar_{name}.png')
        self.save_image(x_gt, f'/home/proj/image-restoration-sde-main/codes/temp/x_gt_{name}.png')

        exit()
        return x

    def inference_multi_steploss_traditional_reg_c(self, xt, x_gt, name, condition=None, begin=1, end=100, **kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)
        for t in tqdm(reversed(range(T, T + 1))):
            # 此处直接用conditionalUnet预测x0_bar即可，无需转换为score
            x0_bar_and_c = self.noise_fn(x, t, **kwargs)
            x0_bar = x0_bar_and_c * 2 - condition           
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        # 画图部分

        # 将判别式模型的单步结果复制为多步
        multi_step = multi_step * (end-begin-1)
        plt.plot(multi_step[begin:end])
        # plt.plot (multi_step[begin:end])
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Multi-step Loss Curve')
        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp/multi_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突

        # self.save_image(x0_bar , f'/home/proj/image-restoration-sde-main/codes/temp/x0_bar_{name}.png')
        # self.save_image(x_gt, f'/home/proj/image-restoration-sde-main/codes/temp/x_gt_{name}.png')

        exit()
        return x


    def inference_multi_steploss_x0(self, xt, x_gt, name, begin=70, end=100, **kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        multi_step_mask = []
        multi_step_unmask = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)

        mask = torch.any(x_gt.float() != self.mu.float(), dim=1).unsqueeze(1) #[1, 1, 256, 256]

        # 通过 expand 扩展到 [1, 3, 256, 256]
        mask = mask.expand(-1, 3, -1, -1)
        for t in tqdm(reversed(range(1, T + 1))):
            x0_bar = self.noise_fn(x, t, **kwargs)
            noise = self.get_noise_from_init_state(x0_bar, x, t)
            score = self.get_score_from_noise(noise, t)
            x = self.reverse_sde_step(x, score, t)
            # score = self.score_fn(x, t, **kwargs)
            # x = self.reverse_sde_step(x, score, t)
            # x0_bar = self.get_init_state_from_noise(x, self.model(x, self.mu, t, **kwargs), t)
            
            loss_multi_mask = loss_fn(x0_bar[mask], x_gt[mask], reduction='none')
            loss_multi_mask = einops.reduce(loss_multi_mask, 'b ... -> b (...)', 'mean')
            multi_step_mask.append(loss_multi_mask.mean().item())
            print(loss_multi_mask.mean().item())
            loss_multi_unmask = loss_fn(x0_bar[~mask], x_gt[~mask], reduction='none')
            loss_multi_unmask = einops.reduce(loss_multi_unmask, 'b ... -> b (...)', 'mean')
            multi_step_unmask.append(loss_multi_unmask.mean().item())
            print(loss_multi_unmask.mean().item())   
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        # 绘制各条曲线
        plt.plot(multi_step_mask[begin:end], label='Mask Loss', color='green', linestyle='-', linewidth=1.5)  # 第一条曲线
        plt.plot(multi_step_unmask[begin:end], label='Unmask Loss', color='blue', linestyle='-', linewidth=1.5)  # 第二条曲线
        plt.plot(multi_step[begin:end], label='Total Loss', color='red', linestyle='-', linewidth=1.5)  # 第三条曲线

        # 添加图例
        plt.legend()
        
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Multi-step Loss Curve')
        # 设置横坐标从1开始
        # 设置横坐标从begin + 1开始，到end + 1结束

        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp_seperate/multi_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突
        exit()

    def inference_multi_steploss_x0_c(self, xt, x_gt, name, condition=None, begin=1, end=100, **kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)
        for t in tqdm(reversed(range(1, T + 1))):
            x0_bar_and_c = self.noise_fn(x, t, **kwargs)
            x0_bar = x0_bar_and_c * 2- condition
            noise = self.get_noise_from_init_state(x0_bar, x, t)
            score = self.get_score_from_noise(noise, t)
            x = self.reverse_sde_step(x, score, t)
            # score = self.score_fn(x, t, **kwargs)
            # x = self.reverse_sde_step(x, score, t)
            # x0_bar = self.get_init_state_from_noise(x, self.model(x, self.mu, t, **kwargs), t)
            
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        # 画图部分
        plt.plot(multi_step[begin:end])
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Multi-step Loss Curve')
        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp/multi_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突

        self.save_image(x0_bar , f'/home/proj/image-restoration-sde-main/codes/temp/x0_x0_bar_{name}.png')
        self.save_image(x_gt, f'/home/proj/image-restoration-sde-main/codes/temp/x0_x_gt_{name}.png')

        exit()
        return x

    # 保存单张图像的函数
    def save_image(self, tensor, filename, cmap='gray'):
        import torch
        import matplotlib.pyplot as plt
        import numpy as np

        # 将张量移动到 CPU 并转换为 NumPy 格式
        tensor = tensor.cpu().squeeze().numpy()  # 假设 tensor 形状是 (1, C, H, W) 或 (C, H, W)

        # 如果是彩色图像 (C, H, W)，则转置为 (H, W, C)
        if tensor.ndim == 3 and tensor.shape[0] == 3:
            tensor = np.transpose(tensor, (1, 2, 0))

        # 绘制图像
        plt.imshow(tensor, cmap=cmap)
        plt.axis('off')  # 关闭坐标轴显示

        # 保存图像
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()




#############################################
#############################################
#############Traditional Reg#################
#############################################
#############################################

class  Traditional_Reg(SDE):
    '''
    Let timestep t start from 1 to T, state t=0 is never used
    '''
    def __init__(self, max_sigma, T=100, schedule='cosine', eps=0.01,  device=None):
        super().__init__(T, device)
        self.max_sigma = max_sigma / 255 if max_sigma > 1 else max_sigma
        self.max_sigma = max_sigma / 255 if max_sigma > 1 else max_sigma
        self._initialize(self.max_sigma, T, schedule, eps)

    def _initialize(self, max_sigma, T, schedule, eps=0.01):

        def constant_theta_schedule(timesteps, v=1.):
            """
            constant schedule
            """
            print('constant schedule')
            timesteps = timesteps + 1 # T from 1 to 100
            return torch.ones(timesteps, dtype=torch.float32)

        def linear_theta_schedule(timesteps):
            """
            linear schedule
            """
            print('linear schedule')
            timesteps = timesteps + 1 # T from 1 to 100
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        def cosine_theta_schedule(timesteps, s = 0.008):
            """
            cosine schedule
            """
            print('cosine schedule')
            timesteps = timesteps + 2 # for truncating from 1 to -1
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:-1]
            return betas

        def cosine_theta_schedule_contractive(timesteps, s = 0.008):
            """
            cosine schedule contractive
            """
            print('cosine schedule')
            timesteps = timesteps + 2 # for truncating from 1 to -1
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            # reverse
            alphas_cumprod = torch.flip(alphas_cumprod, dims=[0])
            
            betas = 1 - alphas_cumprod[1:-1]
            return betas

        def get_thetas_cumsum(thetas):
            return torch.cumsum(thetas, dim=0)

        def get_sigmas(thetas):
            return torch.sqrt(max_sigma**2 * 2 * thetas)

        def get_sigma_bars(thetas_cumsum):
            return torch.sqrt(max_sigma**2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))
            
        if schedule == 'cosine':
            thetas = cosine_theta_schedule(T)
        elif schedule == 'linear':
            thetas = linear_theta_schedule(T)
        elif schedule == 'constant':
            thetas = constant_theta_schedule(T)
        # elif cosine-contractive == 'consine-contractive':
        #     schedule == cosine_theta_schedule_contractive(T)
        else:
            print('Not implemented such schedule yet!!!')

        sigmas = get_sigmas(thetas)
        thetas_cumsum = get_thetas_cumsum(thetas) - thetas[0] # for that thetas[0] is not 0
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = get_sigma_bars(thetas_cumsum)
        
        self.thetas = thetas.to(self.device)
        self.sigmas = sigmas.to(self.device)
        self.thetas_cumsum = thetas_cumsum.to(self.device)
        self.sigma_bars = sigma_bars.to(self.device)

        self.mu = 0.
        self.model = None

    #####################################

    # set mu for different cases
    def set_mu(self, mu):
        self.mu = mu
        # self.mu = torch.zeros_like(mu)

    # set score model for reverse process
    def set_model(self, model):
        self.model = model

    #####################################

    def mu_bar(self, x0, t):
        return  x0 * torch.exp(-self.thetas_cumsum[t] * self.dt)
        # return self.mu + (x0 - self.mu) * torch.exp(-self.thetas_cumsum[t] * self.dt)

    def sigma_bar(self, t):
        return self.sigma_bars[t]

    def drift(self, x, t):
        return self.thetas[t] * (self.mu - x) * self.dt

    # def sde_reverse_drift(self, x, score, t):
    #     return (self.thetas[t] * (self.mu - x) - self.sigmas[t]**2 * score) * self.dt

    def sde_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (-x) - self.sigmas[t]**2 * score) * self.dt
        
    def sde_reverse_drift_contractive(self, x, score, t):
        return (-self.thetas[t] * (-x) - self.sigmas[t]**2 * score) * self.dt

    def ode_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (self.mu - x) - 0.5 * self.sigmas[t]**2 * score) * self.dt

    def dispersion(self, x, t):
        return self.sigmas[t] * (torch.randn_like(x) * math.sqrt(self.dt)).to(self.device)

    def get_score_from_noise(self, noise, t):
        return -noise / self.sigma_bar(t)

    def score_fn(self, x, t, **kwargs):
        # need to pre-set mu and score_model
        noise = self.model(x, self.mu, t, **kwargs)
        return self.get_score_from_noise(noise, t)

    def noise_fn(self, x, t, **kwargs):
        # need to pre-set mu and score_model
        return self.model(x, self.mu, t, **kwargs)

    # optimum x_{t-1}
    def reverse_optimum_step(self, xt, x0, t):
        A = torch.exp(-self.thetas[t] * self.dt)
        B = torch.exp(-self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-self.thetas_cumsum[t-1] * self.dt)

        term1 = A * (1 - C**2) / (1 - B**2)
        term2 = C * (1 - A**2) / (1 - B**2)

        # return term1 * (xt - self.mu) + term2 * (x0 - self.mu) + self.mu
        return term1 * (xt) + term2 * (x0)

    def reverse_optimum_std(self, t):
        A = torch.exp(-2*self.thetas[t] * self.dt)
        B = torch.exp(-2*self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-2*self.thetas_cumsum[t-1] * self.dt)

        posterior_var = (1 - A) * (1 - C) / (1 - B)
        # return torch.sqrt(posterior_var)

        min_value = (1e-20 * self.dt).to(self.device)
        log_posterior_var = torch.log(torch.clamp(posterior_var, min=min_value))
        return (0.5 * log_posterior_var).exp() * self.max_sigma

    def reverse_posterior_step(self, xt, noise, t):
        x0 = self.get_init_state_from_noise(xt, noise, t)
        mean = self.reverse_optimum_step(xt, x0, t)
        std = self.reverse_optimum_std(t)
        return mean + std * torch.randn_like(xt)

    def sigma(self, t):
        return self.sigmas[t]

    def theta(self, t):
        return self.thetas[t]

    def get_real_noise(self, xt, x0, t):
        return (xt - self.mu_bar(x0, t)) / self.sigma_bar(t)

    def get_real_score(self, xt, x0, t):
        return -(xt - self.mu_bar(x0, t)) / self.sigma_bar(t)**2

    def get_init_state_from_noise(self, xt, noise, t):
        A = torch.exp(self.thetas_cumsum[t] * self.dt)
        return (xt - self.sigma_bar(t) * noise) * A
        # return (xt - self.mu - self.sigma_bar(t) * noise) * A + self.mu

    # mu =0 for vanilla ddpm
    def get_noise_from_init_state(self, state, xt, t):
        A = torch.exp(self.thetas_cumsum[t] * self.dt) # 计算 A
        sigma_bar_t = self.sigma_bar(t) # 计算 sigma_bar(t)

        # 根据推导公式反推 noise
        # noise = (xt - self.mu - (state - self.mu) / A) / sigma_bar_t
        noise = (xt - (state) / A) / sigma_bar_t

        return noise


    # forward process to get x(T) from x(0)
    def forward(self, x0, T=-1, save_dir='forward_state'):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{t}.png', normalize=False)
        return x    
    
    def reverse_sde_visual(self, xt, current_step, T=-1, save_states=False, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t, **kwargs)
            x = self.reverse_sde_step(x, score, t)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x
    
    def reverse_sde_visual_ddpm_ip(self, xt, current_step, T=-1, save_states=False, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t, **kwargs)
            x = self.reverse_sde_step(x, score, t)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x

    def reverse_sde_visual_contractive(self, xt, current_step, T=-1, save_states=False, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t, **kwargs)
            x = self.reverse_sde_step_contractive(x, score, t)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x

    def reverse_sde_visual_x0(self, xt, current_step, T=-1, save_states=False, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            #noise from init_state
            # noise_fn() actually predicts x0 at present
            state =self.noise_fn(x, t, **kwargs)
            noise = self.get_noise_from_init_state(state, x, t)
            score = self.get_score_from_noise(noise, t)
            # score = self.score_fn(x, t, **kwargs)
            x = self.reverse_sde_step(x, score, t)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x


    def reverse_sde_visual_x0_c(self, xt, current_step, T=-1, save_states=False, condition=None, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            #noise from init_state
            # noise_fn() actually predicts x0 at present
            state_and_c =self.noise_fn(x, t, **kwargs)
            state = state_and_c * 2 - condition
            noise = self.get_noise_from_init_state(state, x, t)
            score = self.get_score_from_noise(noise, t)
            # score = self.score_fn(x, t, **kwargs)
            x = self.reverse_sde_step(x, score, t)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x


    def reverse_sde_visual_traditional_reg(self, xt, current_step=-1, T=-1, save_states=False, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(T, T + 1))):
            #noise from init_state
            # noise_fn() actually predicts x0 at present
            state =self.noise_fn(x, t, **kwargs)
            x = state
            # noise = self.get_noise_from_init_state(state, xt, t)
            # score = self.get_score_from_noise(noise, t)
            # # score = self.score_fn(x, t, **kwargs)
            # x = self.reverse_sde_step(x, score, t)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x

    def reverse_sde_visual_traditional_reg_c(self, xt, current_step=-1, T=-1, save_states=False, condition=None, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(T, T + 1))):
            #noise from init_state
            # noise_fn() actually predicts x0 at present
            state_and_c =self.noise_fn(x, t, **kwargs)
            state = state_and_c * 2 -condition
            x = state
            # noise = self.get_noise_from_init_state(state, xt, t)
            # score = self.get_score_from_noise(noise, t)
            # # score = self.score_fn(x, t, **kwargs)
            # x = self.reverse_sde_step(x, score, t)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x

        

    # def reverse_sde(self, xt, T=-1, save_states=False, save_dir='sde_state', **kwargs):
    # # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
    #     T = self.T if T < 0 else T
    #     x = xt.clone()
    #     for t in tqdm(reversed(range(1, T + 1))):
    #         score = self.score_fn(x, t, **kwargs)
    #         x = self.reverse_sde_step(x, score, t)

    #         if save_states: # only consider to save 100 images
    #             interval = self.T // 100
    #             if t % interval == 0:
    #                 idx = t // interval
    #                 os.makedirs(save_dir, exist_ok=True)
    #                 tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

    #     return x

    def reverse_ode(self, xt, T=-1, save_states=False, save_dir='ode_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t, **kwargs)
            x = self.reverse_ode_step(x, score, t)

            if save_states: # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x

    def reverse_posterior(self, xt, T=-1, save_states=False, save_dir='posterior_state', **kwargs):
        T = self.T if T < 0 else T

        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            noise = self.noise_fn(x, t, **kwargs)
            x = self.reverse_posterior_step(x, noise, t)

            if save_states: # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x


    # sample ode using Black-box ODE solver (not used)
    def ode_sampler(self, xt, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3,):
        shape = xt.shape

        def to_flattened_numpy(x):
          """Flatten a torch tensor `x` and convert it to numpy."""
          return x.detach().cpu().numpy().reshape((-1,))

        def from_flattened_numpy(x, shape):
          """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
          return torch.from_numpy(x.reshape(shape))

        def ode_func(t, x):
            t = int(t)
            x = from_flattened_numpy(x, shape).to(self.device).type(torch.float32)
            score = self.score_fn(x, t)
            drift = self.ode_reverse_drift(x, score, t)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(ode_func, (self.T, eps), to_flattened_numpy(xt),
                                     rtol=rtol, atol=atol, method=method)

        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(self.device).type(torch.float32)

        return x

    def optimal_reverse(self, xt, x0, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            x = self.reverse_optimum_step(x, x0, t)

        return x

    ################################################################

    def weights(self, t):
        return torch.exp(-self.thetas_cumsum[t] * self.dt)

    # sample states for training
    # def generate_random_states(self, x0, mu):
    #     x0 = x0.to(self.device)
    #     mu = mu.to(self.device)

    #     self.set_mu(mu)

    #     batch = x0.shape[0]

    #     timesteps = torch.randint(1, self.T + 1, (batch, 1, 1, 1)).long()

    #     state_mean = self.mu_bar(x0, timesteps)
    #     noises = torch.randn_like(state_mean)
    #     noise_level = self.sigma_bar(timesteps)
    #     noisy_states = noises * noise_level + state_mean

    #     # return timesteps, noisy_states.to(torch.float32)
    #     return timesteps, noisy_states.to(torch.float32), noises

    def generate_random_states(self, x0, mu):
        x0 = x0.to(self.device)
        mu = mu.to(self.device)

        self.set_mu(mu)

        batch = x0.shape[0]

        timesteps = torch.randint(1, self.T + 1, (batch, 1, 1, 1)).long()

        # test
        # timesteps = 30
        timesteps_fix = (self.T)  * torch.ones_like(timesteps).long()
        # timesteps_fix = (self.T) * torch.ones_like(timesteps).long()
        state_mean = self.mu_bar(x0, timesteps_fix)
        noises = torch.randn_like(state_mean)
        noise_level = self.sigma_bar(timesteps_fix)
        noisy_states = noises * noise_level + state_mean
        
        import numpy as np
        from torchvision.utils import make_grid
        def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
            """
            Converts a torch Tensor into an image Numpy array
            Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
            Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
            """
            tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
            tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
            n_dim = tensor.dim()
            if n_dim == 4:
                n_img = len(tensor)
                img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
                img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
            elif n_dim == 3:
                img_np = tensor.numpy()
                img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
            elif n_dim == 2:
                img_np = tensor.numpy()
            else:
                raise TypeError(
                    "Only support 4D, 3D and 2D tensor. But received with dimension: {:d}".format(
                        n_dim
                    )
                )
            if out_type == np.uint8:
                img_np = (img_np * 255.0).round()
                # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
            return img_np.astype(out_type)
        
        # import cv2
        # noisy_states = tensor2img(noisy_states)
        # x0 = tensor2img(x0)
        # cv2.imwrite('/home/proj/image-restoration-sde-main/codes/temp/noisy_maxsigma30.png', noisy_states)
        # cv2.imwrite('/home/proj/image-restoration-sde-main/codes/temp/x0.png', x0)
        
        # exit()

        state_mean = self.mu_bar(x0, timesteps)
        noises = torch.randn_like(state_mean)
        noise_level = self.sigma_bar(timesteps)
        noisy_states = noises * noise_level + state_mean

        return timesteps, noisy_states.to(torch.float32), noises


    # sample states for training
    def generate_random_states_traditional_reg(self, x0, mu):
        x0 = x0.to(self.device)
        mu = mu.to(self.device)

        self.set_mu(mu)

        batch = x0.shape[0]

        # timesteps = torch.randint(self.T, self.T + 1, (batch, 1, 1, 1)).long()
        timesteps = torch.ones((batch, 1, 1, 1), dtype=torch.long) * 1
        

        state_mean = self.mu_bar(x0, timesteps)
        noises = torch.randn_like(state_mean)
        noise_level = self.sigma_bar(timesteps)
        noisy_states = noises * noise_level + state_mean
    
        # return timesteps, noisy_states.to(torch.float32)
        return timesteps, noisy_states.to(torch.float32), noises

    # def noise_state(self, tensor):
    #     return tensor + torch.randn_like(tensor) * self.max_sigma


    def save_noisy_images(self, noisy_states, filename='noisy_image.png'):
        import torch
        import matplotlib.pyplot as plt
        import numpy as np
        # 如果 noisy_states 是 GPU 张量，先将其转到 CPU
        noisy_states = noisy_states.cpu()

        # 将 noisy_states 转换为 NumPy 数组
        noisy_states_np = noisy_states.squeeze().numpy()  # 去掉 batch 维度（假设是单张图）

        # 如果图像是多通道 (如 RGB)，调整维度 (C, H, W) -> (H, W, C)
        if noisy_states_np.ndim == 3 and noisy_states_np.shape[0] == 3:
            noisy_states_np = np.transpose(noisy_states_np, (1, 2, 0))

        # 创建绘图
        plt.imshow(noisy_states_np, cmap='gray')  # 对于灰度图像使用 cmap='gray'
        plt.axis('off')  # 不显示坐标轴

        # 保存图像
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()


    def noise_state(self, tensor):
        # return torch.randn_like(tensor) * self.max_sigma
        return tensor + torch.randn_like(tensor) * self.max_sigma
        # return tensor + torch.ones_like(tensor) - (torch.randn_like(tensor) * self.max_sigma)


    def noise_t_state(self, tensor, t):
        batch = tensor.shape[0]
        # timesteps = torch.full((batch, 1, 1, 1), self.T).long()
        timesteps = torch.full((batch, 1, 1, 1), t).long()
        tensor = tensor.to(self.device)
        noise_level = self.sigma_bar(timesteps).to(self.device)
        alpha_bar = torch.exp(-self.thetas_cumsum[t] * self.dt).to(self.device)
        
        return tensor * alpha_bar + torch.randn_like(tensor) * noise_level


    def inference_multi_steploss(self, xt, x_gt, name, begin=95, end=100, **kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t, **kwargs)
            x = self.reverse_sde_step(x, score, t)
            x0_bar = self.get_init_state_from_noise(x, self.model(x, self.mu, t, **kwargs), t)
            
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        # 画图部分
        plt.plot(multi_step[begin:end])
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Multi-step Loss Curve')
        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp/multi_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突
        exit()
        return x

    def inference_single_steploss(self, xt, x_gt, name, begin=1, end=100, **kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)
        for t in tqdm(reversed(range(1, T + 1))):
            
            xt_temp = self.noise_t_state(x_gt, t)
            score = self.score_fn(x, t, **kwargs)
            # x = self.reverse_sde_step(x, score, t)
            x0_bar = self.get_init_state_from_noise(xt_temp, self.model(xt_temp, self.mu, t, **kwargs), t)
            
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        
        # 画图部分
        plt.plot(multi_step[begin:end])
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Single-step Loss Curve')
        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp/single_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突
        exit()
        return x

    def inference_single_steploss_x0(self, xt, x_gt, name, begin=1, end=30,**kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)
        for t in tqdm(reversed(range(1, T + 1))):     
            xt_temp = self.noise_t_state(x_gt, t)
            x0_bar = self.noise_fn(xt_temp, t, **kwargs)
            # score = self.score_fn(x, t, **kwargs)
            # # x = self.reverse_sde_step(x, score, t)
            # x0_bar = self.get_init_state_from_noise(xt_temp, self.model(xt_temp, self.mu, t, **kwargs), t)
            
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        
        # 画图部分
        plt.plot(multi_step[begin:end])
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Single-step Loss Curve')
        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp/single_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突
        exit()
        return x

    def inference_single_steploss_x0_c(self, xt, x_gt, name, condition=None, begin=70, end=100,**kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)
        for t in tqdm(reversed(range(1, T + 1))):     
            xt_temp = self.noise_t_state(x_gt, t)
            x0_bar_and_c = self.noise_fn(xt_temp, t, **kwargs)
            x0_bar = x0_bar_and_c * 2 - condition
            # score = self.score_fn(x, t, **kwargs)
            # # x = self.reverse_sde_step(x, score, t)
            # x0_bar = self.get_init_state_from_noise(xt_temp, self.model(xt_temp, self.mu, t, **kwargs), t)
            
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        
        # 画图部分
        plt.plot(multi_step[begin:end])
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Single-step Loss Curve')
        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp/single_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突
        exit()
        return x
    

    def inference_multi_steploss_traditional_reg(self, xt, x_gt, name, begin=1, end=100, **kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)
        for t in tqdm(reversed(range(T, T + 1))):
            # 此处直接用conditionalUnet预测x0_bar即可，无需转换为score
            x0_bar = self.noise_fn(x, t, **kwargs)
            # score = self.score_fn(x, t, **kwargs)
            # x = self.reverse_sde_step(x, score, t)
            # x0_bar = self.get_init_state_from_noise(x, self.model(x, self.mu, t, **kwargs), t)
            
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        # 画图部分

        # 将判别式模型的单步结果复制为多步
        multi_step = multi_step * (end-begin-1)
        plt.plot(multi_step[begin:end])
        # plt.plot (multi_step[begin:end])
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Multi-step Loss Curve')
        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp/multi_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突

        # 保存 xt 和 x_gt 的图像
        # x0_bar = 1-(x0_bar - x0_bar.min()) / (x0_bar.max() - x0_bar.min())
        # # x0_bar = (x0_bar + 1) / 2 
        self.save_image(x0_bar , f'/home/proj/image-restoration-sde-main/codes/temp/x0_bar_{name}.png')
        self.save_image(x_gt, f'/home/proj/image-restoration-sde-main/codes/temp/x_gt_{name}.png')

        exit()
        return x

    def inference_multi_steploss_traditional_reg_c(self, xt, x_gt, name, condition=None, begin=1, end=100, **kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)
        for t in tqdm(reversed(range(T, T + 1))):
            # 此处直接用conditionalUnet预测x0_bar即可，无需转换为score
            x0_bar_and_c = self.noise_fn(x, t, **kwargs)
            x0_bar = x0_bar_and_c * 2 - condition           
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        # 画图部分

        # 将判别式模型的单步结果复制为多步
        multi_step = multi_step * (end-begin-1)
        plt.plot(multi_step[begin:end])
        # plt.plot (multi_step[begin:end])
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Multi-step Loss Curve')
        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp/multi_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突

        # self.save_image(x0_bar , f'/home/proj/image-restoration-sde-main/codes/temp/x0_bar_{name}.png')
        # self.save_image(x_gt, f'/home/proj/image-restoration-sde-main/codes/temp/x_gt_{name}.png')

        exit()
        return x

    # 保存单张图像的函数
    def save_image(self, tensor, filename, cmap='gray'):
        import torch
        import matplotlib.pyplot as plt
        import numpy as np

        # 将张量移动到 CPU 并转换为 NumPy 格式
        tensor = tensor.cpu().squeeze().numpy()  # 假设 tensor 形状是 (1, C, H, W) 或 (C, H, W)

        # 如果是彩色图像 (C, H, W)，则转置为 (H, W, C)
        if tensor.ndim == 3 and tensor.shape[0] == 3:
            tensor = np.transpose(tensor, (1, 2, 0))

        # 绘制图像
        plt.imshow(tensor, cmap=cmap)
        plt.axis('off')  # 关闭坐标轴显示

        # 保存图像
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()





#############################################
#############################################
#############Gaussian Diffusion##############
#############################################
#############################################

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()
    
class  Gaussian_Diffusion(SDE):
    '''
    Let timestep t start from 1 to T, state t=0 is never used
    '''
    def __init__(self, max_sigma, T=100, schedule='cosine', eps=0.01,  device=None):
        super().__init__(T, device)
        self.max_sigma = max_sigma / 255 if max_sigma > 1 else max_sigma
        self.max_sigma = max_sigma / 255 if max_sigma > 1 else max_sigma
        self._initialize(self.max_sigma, T, schedule, eps)

    def _initialize(self, max_sigma, T, schedule, eps=0.01):

        def constant_theta_schedule(timesteps, v=1.):
            """
            constant schedule
            """
            print('constant schedule')
            timesteps = timesteps + 1 # T from 1 to 100
            return torch.ones(timesteps, dtype=torch.float32)

        def linear_theta_schedule(timesteps):
            """
            linear schedule
            """
            print('linear schedule')
            timesteps = timesteps + 1 # T from 1 to 100
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        def cosine_theta_schedule(timesteps, s = 0.008):
            """
            cosine schedule
            """
            print('cosine schedule')
            timesteps = timesteps + 2 # for truncating from 1 to -1
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:-1]
            return betas
            # 此处betas和ddpm的batas不同，指的是irsde中的theta。

        def get_thetas_cumsum(thetas):
            return torch.cumsum(thetas, dim=0)

        def get_sigmas(thetas):
            return torch.sqrt(max_sigma**2 * 2 * thetas)

        def get_sigma_bars(thetas_cumsum):
            return torch.sqrt(max_sigma**2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))

        def get_alpha_bars(thetas_cumsum):
            return torch.exp(-2 * thetas_cumsum * self.dt)

        def reverse_alphas_cumprod(alphas_cumprod):
            alphas_cumprod =alphas_cumprod.cpu()
            alphas = np.zeros_like(alphas_cumprod)
            alphas[0] = alphas_cumprod[0]
            
            for i in range(1, len(alphas_cumprod)):
                alphas[i] = alphas_cumprod[i] / alphas_cumprod[i-1]
            
            return alphas
            
        if schedule == 'cosine':
            thetas = cosine_theta_schedule(T)
        elif schedule == 'linear':
            thetas = linear_theta_schedule(T)
        elif schedule == 'constant':
            thetas = constant_theta_schedule(T)
        else:
            print('Not implemented such schedule yet!!!')

        sigmas = get_sigmas(thetas)
        thetas_cumsum = get_thetas_cumsum(thetas) - thetas[0] # for that thetas[0] is not 0
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = get_sigma_bars(thetas_cumsum)
        
        self.thetas = thetas.to(self.device)
        self.sigmas = sigmas.to(self.device)
        # self.thetas_cumsum = thetas_cumsum.to(self.device)
        self.sigma_bars = sigma_bars.to(self.device)

        self.mu = 0.
        self.model = None


        # alphas = 1.0 - betas
        self.alphas_cumprod = get_alpha_bars(thetas_cumsum=thetas_cumsum)
        alphas = reverse_alphas_cumprod(self.alphas_cumprod)
        # print(alphas.shape)
        # self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        self.num_timesteps = int(alphas.shape[0])
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., self.alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(self.device)
        # self.sqrt_recipm1_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(self.device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(self.device)
        self.one_minus_sqrt_one_minus_alphas_cumprod = 1.0 - self.sqrt_one_minus_alphas_cumprod
        


        # Use float64 for accuracy.
        betas = 1. - alphas
        # betas = thetas
        betas = np.array(betas)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        # assert (betas > 0).all() and (betas <= 1).all()


        # linear schedule
        # start = 1.011 - self.eps_scaler * ((self.num_timesteps - 1) / 2)
        # self.sampling_scaler = [(start + i * self.eps_scaler) for i in range(0, self.num_timesteps)]

        # uniform schedule
        # self.eps_scaler = 0.99
        # self.sampling_scaler = [self.eps_scaler for i in range(0, self.num_timesteps)]
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )


    
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance
    
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        timesteps_ori = torch.ones([1, 1, 1, 1]).long() 
        # 将 t 乘到每个位置
        # t = timesteps_ori * (1)
        # noise_level = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t.to(self.device), x_start.shape)
        # print(noise_level)

        # for i in range(1,101):
        #     ddpm_noise_level = []
        #     t = timesteps_ori * (i)
        #     noise_level = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t.to(self.device), x_start.shape)
        #     print(noise_level[0, 0, 0, 0].item())
        #     ddpm_noise_level.append(noise_level[0, 0, 0, 0].item())

        # # 保存列表到 .txt 文件
        # with open('ddpm_noise_level_2.txt', 'a') as f:
        #     for level in ddpm_noise_level:
        #         f.write(f"{level}\n")  # 每个 noise level 写一行
        # exit(0)
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t.to(self.device), x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t.to(self.device), x_start.shape)
            * noise
        ),noise

    def q_sample_transition(self, hq, lq, t, type='alpha', noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(hq)
        assert noise.shape == hq.shape

        # 设置抛硬币的概率
        p = 0.5  # 你可以根据需要修改 p 的值

        if type == 'half':
            # 模拟抛硬币
            if random.random() < p:       
                x_start = hq
            else:
                x_start = lq

        elif type == 'alpha':
            x_start = lq * _extract_into_tensor(self.sqrt_alphas_cumprod, t.to(self.device), hq.shape)
            + hq * _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t.to(self.device), hq.shape)
        
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t.to(self.device), x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t.to(self.device), x_start.shape)
            * noise
        ),noise


    def q_sample_transition_reverse(self, hq, lq, t, type='alpha', noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(hq)
        assert noise.shape == hq.shape

        # 设置抛硬币的概率
        p = 0.5  # 你可以根据需要修改 p 的值

        if type == 'half':
            # 模拟抛硬币
            if random.random() < p:       
                x_start = hq
            else:
                x_start = lq

        elif type == 'alpha':
            x_start = hq * _extract_into_tensor(self.sqrt_alphas_cumprod, t.to(self.device), hq.shape)
            + lq * _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t.to(self.device), hq.shape)
        
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t.to(self.device), x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t.to(self.device), x_start.shape)
            * noise
        ),noise


    def q_sample_transition_dream(self, hq, lq, t, type='alpha', noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(hq)
        assert noise.shape == hq.shape

        if type == 'alpha':
            # ones_tensor = torch.ones_like(hq)
            x_start = lq * _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,  t.to(self.device), hq.shape)
            + hq * _extract_into_tensor(self.one_minus_sqrt_one_minus_alphas_cumprod, t.to(self.device), hq.shape)
        
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t.to(self.device), x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t.to(self.device), x_start.shape)
            * noise
        ),noise



    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
        return posterior_mean + noise * (0.5 * posterior_log_variance_clipped).exp()

    def q_posterior_mean(self, x_start, x_t, t):
        """
        Compute the mean the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        return posterior_mean


    def multi_step_fuse(self, x_tk, x0, t, k):
        batch_size = x_tk.shape[0]
        gamma = torch.ones_like(x_tk)
        omega = torch.zeros_like(x_tk)

        for n in range(batch_size):
            gamma_i = gamma[n]
            omega_i = 0
            t_i = t[n]

            # Compute gamma for each sample in the batch
            for i in range(t_i, t_i + k):
                gamma_i = gamma_i * _extract_into_tensor(self.posterior_mean_coef2, t_i, x_tk[0].shape)
            
            # Outer summation for each sample in the batch
            for j in range(t_i, t_i + k - 1):
                alphas = 1. - self.betas
                # Compute the product ∏ part
                prod_n = torch.prod(torch.sqrt(torch.from_numpy(alphas[t_i+1:j+2])) * (1 - self.alphas_cumprod[t_i:j+1]))
                
                # Compute numerator
                numerator = prod_n * torch.sqrt(self.alphas_cumprod[j+1]) * torch.sqrt(torch.tensor(self.betas[j + 2], dtype=torch.float32))
                
                # Compute denominator
                denominator = torch.prod(1 - self.alphas_cumprod[t_i+1:j+3])
                
                # Accumulate omega_i
                omega_i += numerator / denominator

            # Expand omega to match shape and move to the device
            omega[n] = omega_i.to(device=self.device).float().expand(x_tk[0].shape)

        # Final output
        out = gamma * x_tk + omega * x0 + \
            _extract_into_tensor(self.posterior_mean_coef1, t, x_tk.shape) * x0
        
        return out





    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }


    #####################################

    # set mu for different cases
    def set_mu(self, mu):
        self.mu = mu
        # self.mu = torch.zeros_like(mu)

    # set score model for reverse process
    def set_model(self, model):
        self.model = model

    #####################################

    def mu_bar(self, x0, t):
        return  x0 * torch.exp(-self.thetas_cumsum[t] * self.dt)
        # return self.mu + (x0 - self.mu) * torch.exp(-self.thetas_cumsum[t] * self.dt)

    def sigma_bar(self, t):
        return self.sigma_bars[t]

    def drift(self, x, t):
        return self.thetas[t] * (self.mu - x) * self.dt

    # def sde_reverse_drift(self, x, score, t):
    #     return (self.thetas[t] * (self.mu - x) - self.sigmas[t]**2 * score) * self.dt

    def sde_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (-x) - self.sigmas[t]**2 * score) * self.dt
        
    def sde_reverse_drift_contractive(self, x, score, t):
        return (-self.thetas[t] * (-x) - self.sigmas[t]**2 * score) * self.dt

    def ode_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (self.mu - x) - 0.5 * self.sigmas[t]**2 * score) * self.dt

    def dispersion(self, x, t):
        return self.sigmas[t] * (torch.randn_like(x) * math.sqrt(self.dt)).to(self.device)

    def get_score_from_noise(self, noise, t):
        return -noise / self.sigma_bar(t)

    def score_fn(self, x, t, **kwargs):
        # need to pre-set mu and score_model
        noise = self.model(x, self.mu, t, **kwargs)
        return self.get_score_from_noise(noise, t)

    def noise_fn(self, x, t, **kwargs):
        # need to pre-set mu and score_model
        return self.model(x, self.mu, t, **kwargs)

    def noise_fn_index(self, x, t, index, **kwargs):
        # need to pre-set mu and score_model
        return self.model(x, self.mu[index:index+1], t, **kwargs)

    # # optimum x_{t-1}
    # def reverse_optimum_step(self, xt, x0, t):
    #     A = torch.exp(-self.thetas[t] * self.dt)
    #     B = torch.exp(-self.thetas_cumsum[t] * self.dt)
    #     C = torch.exp(-self.thetas_cumsum[t-1] * self.dt)

    #     term1 = A * (1 - C**2) / (1 - B**2)
    #     term2 = C * (1 - A**2) / (1 - B**2)

    #     # return term1 * (xt - self.mu) + term2 * (x0 - self.mu) + self.mu
    #     return term1 * (xt) + term2 * (x0)

    # optimum x_{t-1} on Gaussian_diffusion framework
    def reverse_optimum_step(self, xt, x0, t):
        A = torch.exp(-self.thetas[t] * self.dt)
        B = torch.exp(-self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-self.thetas_cumsum[t-1] * self.dt)

        term1 = A * (1 - C**2) / (1 - B**2)
        term2 = C * (1 - A**2) / (1 - B**2)

        # return term1 * (xt - self.mu) + term2 * (x0 - self.mu) + self.mu
        return term1 * (xt) + term2 * (x0)

    def reverse_optimum_std(self, t):
        A = torch.exp(-2*self.thetas[t] * self.dt)
        B = torch.exp(-2*self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-2*self.thetas_cumsum[t-1] * self.dt)

        posterior_var = (1 - A) * (1 - C) / (1 - B)
        # return torch.sqrt(posterior_var)

        min_value = (1e-20 * self.dt).to(self.device)
        log_posterior_var = torch.log(torch.clamp(posterior_var, min=min_value))
        return (0.5 * log_posterior_var).exp() * self.max_sigma

    def reverse_posterior_step(self, xt, noise, t):
        x0 = self.get_init_state_from_noise(xt, noise, t)
        mean = self.reverse_optimum_step(xt, x0, t)
        std = self.reverse_optimum_std(t)
        return mean + std * torch.randn_like(xt)

    def sigma(self, t):
        return self.sigmas[t]

    def theta(self, t):
        return self.thetas[t]

    def get_real_noise(self, xt, x0, t):
        return (xt - self.mu_bar(x0, t)) / self.sigma_bar(t)

    def get_real_score(self, xt, x0, t):
        return -(xt - self.mu_bar(x0, t)) / self.sigma_bar(t)**2

    def get_init_state_from_noise(self, xt, noise, t):
        A = torch.exp(self.thetas_cumsum[t] * self.dt)
        return (xt - self.sigma_bar(t) * noise) * A
        # return (xt - self.mu - self.sigma_bar(t) * noise) * A + self.mu

    # mu =0 for vanilla ddpm
    def get_noise_from_init_state(self, state, xt, t):
        A = torch.exp(self.thetas_cumsum[t] * self.dt) # 计算 A
        sigma_bar_t = self.sigma_bar(t) # 计算 sigma_bar(t)

        # 根据推导公式反推 noise
        # noise = (xt - self.mu - (state - self.mu) / A) / sigma_bar_t
        noise = (xt - (state) / A) / sigma_bar_t

        return noise


    # forward process to get x(T) from x(0)
    def forward(self, x0, T=-1, save_dir='forward_state'):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{t}.png', normalize=False)
        return x

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean 

    # def p_sample(
    #     self,
    #     model,
    #     x,
    #     t,
    #     clip_denoised=True,
    #     denoised_fn=None,
    #     cond_fn=None,
    #     model_kwargs=None,
    # ):
    #     """
    #     Sample x_{t-1} from the model at the given timestep.

    #     :param model: the model to sample from.
    #     :param x: the current tensor at x_{t-1}.
    #     :param t: the value of t, starting at 0 for the first diffusion step.
    #     :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
    #     :param denoised_fn: if not None, a function which applies to the
    #         x_start prediction before it is used to sample.
    #     :param cond_fn: if not None, this is a gradient function that acts
    #                     similarly to the model.
    #     :param model_kwargs: if not None, a dict of extra keyword arguments to
    #         pass to the model. This can be used for conditioning.
    #     :return: a dict containing the following keys:
    #              - 'sample': a random sample from the model.
    #              - 'pred_xstart': a prediction of x_0.
    #     """
    #     out = self.p_mean_variance(
    #         model,
    #         x,
    #         t,
    #         clip_denoised=clip_denoised,
    #         denoised_fn=denoised_fn,
    #         model_kwargs=model_kwargs,
    #     )
    #     noise = torch.randn_like(x)
    #     nonzero_mask = (
    #         (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
    #     )  # no noise when t == 0
    #     if cond_fn is not None:
    #         out["mean"] = self.condition_mean(
    #             cond_fn, out, x, t, model_kwargs=model_kwargs
    #         )
    #     sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
    #     return {"sample": sample, "pred_xstart": out["pred_xstart"]}   
    
    def q_posterior(self, x_start, x_t, t):
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(self.device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(self.device)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t).to(self.posterior_mean_coef1.device)
        else:
            t = t.clone().detach().to(self.posterior_mean_coef1.device)
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise
    

    def p_mean_variance(self, x, t, clip_denoised: bool, **kwargs):
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.noise_fn(x, t, **kwargs))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return x_recon, model_mean, posterior_log_variance

    def p_mean(self, x, t, index, clip_denoised: bool, **kwargs):
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.noise_fn_index(x, t, index, **kwargs))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean
    
    def p_mean_x0(self, x, t, index, clip_denoised: bool, **kwargs):
        x_recon = self.noise_fn_index(x, t, index, **kwargs)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean
    
    def p_mean_variance_x0(self, x, t, clip_denoised: bool, **kwargs):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)

         #模型以x0为target，因此需要用x0_bar来表示epsilon
        x_recon = self.noise_fn(x, t, **kwargs)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return x_recon, model_mean, posterior_log_variance
    
    def p_mean_variance_x0_scaling(self, x, t, eps_scaler, clip_denoised: bool, **kwargs):
         #模型以x0为target，因此需要用x0_bar来表示epsilon
        # x_recon = self.noise_fn(x, t, **kwargs)
        self.sampling_scaler = [eps_scaler for i in range(0, 100)]
        x_recon = self.noise_fn(x, t, **kwargs) * self.sampling_scaler[t-1]


        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return x_recon, model_mean, posterior_log_variance

    def p_mean_variance_x0_unfolded(self, x, t, clip_denoised: bool, **kwargs):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)

        x_0_t = self.noise_fn(x, t, **kwargs)
        noise = (x - self.sqrt_alphas_cumprod[t] * x_0_t) / self.sqrt_one_minus_alphas_cumprod[t]
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=noise)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return x_recon, model_mean, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, **kwargs):
        x_recon, model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, **kwargs)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        model_log_variance = torch.tensor(model_log_variance, dtype=torch.float32).to(self.device)
        # 此处x_recon对于预测x0的ddpm无用，因为noise-fn得到的就是x-recon
        # return  x_recon, model_mean + noise * (0.5 * model_log_variance).exp()
        return  x_recon, model_mean + noise * (0.5 * model_log_variance).exp()

    # @torch.no_grad()
    # 在训练阶段使用
    def p_sample_mean(self, x, t, index, clip_denoised=True, **kwargs):
        model_mean = self.p_mean(
            x=x, t=t, index=index, clip_denoised=clip_denoised, **kwargs)
        return  model_mean

    def p_sample_mean_x0(self, x, t, index, clip_denoised=True, **kwargs):
        model_mean = self.p_mean_x0(
            x=x, t=t, index=index, clip_denoised=clip_denoised, **kwargs)
        return  model_mean
    
    @torch.no_grad()
    def p_sample_x0(self, x, t, clip_denoised=True, **kwargs):
        x_recon, model_mean, model_log_variance = self.p_mean_variance_x0(
            x=x, t=t, clip_denoised=clip_denoised, **kwargs)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        model_log_variance = torch.tensor(model_log_variance, dtype=torch.float32).to(self.device)
        # 此处x_recon对于预测x0的ddpm无用，因为noise-fn得到的就是x-reconssss
        
        # return  x_recon, model_mean + noise * (0.5 * model_log_variance).exp()
        return  x_recon, model_mean + noise * (0.5 * model_log_variance).exp()

    def p_sample_x0_scaling(self, x, t, eps_scaler,clip_denoised=True, **kwargs):
        x_recon, model_mean, model_log_variance = self.p_mean_variance_x0_scaling(
            x=x, t=t, eps_scaler=eps_scaler, clip_denoised=clip_denoised, **kwargs)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        model_log_variance = torch.tensor(model_log_variance, dtype=torch.float32).to(self.device)
        # 此处x_recon对于预测x0的ddpm无用，因为noise-fn得到的就是x-reconssss
        
        # return  x_recon, model_mean + noise * (0.5 * model_log_variance).exp()
        return  x_recon, model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_x0_unfolded(self, x, t, clip_denoised=True, **kwargs):
        # x_0_t = self.noise_fn(x, t, **kwargs)
        # noise = (x - self.sqrt_alphas_cumprod[t] * x_0_t) / self.sqrt_one_minus_alphas_cumprod[t]
        # x_recon = self.predict_start_from_noise(
        #     x, t=t, noise=noise)
        x_recon = self.noise_fn(x, t, **kwargs)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        
        t = torch.tensor(t).long()
        t_sub_1 = torch.tensor(t-1).long()
        # 创建 timesteps 张量
        # timesteps = torch.ones([batch, 1, 1, 1]).long()
        timesteps_ori = torch.ones([1, 1, 1, 1]).long() 
        # 将 t 乘到每个位置
        timesteps = timesteps_ori * t
        timesteps_sub_1 = timesteps_ori * (t_sub_1)
        x_t_bar = x_recon
        if t != 0:
            x_t_bar = self.q_sample(x_start=x_recon,t=timesteps)
        x_t_sub_1_bar = x_recon
        if t-1 != 0:
            x_t_sub_1_bar = self.q_sample(x_start=x_recon,t=timesteps_sub_1)
        x = x - x_t_bar + x_t_sub_1_bar
        return x_recon, x

    def p_sample_ddrm(self, x, H_funcs=None, y_0=None, sigma_0=0.05, last=True,condition=None):
        H_funcs = self.get_H_funcs('sr_bicubic4')
        # H_funcs = self.get_H_funcs('deno')
        # skip = self.num_timesteps // self.args.timesteps
        skip = 1
        # seq = range(0, self.num_timesteps, skip)
        seq = range(1, self.num_timesteps, skip)
        y_0 = condition
        etaB = 1
        eta = 0.85
        model = None
        x = self.efficient_generalized_steps(x, seq, model, self.betas, H_funcs, y_0, sigma_0, \
            etaB=etaB, etaA=eta, etaC=eta, cls_fn=None, classes=None)
        # --timesteps 20 --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05
        if last:
            # x0_preds = x0_preds[0][-1]
            x0_preds = x[0][-1]
        return x0_preds
    
    def p_sample_ddrm_x0(self, x, H_funcs=None, y_0=None, sigma_0=0.05, last=True,condition=None):
        # H_funcs = self.get_H_funcs('sr_bicubic4')
        H_funcs = self.get_H_funcs('deno')
        # skip = self.num_timesteps // self.args.timesteps
        skip = 1
        # seq = range(0, self.num_timesteps, skip)
        seq = range(1, self.num_timesteps, skip)
        y_0 = condition
        etaB = 1
        eta = 0.85
        model = None
        _, x0_preds = self.efficient_generalized_steps(x, seq, model, self.betas, H_funcs, y_0, sigma_0, \
            etaB=etaB, etaA=eta, etaC=eta, cls_fn=None, classes=None)
        # --timesteps 20 --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05
        if last:
            # x0_preds = x0_preds[0][-1]
            x0_preds = x0_preds[-1]
        return x0_preds
    
    def p_sample_ddrm_x0(self, x, H_funcs=None, y_0=None, sigma_0=0.05, last=True,condition=None):
        H_funcs = self.get_H_funcs('sr_bicubic4')
        # H_funcs = self.get_H_funcs('deno')
        # skip = self.num_timesteps // self.args.timesteps
        skip = 1
        # seq = range(0, self.num_timesteps, skip)
        seq = range(1, self.num_timesteps, skip)
        y_0 = condition
        etaB = 1
        eta = 0.85
        model = None
        _, x0_preds = self.efficient_generalized_steps_x0(x, seq, model, self.betas, H_funcs, y_0, sigma_0, \
            etaB=etaB, etaA=eta, etaC=eta, cls_fn=None, classes=None)
        # --timesteps 20 --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05
        if last:
            # x0_preds = x0_preds[0][-1]
            x0_preds = x0_preds[-1]
        return x0_preds

    
    def get_H_funcs(self, type):
        # deg = args.deg
        deg = type
        H_funcs = None
        if deg == 'cs':
            compress_by = int(deg[2:])
            from functions.svd_replacement import WalshHadamardCS
            H_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size, compress_by, torch.randperm(self.config.data.image_size**2, device=self.device), self.device)
        elif deg[:10] == 'sr_bicubic':
            # factor = int(deg[10:])
            factor = 1
            from functions.svd_replacement import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1/factor)*(i- np.floor(factor*4/2) +0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(self.device)
            # H_funcs = SRConv(kernel / kernel.sum(), \
            #                  config.data.channels, self.config.data.image_size, self.device, stride = factor)
            H_funcs = SRConv(kernel / kernel.sum(), \
                             3, 1536, self.device, stride = factor)
        elif deg == 'deblur_uni':
            from functions.svd_replacement import Deblurring
            H_funcs = Deblurring(torch.Tensor([1/9] * 9).to(self.device), 3, 256, self.device)
        elif deg == 'deblur_gauss':
            from functions.svd_replacement import Deblurring
            sigma = 10
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.device)
            H_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'deblur_aniso':
            from functions.svd_replacement import Deblurring2D
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(self.device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(self.device)
            H_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels, self.config.data.image_size, self.device)
        elif deg[:2] == 'sr':
            blur_by = int(deg[2:])
            from functions.svd_replacement import SuperResolution
            H_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)
        elif deg == 'color':
            from functions.svd_replacement import Colorization
            H_funcs = Colorization(config.data.image_size, self.device)
        elif deg == 'deno':
            from functions.svd_replacement import Denoising
            H_funcs = Denoising(3, 256, self.device)

        else:
            print("ERROR: degradation type not supported")
            quit()

        return H_funcs


    # @torch.no_grad()
    # def p_sample_x0(self, x, t, clip_denoised=True, **kwargs):
    #     x_recon, model_mean, model_log_variance = self.p_mean_variance_x0(
    #         x=x, t=t, clip_denoised=clip_denoised, **kwargs)
    #     noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
    #     model_log_variance = torch.tensor(model_log_variance, dtype=torch.float32).to(self.device)
    #     # return  x_recon, model_mean + noise * (0.5 * model_log_variance).exp()
    #     return  x_recon, model_mean + noise * (0.5 * model_log_variance).exp()
    
    def reverse_sde_visual(self, xt, current_step, T=-1, save_states=False, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            # score = self.score_fn(x, t, **kwargs)
            # noise = self.noise_fn(x, t, **kwargs)
            # x = self.reverse_sde_step(x, score, t)
            _, x = self.p_sample(x, t, **kwargs)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x


    def reverse_double_x0(self, xt, t1, T=-1, save_states=False, t2=None, **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        # for t in tqdm(reversed(range(1, t_expectation + 1))):
        #     if t == t_expectation:
        #         x0_bar = self.noise_fn(x, t, **kwargs)
        #         break
        #     _, x = self.p_sample_x0(x, t, **kwargs)

        x0_bar = self.noise_fn(x, t1, **kwargs)
        t_2 = t2
        tensor_t_2 = torch.tensor(t_2).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, -1, -1, -1)
        x_2,_ = self.q_sample(x0_bar, tensor_t_2)
        x0_bar_bar = self.noise_fn(x_2, t_2, **kwargs)

        return x0_bar_bar

    def reverse_ddrm(self, xt, current_step, T=-1, save_states=False, save_dir='sde_state', condition=None, **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        # for t in tqdm(reversed(range(1, T + 1))):
        #     x = self.p_sample_ddrm(x, t, **kwargs)

        x = self.p_sample_ddrm(x, condition=condition, **kwargs)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x


    def reverse_single_ddrm(self, xt, current_step, T=-1, save_states=False, save_dir='sde_state', condition=None, **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        # for t in tqdm(reversed(range(1, T + 1))):
        #     x = self.p_sample_ddrm(x, t, **kwargs)

        x = self.p_sample_ddrm(x, condition=condition, **kwargs)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x
    

    def reverse_ddrm_x0(self, xt, current_step, T=-1, save_states=False, save_dir='sde_state', condition=None, **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        # for t in tqdm(reversed(range(1, T + 1))):
        #     x = self.p_sample_ddrm(x, t, **kwargs)

        x = self.p_sample_ddrm_x0(x, condition=condition, **kwargs)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x
    
    def reverse_sde_visual_ddpm_ip(self, xt, current_step, T=-1, save_states=False, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t, **kwargs)
            x = self.reverse_sde_step(x, score, t)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x

    def reverse_sde_visual_contractive(self, xt, current_step, T=-1, save_states=False, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t, **kwargs)
            x = self.reverse_sde_step_contractive(x, score, t)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x

    def reverse_sde_visual_x0(self, xt, current_step, T=-1, save_states=False, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            # score = self.score_fn(x, t, **kwargs)
            # noise = self.noise_fn(x, t, **kwargs)
            # x = self.reverse_sde_step(x, score, t)
            _, x = self.p_sample_x0(x, t, **kwargs)
            # _, x = self.p_sample_x0_scaling(x, t, **kwargs)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x

    def reverse_single_x0(self, xt, current_step, T=-1, save_states=False, t_expectation=None, **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, t_expectation + 1))):
            if t == t_expectation:
                x0_bar = self.noise_fn(x, t, **kwargs)
                break
            _, x = self.p_sample_x0(x, t, **kwargs)

        return x0_bar


        

    def reverse_sde_visual_x0_unfolded(self, xt, current_step, T=-1, save_states=False, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        x_t_plus_1 = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            _, x = self.p_sample_x0_unfolded(x, t, **kwargs)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x


    def reverse_sde_visual_x0_c(self, xt, current_step, T=-1, save_states=False, condition=None, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            #noise from init_state
            # noise_fn() actually predicts x0 at present
            state_and_c =self.noise_fn(x, t, **kwargs)
            state = state_and_c * 2 - condition
            noise = self.get_noise_from_init_state(state, x, t)
            score = self.get_score_from_noise(noise, t)
            # score = self.score_fn(x, t, **kwargs)
            x = self.reverse_sde_step(x, score, t)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x


    def reverse_sde_visual_traditional_reg(self, xt, current_step=-1, T=-1, save_states=False, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(T, T + 1))):
            #noise from init_state
            # noise_fn() actually predicts x0 at present
            state =self.noise_fn(x, t, **kwargs)
            x = state
            # noise = self.get_noise_from_init_state(state, xt, t)
            # score = self.get_score_from_noise(noise, t)
            # # score = self.score_fn(x, t, **kwargs)
            # x = self.reverse_sde_step(x, score, t)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x

    def reverse_sde_visual_traditional_reg_c(self, xt, current_step=-1, T=-1, save_states=False, condition=None, save_dir='sde_state', **kwargs):
    # def reverse_sde(self, xt, T=-1, save_states=True, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(T, T + 1))):
            #noise from init_state
            # noise_fn() actually predicts x0 at present
            state_and_c =self.noise_fn(x, t, **kwargs)
            state = state_and_c * 2 -condition
            x = state
            # noise = self.get_noise_from_init_state(state, xt, t)
            # score = self.get_score_from_noise(noise, t)
            # # score = self.score_fn(x, t, **kwargs)
            # x = self.reverse_sde_step(x, score, t)

        if save_states: # only consider to save 100 images
            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{current_step}.png', normalize=False)

        return x

    def compute_alpha(self, beta, t):
        if isinstance(beta, np.ndarray):
            beta = torch.from_numpy(beta).float().to(self.device)
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def efficient_generalized_steps(self, x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, cls_fn=None, classes=None, **kwargs):
        with torch.no_grad():
            #setup vectors used in the algorithm
            singulars = H_funcs.singulars()
            Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
            Sigma[:singulars.shape[0]] = singulars
            U_t_y = H_funcs.Ut(y_0)
            Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]]

            #initialize x_T as given in the paper
            largest_alphas = self.compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
            largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
            large_singulars_index = torch.where(singulars * largest_sigmas[0, 0, 0, 0] > sigma_0)
            inv_singulars_and_zero = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3]).to(singulars.device)
            inv_singulars_and_zero[large_singulars_index] = sigma_0 / singulars[large_singulars_index]
            inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)     

            # implement p(x_T | x_0, y) as given in the paper
            # if eigenvalue is too small, we just treat it as zero (only for init) 
            init_y = torch.zeros(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]).to(x.device)
            init_y[:, large_singulars_index[0]] = U_t_y[:, large_singulars_index[0]] / singulars[large_singulars_index].view(1, -1)
            init_y = init_y.view(*x.size())
            remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero ** 2
            remaining_s = remaining_s.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).clamp_min(0.0).sqrt()
            init_y = init_y + remaining_s * x
            init_y = init_y / largest_sigmas
            
            #setup iteration variables
            x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            x0_preds = []
            xs = [x]
       
            #iterate over the timesteps
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            if cls_fn == None:
                # et = model(xt, t)
                # 预测噪声
                et = self.noise_fn(xt, t, **kwargs)
            else:
                # et = model(xt, t, classes)
                # 预测噪声
                et = self.noise_fn(xt, t, **kwargs)

                et = et[:, :3]
                et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)
            
            if et.size(1) == 6:
                et = et[:, :3]
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            #variational inference conditioned on y
            sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
            sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
            xt_mod = xt / at.sqrt()[0, 0, 0, 0]
            V_t_x = H_funcs.Vt(xt_mod)
            SVt_x = (V_t_x * Sigma)[:, :U_t_y.shape[1]]
            V_t_x0 = H_funcs.Vt(x0_t)
            SVt_x0 = (V_t_x0 * Sigma)[:, :U_t_y.shape[1]]

            falses = torch.zeros(V_t_x0.shape[1] - singulars.shape[0], dtype=torch.bool, device=xt.device)
            cond_before_lite = singulars * sigma_next > sigma_0
            cond_after_lite = singulars * sigma_next < sigma_0
            cond_before = torch.hstack((cond_before_lite, falses))
            cond_after = torch.hstack((cond_after_lite, falses))

            std_nextC = sigma_next * etaC
            sigma_tilde_nextC = torch.sqrt(sigma_next ** 2 - std_nextC ** 2)

            std_nextA = sigma_next * etaA
            sigma_tilde_nextA = torch.sqrt(sigma_next**2 - std_nextA**2)
            
            diff_sigma_t_nextB = torch.sqrt(sigma_next ** 2 - sigma_0 ** 2 / singulars[cond_before_lite] ** 2 * (etaB ** 2))

            #missing pixels
            Vt_xt_mod_next = V_t_x0 + sigma_tilde_nextC * H_funcs.Vt(et) + std_nextC * torch.randn_like(V_t_x0)

            #less noisy than y (after)
            Vt_xt_mod_next[:, cond_after] = \
                V_t_x0[:, cond_after] + sigma_tilde_nextA * ((U_t_y - SVt_x0) / sigma_0)[:, cond_after_lite] + std_nextA * torch.randn_like(V_t_x0[:, cond_after])
            
            #noisier than y (before)
            Vt_xt_mod_next[:, cond_before] = \
                (Sig_inv_U_t_y[:, cond_before_lite] * etaB + (1 - etaB) * V_t_x0[:, cond_before] + diff_sigma_t_nextB * torch.randn_like(U_t_y)[:, cond_before_lite])

            #aggregate all 3 cases and give next prediction
            xt_mod_next = H_funcs.V(Vt_xt_mod_next)
            xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape)

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))
            # break


        return xs, x0_preds

    def efficient_generalized_steps_x0(self, x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, cls_fn=None, classes=None, **kwargs):
        with torch.no_grad():
            #setup vectors used in the algorithm
            singulars = H_funcs.singulars()
            Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
            Sigma[:singulars.shape[0]] = singulars
            U_t_y = H_funcs.Ut(y_0)
            Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]]

            #initialize x_T as given in the paper
            largest_alphas = self.compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
            largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
            large_singulars_index = torch.where(singulars * largest_sigmas[0, 0, 0, 0] > sigma_0)
            inv_singulars_and_zero = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3]).to(singulars.device)
            inv_singulars_and_zero[large_singulars_index] = sigma_0 / singulars[large_singulars_index]
            inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)     

            # implement p(x_T | x_0, y) as given in the paper
            # if eigenvalue is too small, we just treat it as zero (only for init) 
            init_y = torch.zeros(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]).to(x.device)
            init_y[:, large_singulars_index[0]] = U_t_y[:, large_singulars_index[0]] / singulars[large_singulars_index].view(1, -1)
            init_y = init_y.view(*x.size())
            remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero ** 2
            remaining_s = remaining_s.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).clamp_min(0.0).sqrt()
            init_y = init_y + remaining_s * x
            init_y = init_y / largest_sigmas
            
            #setup iteration variables
            x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            x0_preds = []
            xs = [x]
       
            #iterate over the timesteps
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            if cls_fn == None:
                # et = model(xt, t)
                # 预测噪声
                x0_t = self.noise_fn(xt, t, **kwargs)


            #variational inference conditioned on y
            sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
            sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
            xt_mod = xt / at.sqrt()[0, 0, 0, 0]
            V_t_x = H_funcs.Vt(xt_mod)
            SVt_x = (V_t_x * Sigma)[:, :U_t_y.shape[1]]
            V_t_x0 = H_funcs.Vt(x0_t)
            SVt_x0 = (V_t_x0 * Sigma)[:, :U_t_y.shape[1]]

            falses = torch.zeros(V_t_x0.shape[1] - singulars.shape[0], dtype=torch.bool, device=xt.device)
            cond_before_lite = singulars * sigma_next > sigma_0
            cond_after_lite = singulars * sigma_next < sigma_0
            cond_before = torch.hstack((cond_before_lite, falses))
            cond_after = torch.hstack((cond_after_lite, falses))

            std_nextC = sigma_next * etaC
            sigma_tilde_nextC = torch.sqrt(sigma_next ** 2 - std_nextC ** 2)

            std_nextA = sigma_next * etaA
            sigma_tilde_nextA = torch.sqrt(sigma_next**2 - std_nextA**2)
            
            diff_sigma_t_nextB = torch.sqrt(sigma_next ** 2 - sigma_0 ** 2 / singulars[cond_before_lite] ** 2 * (etaB ** 2))

            #missing pixels
            Vt_xt_mod_next = V_t_x0 + sigma_tilde_nextC * H_funcs.Vt(et) + std_nextC * torch.randn_like(V_t_x0)

            #less noisy than y (after)
            Vt_xt_mod_next[:, cond_after] = \
                V_t_x0[:, cond_after] + sigma_tilde_nextA * ((U_t_y - SVt_x0) / sigma_0)[:, cond_after_lite] + std_nextA * torch.randn_like(V_t_x0[:, cond_after])
            
            #noisier than y (before)
            Vt_xt_mod_next[:, cond_before] = \
                (Sig_inv_U_t_y[:, cond_before_lite] * etaB + (1 - etaB) * V_t_x0[:, cond_before] + diff_sigma_t_nextB * torch.randn_like(U_t_y)[:, cond_before_lite])

            #aggregate all 3 cases and give next prediction
            xt_mod_next = H_funcs.V(Vt_xt_mod_next)
            xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape)

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))


        return xs, x0_preds

        

    def reverse_posterior(self, xt, T=-1, save_states=False, save_dir='posterior_state', **kwargs):
        T = self.T if T < 0 else T

        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            noise = self.noise_fn(x, t, **kwargs)
            x = self.reverse_posterior_step(x, noise, t)

            if save_states: # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x


    # sample ode using Black-box ODE solver (not used)
    def ode_sampler(self, xt, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3,):
        shape = xt.shape

        def to_flattened_numpy(x):
          """Flatten a torch tensor `x` and convert it to numpy."""
          return x.detach().cpu().numpy().reshape((-1,))

        def from_flattened_numpy(x, shape):
          """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
          return torch.from_numpy(x.reshape(shape))

        def ode_func(t, x):
            t = int(t)
            x = from_flattened_numpy(x, shape).to(self.device).type(torch.float32)
            score = self.score_fn(x, t)
            drift = self.ode_reverse_drift(x, score, t)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(ode_func, (self.T, eps), to_flattened_numpy(xt),
                                     rtol=rtol, atol=atol, method=method)

        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(self.device).type(torch.float32)

        return x

    def optimal_reverse(self, xt, x0, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            x = self.reverse_optimum_step(x, x0, t)

        return x

    ################################################################

    def weights(self, t):
        return torch.exp(-self.thetas_cumsum[t] * self.dt)


    def generate_random_states(self, x0, mu):
        x0 = x0.to(self.device)
        mu = mu.to(self.device)

        self.set_mu(mu)

        batch = x0.shape[0]

        timesteps = torch.randint(1, self.T + 1, (batch, 1, 1, 1)).long()
        # timesteps = torch.randint(1, 2, (batch, 1, 1, 1)).long()
        # t = torch.randint(1, self.T + 1)
        # t = torch.randint(1, self.T + 1, (1,)).item()  # 生成一个随机整数并提取标量

        noisy_states,noises = self.q_sample(x0,timesteps)

        # state_mean = self.mu_bar(x0, timesteps)
        # noises = torch.randn_like(state_mean)
        # noise_level = self.sigma_bar(timesteps)
        # noisy_states = noises * noise_level + state_mean

        return timesteps, noisy_states.to(torch.float32),noises

    def generate_random_states_multistep_unrolled_ori(self, x0, mu, k=3):
        x0 = x0.to(self.device)
        mu = mu.to(self.device)

        self.set_mu(mu)

        batch = x0.shape[0]

        timesteps = torch.randint(1, self.T + 1 -k -1, (batch, 1, 1, 1)).long()
        # timesteps = torch.randint(1, 2, (batch, 1, 1, 1)).long()
        # t = torch.randint(1, self.T + 1)
        # t = torch.randint(1, self.T + 1, (1,)).item()  # 生成一个随机整数并提取标量

        noisy_states,noises = self.q_sample(x0,timesteps)

        # state_mean = self.mu_bar(x0, timesteps)
        # noises = torch.randn_like(state_mean)
        # noise_level = self.sigma_bar(timesteps)
        # noisy_states = noises * noise_level + state_mean

        return timesteps, noisy_states.to(torch.float32),noises
    

    
    def generate_random_states_multistep_unrolled_ori_transition(self, x0, mu, k=0, type='half'):
        x0 = x0.to(self.device)
        mu = mu.to(self.device)

        self.set_mu(mu)

        batch = x0.shape[0]

        timesteps = torch.randint(1, self.T + 1 -k -1, (batch, 1, 1, 1)).long()

        noisy_states,noises = self.q_sample_transition(x0, mu, timesteps, type=type)

        return timesteps, noisy_states.to(torch.float32),noises
    
    def generate_random_states_ours(self, x0, mu, type='half'):
        x0 = x0.to(self.device)
        mu = mu.to(self.device)

        self.set_mu(mu)

        batch = x0.shape[0]

        timesteps = torch.randint(1, self.T, (batch, 1, 1, 1)).long()

        noisy_states,noises = self.q_sample_transition(x0, mu, timesteps, type=type)

        return timesteps, noisy_states.to(torch.float32),noises
    
    def generate_random_states_multistep_unrolled_ori_transition_reverse(self, x0, mu, k=3, type='half'):
        x0 = x0.to(self.device)
        mu = mu.to(self.device)

        self.set_mu(mu)

        batch = x0.shape[0]

        timesteps = torch.randint(1, self.T + 1 -k -1, (batch, 1, 1, 1)).long()

        noisy_states,noises = self.q_sample_transition_reverse(x0, mu, timesteps, type=type)

        return timesteps, noisy_states.to(torch.float32),noises
    

    # sample states for training
    def generate_random_states_traditional_reg(self, x0, mu):
        x0 = x0.to(self.device)
        mu = mu.to(self.device)

        self.set_mu(mu)

        batch = x0.shape[0]

        timesteps = torch.randint(self.T, self.T + 1, (batch, 1, 1, 1)).long()

        state_mean = self.mu_bar(x0, timesteps)
        noises = torch.randn_like(state_mean)
        noise_level = self.sigma_bar(timesteps)
        noisy_states = noises * noise_level + state_mean
    
        # return timesteps, noisy_states.to(torch.float32)
        return timesteps, noisy_states.to(torch.float32), noises

    # def noise_state(self, tensor):
    #     return tensor + torch.randn_like(tensor) * self.max_sigma


    def save_noisy_images(self, noisy_states, filename='noisy_image.png'):
        import torch
        import matplotlib.pyplot as plt
        import numpy as np
        # 如果 noisy_states 是 GPU 张量，先将其转到 CPU
        noisy_states = noisy_states.cpu()

        # 将 noisy_states 转换为 NumPy 数组
        noisy_states_np = noisy_states.squeeze().numpy()  # 去掉 batch 维度（假设是单张图）

        # 如果图像是多通道 (如 RGB)，调整维度 (C, H, W) -> (H, W, C)
        if noisy_states_np.ndim == 3 and noisy_states_np.shape[0] == 3:
            noisy_states_np = np.transpose(noisy_states_np, (1, 2, 0))

        # 创建绘图
        plt.imshow(noisy_states_np, cmap='gray')  # 对于灰度图像使用 cmap='gray'
        plt.axis('off')  # 不显示坐标轴

        # 保存图像
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()


    # def noise_state(self, tensor):
    #     return tensor + torch.randn_like(tensor) * self.max_sigma
    
    def noise_state(self, tensor, t = 100):
        batch = tensor.shape[0]
        timesteps = torch.tensor(t).long().view(batch, 1, 1, 1)
        noisy_states, noises = self.q_sample(tensor.to(self.device), timesteps)
        return noisy_states
        # return tensor + torch.randn_like(tensor) * self.max_sigma
    
    # 测试时，只加噪到中间某一步    
    def noise_state_single(self, tensor, t = 70):
        batch = tensor.shape[0]
        timesteps = torch.tensor(t).long().view(batch, 1, 1, 1)
        noisy_states, noises = self.q_sample(tensor.to(self.device), timesteps)
        return noisy_states
        # return tensor + torch.randn_like(tensor) * self.max_sigma


    # def noise_t_state(self, tensor, t):
    #     batch = tensor.shape[0]
    #     # timesteps = torch.full((batch, 1, 1, 1), self.T).long()
    #     timesteps = torch.full((batch, 1, 1, 1), t).long()
    #     tensor = tensor.to(self.device)
    #     noise_level = self.sigma_bar(timesteps).to(self.device)
    #     sqrt_alpha_bar = torch.exp(-self.thetas_cumsum[t] * self.dt).to(self.device)
        
    #     return tensor * sqrt_alpha_bar + torch.randn_like(tensor) * noise_level

    def noise_t_state(self, tensor, t):
        batch = tensor.shape[0]
        # timesteps = torch.full((batch, 1, 1, 1), self.T).long()
        timesteps = torch.full((batch, 1, 1, 1), t).long()
        tensor = tensor.to(self.device)
        noise_level = self.sigma_bar(timesteps).to(self.device)
        alpha_bar = torch.exp(-self.thetas_cumsum[t] * self.dt).to(self.device)
        
        return tensor * alpha_bar + torch.randn_like(tensor) * noise_level


    def inference_multi_steploss(self, xt, x_gt, name, begin=1, end=100, **kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)
        for t in tqdm(reversed(range(1, T + 1))):
            # score = self.score_fn(x, t, **kwargs)
            # x = self.reverse_sde_step(x, score, t)
            # x0_bar = self.get_init_state_from_noise(x, self.model(x, self.mu, t, **kwargs), t)
            x0_bar, x = self.p_sample(x, t, **kwargs)
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        # 画图部分
        plt.plot(multi_step[begin:end])
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Multi-step Loss Curve')
        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/super_resolution_gaussian_diffusion/multi_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突
        exit()
        return x

    # training and sampling discrepancy
    def inference_multi_steploss_x0_mix(self, xt, x_gt, name, begin=0, end=100, **kwargs):
        # 同时绘制多步和单步loss曲线
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt
        import numpy as np
        from tqdm import tqdm
        # 使用L1损失函数
        loss_fn = F.l1_loss
        multi_step = []
        multi_step_2 = []  # 第二条曲线的数据
        T = self.T
        x = xt.clone()

        # 计算第一条曲线
        print("curve in sampling")
        for t in tqdm(reversed(range(1,T + 1))):
            # score = self.score_fn(x, t, **kwargs)
            # x = self.reverse_sde_step(x, score, t)
            # x0_bar = self.get_init_state_from_noise(x, self.model(x, self.mu, t, **kwargs), t)
            x0_bar = self.model(x, self.mu, t, **kwargs)
            
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())

        print("curve in training")
        # 计算第二条曲线
        for t in tqdm(reversed(range(1, T + 1))):
            timesteps = torch.full((1, 1, 1, 1), t).long()
            # xt_temp = self.noise_t_state(x_gt, t)
            xt_temp,_ = self.q_sample(x_gt,timesteps)
            # score = self.score_fn(xt_temp, t, **kwargs)
            # x = self.reverse_sde_step(x, score, t)
            x0_bar_2 = self.model(xt_temp, self.mu, t, **kwargs)
            # x0_bar_2 = self.get_init_state_from_noise(xt_temp, self.model(xt_temp, self.mu, t, **kwargs), t)

            loss_multi_2 = loss_fn(x0_bar_2, x_gt, reduction='none')
            loss_multi_2 = einops.reduce(loss_multi_2, 'b ... -> b (...)', 'mean')
            multi_step_2.append(loss_multi_2.mean().item())
            print(loss_multi_2.mean().item())

        # 截取指定的迭代范围
        multi_step = multi_step[begin:end]
        multi_step_2 = multi_step_2[begin:end]
        timesteps = np.arange(begin, end)

        # 绘制图像
        plt.figure(figsize=(8, 6))

        # 设置全局字体大小
        plt.rc('font', size=14)           # 设置默认字体大小
        # plt.rc('axes', titlesize=16)       # 设置标题字体大小
        plt.rc('axes', labelsize=16)       # 设置坐标轴标签字体大小
        plt.rc('xtick', labelsize=12)      # 设置 x 轴刻度字体大小
        plt.rc('ytick', labelsize=12)      # 设置 y 轴刻度字体大小
        plt.rc('legend', fontsize=14)      # 设置图例字体大小

        # 绘制第一条曲线及其底色
        plt.plot(timesteps, multi_step, label="L1 Loss during Sampling", color='#6A5ACD', linestyle='-', marker='o')  # 深紫色

        # 绘制第二条曲线及其底色
        plt.plot(timesteps, multi_step_2, label="L1 Loss during Training", color='#FFA07A', linestyle='--', marker='x')  # 橙色

        # 添加两条曲线之间的填充区域，不设置标签
        plt.fill_between(timesteps, multi_step, multi_step_2, where=(np.array(multi_step) > np.array(multi_step_2)),
                 color='#DDA0DD', alpha=0.1)  # 使用淡紫色（Plum）作为填充区域



        # 添加图例和标签
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        # plt.title('Multi-step Loss Curve Comparison')
        plt.legend()
        plt.grid(True)

        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp1104_sr/multi_step_ddpm_{name}_sr_comparison_{begin}_{end}.png')

        plt.close()  # 关闭图形，防止后续绘图冲突
                # 保存 multi_step 和 multi_step_2
        np.save(f'/home/proj/image-restoration-sde-main/codes/temp1104_sr/ddpm{name}_step_{begin}_{end}.npy', np.array(multi_step))
        np.save(f'/home/proj/image-restoration-sde-main/codes/temp1104_sr/ddpm{name}_step_2_{begin}_{end}.npy', np.array(multi_step_2))
        exit(0)
        return x


    

    def inference_multi_steploss_scaling(self, xt, x_gt, name, begin=1, end=100, **kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        multi_step_2 = []
        multi_step_3 = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)
        
        # 第一条曲线
        for t in tqdm(reversed(range(1, T + 1))):
            x0_bar, x = self.p_sample(x, t, **kwargs)
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        
        plt.plot(multi_step[begin:end], label='scaler=1')  # 添加 label

        # 第二条曲线
        for t in tqdm(reversed(range(1, T + 1))):
            x0_bar, x = self.p_sample_scaling(x, t, eps_scaler=1.01, **kwargs)
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step_2.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        
        plt.plot(multi_step_2[begin:end], label='scaler=1.01')  # 添加 label

        # 第三条曲线
        for t in tqdm(reversed(range(1, T + 1))):
            x0_bar, x = self.p_sample_scaling(x, t, eps_scaler=0.99, **kwargs)
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step_3.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        
        plt.plot(multi_step_3[begin:end], label='scaler=0.99')  # 添加 label

        # 添加图例、坐标轴标签和标题
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Multi-step Loss Curve')
        plt.grid(True)

        # 显示图例
        plt.legend()

        # 显示和保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/super_resolution_gaussian_diffusion/scaling_multi_step_{name}_{begin}_{end}.png')
        plt.show()
        plt.close()
        
        exit()
        return x

    def inference_multi_steploss_x0_scaling(self, xt, x_gt, name, begin=1, end=100, **kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        multi_step_2 = []
        multi_step_3 = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)
        
        # 第一条曲线
        for t in tqdm(reversed(range(1, T + 1))):
            x0_bar, x = self.p_sample_x0(x, t, **kwargs)
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        
        plt.plot(multi_step[begin:end], label='eps_scaler=1')  # 添加 label

        # 第二条曲线
        for t in tqdm(reversed(range(1, T + 1))):
            x0_bar, x = self.p_sample_x0_scaling(x, t, eps_scaler=1.01, **kwargs)
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step_2.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        
        plt.plot(multi_step_2[begin:end], label='eps_scaler=1.01')  # 添加 label

        # 第三条曲线
        for t in tqdm(reversed(range(1, T + 1))):
            x0_bar, x = self.p_sample_x0_scaling(x, t, eps_scaler=0.99, **kwargs)
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step_3.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        
        plt.plot(multi_step_3[begin:end], label='eps=0.99')  # 添加 label

        # 添加图例、坐标轴标签和标题
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Multi-step Loss Curve')
        plt.grid(True)

        # 显示图例
        plt.legend()

        # 显示和保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/super_resolution_gaussian_diffusion/scaling_multi_step_{name}_{begin}_{end}.png')
        plt.show()
        plt.close()
        
        exit()
        return x

    def inference_multi_steploss_x0(self, xt, x_gt, name, begin=1, end=100, **kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)
        for t in tqdm(reversed(range(1, T + 1))):
            # score = self.score_fn(x, t, **kwargs)
            # x = self.reverse_sde_step(x, score, t)
            # x0_bar = self.get_init_state_from_noise(x, self.model(x, self.mu, t, **kwargs), t)
            x0_bar,x = self.p_sample_x0(x, t, **kwargs)
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        torch.save(multi_step, '/home/proj/image-restoration-sde-main/codes/unfolded.pt')
        # 将列表保存到 num_list.txt 文件
        # 画图部分
        plt.plot(multi_step[begin:end])
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Multi-step Loss Curve')
        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/super_resolution_gaussian_diffusion/multi_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突
        exit()
        return x

    

    def inference_single_steploss(self, xt, x_gt, name, begin=1, end=100, **kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)
        for t in tqdm(reversed(range(1, T + 1))):
            
            xt_temp = self.noise_t_state(x_gt, t)
            x0_bar,_ = self.p_sample(x, t, **kwargs)
            score = self.score_fn(x, t, **kwargs)
            # x = self.reverse_sde_step(x, score, t)
            x0_bar = self.get_init_state_from_noise(xt_temp, self.model(xt_temp, self.mu, t, **kwargs), t)
            
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        

        # with open('/home/proj/image-restoration-sde-main/codes/ours.txt', 'w') as f:
        #     for num in multi_step:
        #         f.write(f"{num}\n")  # 每个数字写一行
        # 画图部分
        plt.plot(multi_step[begin:end])
       

        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Single-step Loss Curve')
        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp/single_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突
        exit()
        return x

    def inference_single_steploss_x0(self, xt, x_gt, name, begin=1, end=30,**kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)
        for t in tqdm(reversed(range(1, T + 1))):     
            xt_temp = self.noise_t_state(x_gt, t)
            x0_bar = self.noise_fn(xt_temp, t, **kwargs)
            # score = self.score_fn(x, t, **kwargs)
            # # x = self.reverse_sde_step(x, score, t)
            # x0_bar = self.get_init_state_from_noise(xt_temp, self.model(xt_temp, self.mu, t, **kwargs), t)
            
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        
        # 画图部分
        plt.plot(multi_step[begin:end])
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Single-step Loss Curve')
        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp/single_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突
        exit()
        return x

    def inference_single_steploss_x0_c(self, xt, x_gt, name, condition=None, begin=70, end=100,**kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)
        for t in tqdm(reversed(range(1, T + 1))):     
            xt_temp = self.noise_t_state(x_gt, t)
            x0_bar_and_c = self.noise_fn(xt_temp, t, **kwargs)
            x0_bar = x0_bar_and_c * 2 - condition
            # score = self.score_fn(x, t, **kwargs)
            # # x = self.reverse_sde_step(x, score, t)
            # x0_bar = self.get_init_state_from_noise(xt_temp, self.model(xt_temp, self.mu, t, **kwargs), t)
            
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        
        # 画图部分
        plt.plot(multi_step[begin:end])
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Single-step Loss Curve')
        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp/single_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突
        exit()
        return x
    

    def inference_multi_steploss_traditional_reg(self, xt, x_gt, name, begin=1, end=100, **kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)
        for t in tqdm(reversed(range(T, T + 1))):
            # 此处直接用conditionalUnet预测x0_bar即可，无需转换为score
            x0_bar = self.noise_fn(x, t, **kwargs)
            # score = self.score_fn(x, t, **kwargs)
            # x = self.reverse_sde_step(x, score, t)
            # x0_bar = self.get_init_state_from_noise(x, self.model(x, self.mu, t, **kwargs), t)
            
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        # 画图部分

        # 将判别式模型的单步结果复制为多步
        multi_step = multi_step * (end-begin-1)
        plt.plot(multi_step[begin:end])
        # plt.plot (multi_step[begin:end])
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Multi-step Loss Curve')
        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp/multi_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突

        # 保存 xt 和 x_gt 的图像
        # x0_bar = 1-(x0_bar - x0_bar.min()) / (x0_bar.max() - x0_bar.min())
        # # x0_bar = (x0_bar + 1) / 2 
        self.save_image(x0_bar , f'/home/proj/image-restoration-sde-main/codes/temp/x0_bar_{name}.png')
        self.save_image(x_gt, f'/home/proj/image-restoration-sde-main/codes/temp/x_gt_{name}.png')

        exit()
        return x

    def inference_multi_steploss_traditional_reg_c(self, xt, x_gt, name, condition=None, begin=1, end=100, **kwargs):
        import torch.nn.functional as F
        import einops
        import matplotlib.pyplot as plt

        loss_fn = F.l1_loss
        multi_step = []
        T = self.T 
        x = xt.clone()
        print(xt.shape)
        print(x_gt.shape)
        for t in tqdm(reversed(range(T, T + 1))):
            # 此处直接用conditionalUnet预测x0_bar即可，无需转换为score
            x0_bar_and_c = self.noise_fn(x, t, **kwargs)
            x0_bar = x0_bar_and_c * 2 - condition           
            loss_multi = loss_fn(x0_bar, x_gt, reduction='none')
            loss_multi = einops.reduce(loss_multi, 'b ... -> b (...)', 'mean')
            multi_step.append(loss_multi.mean().item())
            print(loss_multi.mean().item())
        # 画图部分

        # 将判别式模型的单步结果复制为多步
        multi_step = multi_step * (end-begin-1)
        plt.plot(multi_step[begin:end])
        # plt.plot (multi_step[begin:end])
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Multi-step Loss Curve')
        plt.grid(True)
        plt.show()
        # 保存图像
        plt.savefig(f'/home/proj/image-restoration-sde-main/codes/temp/multi_step_{name}_{begin}_{end}.png')  # 以PNG格式保存，分辨率300 DPI
        plt.close()  # 关闭图形，防止后续绘图冲突

        # self.save_image(x0_bar , f'/home/proj/image-restoration-sde-main/codes/temp/x0_bar_{name}.png')
        # self.save_image(x_gt, f'/home/proj/image-restoration-sde-main/codes/temp/x_gt_{name}.png')

        exit()
        return x

    # 保存单张图像的函数
    def save_image(self, tensor, filename, cmap='gray'):
        import torch
        import matplotlib.pyplot as plt
        import numpy as np

        # 将张量移动到 CPU 并转换为 NumPy 格式
        tensor = tensor.cpu().squeeze().numpy()  # 假设 tensor 形状是 (1, C, H, W) 或 (C, H, W)

        # 如果是彩色图像 (C, H, W)，则转置为 (H, W, C)
        if tensor.ndim == 3 and tensor.shape[0] == 3:
            tensor = np.transpose(tensor, (1, 2, 0))

        # 绘制图像
        plt.imshow(tensor, cmap=cmap)
        plt.axis('off')  # 关闭坐标轴显示

        # 保存图像
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    # res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    # Check if arr is already a tensor, if not convert it from NumPy to tensor
    if isinstance(arr, np.ndarray):
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    else:  # If it's already a tensor, no need to convert
        res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


    

