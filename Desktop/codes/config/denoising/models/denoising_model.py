import logging
from collections import OrderedDict
import os
import numpy as np

import math
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm
from ema_pytorch import EMA

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion

from models.modules.loss import MatchingLoss

from .base_model import BaseModel
import matplotlib.pyplot as plt

from torchvision.transforms import transforms


logger = logging.getLogger("base")


class DenoisingModel(BaseModel):
    def __init__(self, opt):
        super(DenoisingModel, self).__init__(opt)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.model = networks.define_G(opt).to(self.device)
        if opt["dist"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[torch.cuda.current_device()]
            )
        else:
            self.model = DataParallel(self.model)
        # print network
        # self.print_network()
        self.load()

        if self.is_train:
            self.model.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            self.weight = opt['train']['weight']

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (
                k,
                v,
            ) in self.model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            if train_opt['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'Lion':
                self.optimizer = Lion(
                    optim_params, 
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print('Not implemented optimizer, default using Adam!')

            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, 
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"])
                    ) 
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()

    def feed_data(self, state, LQ, GT=None):
        self.state = state.to(self.device)    # noisy_state
        self.condition = LQ.to(self.device)  # LQ
        if GT is not None:
            self.state_0 = GT.to(self.device)  # GT
    
    def save_images(self,images, title, filename):
        """
        Save a grid of images to a file.
        
        Args:
            images (torch.Tensor): A tensor of shape [N, C, H, W] where N is the number of images.
            title (str): Title for the grid of images.
            filename (str): Path to save the output image.
        """
        # Convert tensor to NumPy array
        images = images.cpu().detach().numpy()
        
        # Create a grid of images
        N = images.shape[0]  # Number of images
        C, H, W = images.shape[1], images.shape[2], images.shape[3]
        
        fig, axes = plt.subplots(1, N, figsize=(N*4, 4))
        for i in range(N):
            ax = axes[i]
            img = images[i].transpose(1, 2, 0)  # Change to HWC format
            if C == 1:
                img = img.squeeze(-1)  # Remove the channel dimension if grayscale
            ax.imshow(img)
            ax.axis('off')
        plt.suptitle(title)
        plt.savefig(filename)
        plt.close()


    def optimize_parameters_x0_ours(self, step, timesteps, sde=None):
        sde.set_mu(self.condition)

        self.optimizer.zero_grad()

        timesteps = timesteps.to(self.device)
        t_1 = timesteps.squeeze()
        # 生成 batch 个随机数，形状为 (batch,)
        batch = self.state.shape[0]

        t_2 = torch.zeros_like(t_1)
        for i in range(batch):
            t_2[i] = torch.randint(1, t_1[i]+1, (1,)).long()
        timesteps_2 = t_2.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(self.device)

        x0_bar = sde.noise_fn(self.state, timesteps.squeeze())
        state_2,_ = sde.q_sample(x0_bar,timesteps_2)
        x0_bar_bar = sde.noise_fn(state_2, timesteps_2.squeeze())

        loss = self.loss_fn(x0_bar, self.state_0) + self.loss_fn(x0_bar_bar, self.state_0)

        loss.backward()
        self.optimizer.step()
        self.ema.update()

        # set log
        self.log_dict["loss"] = loss.item()

    def optimize_parameters_epsilon_unfolded(self, step, timesteps, sde=None, noise_gt=None):
        sde.set_mu(self.condition)

        self.optimizer.zero_grad()

        timesteps = timesteps.to(self.device)
        t_1 = timesteps.squeeze()
        # 生成 batch 个随机数，形状为 (batch,)
        batch = self.state.shape[0]

        t_2 = torch.zeros_like(t_1)
        for i in range(batch):
            t_2[i] = torch.randint(1, t_1[i]+1, (1,)).long()
        timesteps_2 = t_2.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(self.device)

        epsilon_bar = sde.noise_fn(self.state, timesteps.squeeze())
        x0_bar = sde.predict_start_from_noise(self.state, timesteps, epsilon_bar)
        state_2,noise_gt_2 = sde.q_sample(x0_bar,timesteps_2)
        epsilon_bar_bar = sde.noise_fn(state_2, timesteps_2.squeeze())

        #获取x_t_sub_1

        loss = self.loss_fn(epsilon_bar, noise_gt) + self.loss_fn(epsilon_bar_bar, noise_gt_2)

        loss.backward()
        self.optimizer.step()
        self.ema.update()

        # set log
        self.log_dict["loss"] = loss.item()


    def optimize_parameters_x0_multi_step_unrolled(self, step, timesteps, sde=None, k=3):
        sde.set_mu(self.condition)

        self.optimizer.zero_grad()

        timesteps = timesteps.to(self.device)
        t_1 = timesteps.squeeze()
        batch = self.state.shape[0]

        t_2 = torch.zeros_like(t_1)
        for i in range(batch):
            upper_limit = t_1[i].item() + 1 - k - 1
            if upper_limit > 1:  # 边界检查
                t_2[i] = torch.randint(1, upper_limit, (1,)).long()
            else:
                t_2[i] = 1  # 作为fallback

        timesteps_2 = t_2.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(self.device)

        x0_bar = sde.noise_fn(self.state, timesteps.squeeze())
        state_2, _ = sde.q_sample(x0_bar, timesteps_2)
        x0_bar_bar = sde.noise_fn(state_2, timesteps_2.squeeze())

        x_tk1, _ = sde.q_sample(self.state_0, (timesteps_2) + k + 1)
        x_tk1_bar, _ = sde.q_sample(x0_bar, (timesteps_2) + k + 1)
        x_tk_bar_bar = sde.q_posterior_mean(x0_bar_bar, x_tk1_bar, (timesteps_2) + k + 1)
        x_tk = sde.q_posterior_mean(self.state_0, x_tk1, (timesteps_2) + k + 1)

        if k > 0:
            x_t_bar_bar = sde.multi_step_fuse(x_tk_bar_bar, x0_bar, timesteps_2, k)
            x_t = sde.multi_step_fuse(x_tk, self.state_0, timesteps_2, k)

        loss = self.loss_fn(x0_bar, self.state_0) + self.loss_fn(x_t_bar_bar, x_t) + self.loss_fn(x0_bar_bar, self.state_0)

        loss.backward()
        self.optimizer.step()
        self.ema.update()

        self.log_dict["loss"] = loss.item()


    def optimize_parameters_epsilon_multi_step_unrolled(self, step, timesteps, sde=None, noise_gt=None, k=3):
        sde.set_mu(self.condition)

        self.optimizer.zero_grad()

        timesteps = timesteps.to(self.device)
        t_1 = timesteps.squeeze()
        batch = self.state.shape[0]

        t_2 = torch.zeros_like(t_1)
        for i in range(batch):
            upper_limit = t_1[i].item() + 1 - k - 1
            if upper_limit > 1:  # 边界检查
                t_2[i] = torch.randint(1, upper_limit, (1,)).long()
            else:
                t_2[i] = 1  # 作为fallback

        timesteps_2 = t_2.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(self.device)

        # x0_bar = sde.noise_fn(self.state, timesteps.squeeze())
        noise_bar = sde.noise_fn(self.state, timesteps.squeeze())
        x0_bar = sde.predict_start_from_noise(self.state, timesteps, noise_bar)
        state_2, noise_gt_2 = sde.q_sample(x0_bar, timesteps_2)
        # x0_bar_bar = sde.noise_fn(state_2, timesteps_2.squeeze())
        noise_bar_bar = sde.noise_fn(state_2, timesteps_2.squeeze())
        x0_bar_bar = sde.predict_start_from_noise(state_2, timesteps_2, noise_bar_bar)


        x_tk1, _ = sde.q_sample(self.state_0, (timesteps_2) + k + 1)
        x_tk1_bar, _ = sde.q_sample(x0_bar, (timesteps_2) + k + 1)
        x_tk_bar_bar = sde.q_posterior_mean(x0_bar_bar, x_tk1_bar, (timesteps_2) + k + 1)
        x_tk = sde.q_posterior_mean(self.state_0, x_tk1, (timesteps_2) + k + 1)

        if k > 0:
            x_t_bar_bar = sde.multi_step_fuse(x_tk_bar_bar, x0_bar, timesteps_2, k)
            x_t = sde.multi_step_fuse(x_tk, self.state_0, timesteps_2, k)

        loss = self.loss_fn(noise_bar, noise_gt) + self.loss_fn(x_t_bar_bar, x_t) + self.loss_fn(noise_bar_bar, noise_gt_2)

        loss.backward()
        self.optimizer.step()
        self.ema.update()

        self.log_dict["loss"] = loss.item()

    
    def optimize_parameters_x0_multi_step_unrolled_ori(self, step, timesteps, sde=None, k=3):
        sde.set_mu(self.condition)

        self.optimizer.zero_grad()

        timesteps = timesteps.to(self.device)
        t_1 = timesteps.squeeze()
        batch = self.state.shape[0]

        t_2 = torch.zeros_like(t_1)
        for i in range(batch):
            upper_limit = t_1[i].item() + 1
            if upper_limit > 1:  # 边界检查
                t_2[i] = torch.randint(1, upper_limit, (1,)).long()
            else:
                t_2[i] = 1  # 作为fallback

        timesteps_2 = t_2.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(self.device)

        x0_bar = sde.noise_fn(self.state, timesteps.squeeze())
        state_2, _ = sde.q_sample(x0_bar, timesteps_2)
        x0_bar_bar = sde.noise_fn(state_2, timesteps_2.squeeze())

        # !!!!!
        x_tk1, _ = sde.q_sample(self.state_0, (timesteps) + k + 1)
        x_tk_bar = sde.q_posterior_mean(x0_bar, x_tk1, (timesteps) + k + 1)
        x_tk = sde.q_posterior_mean(self.state_0, x_tk1, (timesteps) + k + 1)

        if k > 0:
            x_t_bar = sde.multi_step_fuse(x_tk_bar, self.state_0, timesteps, k)
            x_t = sde.multi_step_fuse(x_tk, self.state_0, timesteps, k)

        loss = self.loss_fn(x0_bar, self.state_0) + self.loss_fn(x_t_bar, x_t) + self.loss_fn(x0_bar_bar, self.state_0)

        loss.backward()
        self.optimizer.step()
        self.ema.update()

        self.log_dict["loss"] = loss.item()


    def optimize_parameters_x0_unfolded_transition(self, step, timesteps, sde=None, type='half', phase=1):

        sde.set_mu(self.condition)

        self.optimizer.zero_grad()

        timesteps = timesteps.to(self.device)
        t_1 = timesteps.squeeze()
        batch = self.state.shape[0]

        t_2 = torch.zeros_like(t_1)
        for i in range(batch):
            upper_limit = t_1[i].item() + 1
            if upper_limit > 1:  # 边界检查
                t_2[i] = torch.randint(1, upper_limit, (1,)).long()
            else:
                t_2[i] = 1  # 作为fallback

        timesteps_2 = t_2.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(self.device)

        x0_bar = sde.noise_fn(self.state, timesteps.squeeze())
        if phase == 2:
            state_2, _ = sde.q_sample_transition(x0_bar, self.condition, timesteps_2, type=type)
        else:
            state_2, _ = sde.q_sample(x0_bar, timesteps_2)

        x0_bar_bar = sde.noise_fn(state_2, timesteps_2.squeeze())

        # # !!!!!
        # # 是否应该transition？
        # x_tk1, _ = sde.q_sample_transition(self.state_0, self.condition, (timesteps) + k + 1, type=type)
        # x_tk_bar = sde.q_posterior_mean(x0_bar, x_tk1, (timesteps) + k + 1)
        # x_tk = sde.q_posterior_mean(self.state_0, x_tk1, (timesteps) + k + 1)

        # if k > 0:
        #     x_t_bar = sde.multi_step_fuse(x_tk_bar, self.state_0, timesteps, k)
        #     x_t = sde.multi_step_fuse(x_tk, self.state_0, timesteps, k)

        loss = self.loss_fn(x0_bar, self.state_0) +  self.loss_fn(x0_bar_bar, self.state_0)

        # loss = self.loss_fn(x0_bar, self.state_0) + self.loss_fn(x_t_bar, x_t) + self.loss_fn(x0_bar_bar, self.state_0)

        loss.backward()
        self.optimizer.step()
        self.ema.update()

        self.log_dict["loss"] = loss.item()

    def optimize_parameters(self, step, timesteps, sde=None, noise_gt=None):
        sde.set_mu(self.condition)

        self.optimizer.zero_grad()

        timesteps = timesteps.to(self.device)

        # Get noise and score
        noise = sde.noise_fn(self.state, timesteps.squeeze())
        # loss = self.weight * self.loss_fn(noise, noise_gt)
        score = sde.get_score_from_noise(noise, timesteps)

        # Learning the maximum likelihood objective for state x_{t-1}
        xt_1_expection = sde.reverse_sde_step_mean(self.state, score, timesteps)
        xt_1_optimum = sde.reverse_optimum_step(self.state, self.state_0, timesteps)
        loss = self.weight * self.loss_fn(xt_1_expection, xt_1_optimum)

        loss.backward()
        self.optimizer.step()
        self.ema.update()

        # set log
        self.log_dict["loss"] = loss.item()



    def test(self, sde=None, save_states=False):
        sde.set_mu(self.condition)

        self.model.eval()
        with torch.no_grad():
            self.output = sde.reverse_sde(self.state, save_states=save_states)
            # self.output = sde.reverse_sde(self.state, save_states=save_states, self.condition)

        self.model.train()

    def test_visual(self, current_step=-1, sde=None, save_states=False, name=None):
        sde.set_mu(self.condition)

        self.model.eval()
        with torch.no_grad():
            # self.output = sde.inference_multi_steploss(self.state, self.state_0, name=name)
            # self.output = sde.inference_single_steploss(self.state, self.state_0, name=name)

            self.output = sde.reverse_sde_visual(self.state, current_step=current_step, save_states=save_states)
            # self.output = sde.reverse_sde(self.state, save_states=save_states, self.condition)

        self.model.train()

    def test_visual_x0(self, current_step=-1, sde=None, save_states=False, name=None,t=None):
        sde.set_mu(self.condition)

        self.model.eval()
        with torch.no_grad():
            self.output = sde.reverse_sde_visual_x0(self.state, current_step=current_step, save_states=save_states)
        self.model.train()



    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.condition.detach()[0].float().cpu()
        out_dict["Output"] = self.output.detach()[0].float().cpu()
        if need_GT:
            out_dict["GT"] = self.state_0.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(
            self.model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.model.__class__.__name__, self.model.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.model, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        # self.save_network(self.model, "G", iter_label)
        self.save_network(self.model, "G", 'best')
        # self.save_network(self.ema.ema_model, "EMA", 'lastest')
