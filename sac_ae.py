from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import copy
import math

import utils
from encoder import make_encoder
from decoder import make_decoder

LOG_FREQ = 10000


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, compute_log_prob: bool = True,
                   dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      compute_log_prob: if ``True``, the logarithm of probability of sample is returned
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution with its log probabilities.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.
    """

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    log_prob = None
    if compute_log_prob:
        log_prob = -torch.sum(-ret * F.log_softmax(logits, dim=dim), dim=dim)
    return ret, log_prob


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class ActorDiscrete(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_dim, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters, gumbel='none', temperature=1.0,
    ):
        super().__init__()
        self.gumbel = gumbel
        if gumbel not in ('none', 'soft', 'hard', 'straight-through'):
            raise ValueError(f'Unexpected value of gumbel parameter:', gumbel)
        self.temperature = temperature

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.log_softmax_temp = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False, detach_logit_log_pi=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)
        logit_unscaled = self.trunk(obs)
        softmax_temp = torch.exp(self.log_softmax_temp(obs))
        logit = logit_unscaled / softmax_temp
        logit_log_pi = logit
        if detach_logit_log_pi:
            assert self.gumbel == 'none'
            logit_log_pi = logit_unscaled.detach() / softmax_temp
        self.outputs['logit'] = logit

        pi = None
        log_pi = None
        if self.gumbel == 'none' or self.gumbel == 'straight-through':
            if compute_pi:
                pi = F.softmax(logit, dim=1)
                if compute_log_pi:
                    log_pi = F.log_softmax(logit_log_pi, dim=1)

                if self.gumbel == 'straight-through':
                    m = torch.distributions.Categorical(probs=pi)
                    sample = m.sample().to(logit.device)
                    one_hot = F.one_hot(sample, num_classes=logit.size()[1]).to(torch.float32)
                    pi = one_hot + pi - pi.detach()
                    if compute_log_pi:
                        log_pi = log_pi.gather(1, sample.unsqueeze(-1)).squeeze(dim=1)

        else:
            if compute_pi:
                pi, log_pi = gumbel_softmax(logit, self.temperature, hard=self.gumbel == 'hard',
                                            compute_log_prob=compute_log_pi, dim=1)

        return logit, pi, log_pi

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class QFunctionDiscrete(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs, action=None):
        return self.trunk(obs)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters
    ):
        super().__init__()


        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class CriticDiscrete(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_dim, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters, gumbel='none'
    ):
        super().__init__()
        self.gumbel = gumbel
        assert gumbel in ('none', 'soft', 'hard', 'straight-through')

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        if self.gumbel != 'none':
            clazz = QFunction
        else:
            clazz = QFunctionDiscrete

        self.Q1 = clazz(
            self.encoder.feature_dim, action_dim, hidden_dim
        )
        self.Q2 = clazz(
            self.encoder.feature_dim, action_dim, hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class SacAeAgent(object):
    """SAC+AE algorithm."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        decoder_type='pixel',
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_latent_lambda=0.0,
        decoder_weight_lambda=0.0,
        num_layers=4,
        num_filters=32
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.decoder_latent_lambda = decoder_latent_lambda

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        self.decoder = None
        if decoder_type != 'identity':
            # create decoder
            self.decoder = make_decoder(
                decoder_type, obs_shape, encoder_feature_dim, num_layers,
                num_filters
            ).to(device)
            self.decoder.apply(weight_init)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            # optimizer for decoder
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(),
                lr=decoder_lr,
                weight_decay=decoder_weight_lambda
            )

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.decoder is not None:
            self.decoder.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        L.log('train_critic/loss', critic_loss, step)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        L.log('train_actor/loss', actor_loss, step)
        L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        L.log('train_alpha/loss', alpha_loss, step)
        L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_decoder(self, obs, target_obs, L, step):
        h = self.critic.encoder(obs)

        if target_obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = utils.preprocess_obs(target_obs)
        rec_obs = self.decoder(h)
        rec_loss = F.mse_loss(target_obs, rec_obs)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        loss = rec_loss + self.decoder_latent_lambda * latent_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        L.log('train_ae/ae_loss', loss, step)

        self.decoder.log(L, step, log_freq=LOG_FREQ)

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

        if self.decoder is not None and step % self.decoder_update_freq == 0:
            self.update_decoder(obs, obs, L, step)

    def save(self, model_dir, step):
        if step is not None:
            step_suffix = f'_{step}'
        else:
            step_suffix = ''

        torch.save(
            self.actor.state_dict(), f'{model_dir}/actor{step_suffix}.pt'
        )
        torch.save(
            self.critic.state_dict(), f'{model_dir}/critic{step_suffix}.pt'
        )
        torch.save(
            self.actor_optimizer.state_dict(), f'{model_dir}/actor_optimizer{step_suffix}.pt'
        )
        torch.save(
            self.critic_optimizer.state_dict(), f'{model_dir}/critic_optimizer{step_suffix}.pt'
        )
        torch.save(
            self.log_alpha_optimizer.state_dict(), f'{model_dir}/log_alpha_optimizer{step_suffix}.pt'
        )
        if self.decoder is not None:
            torch.save(
                self.decoder.state_dict(), f'{model_dir}/decoder{step_suffix}.pt'
            )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        if self.decoder is not None:
            self.decoder.load_state_dict(
                torch.load('%s/decoder_%s.pt' % (model_dir, step))
            )


class SacAeAgentDiscrete(object):
    """SAC+AE algorithm."""
    def __init__(
        self,
        obs_shape,
        action_dim,
        device,
        entropy_scheduler,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        auto_alpha=0,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_update_freq=2,
        encoder='critic',
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        decoder_type='pixel',
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_latent_lambda=0.0,
        decoder_weight_lambda=0.0,
        num_layers=4,
        num_filters=32,
        gumbel='none',
        temperature=1.0,
        detach_logit_log_pi=False,
    ):
        self.action_dim = action_dim
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.decoder_latent_lambda = decoder_latent_lambda
        self.encoder = encoder
        self.gumbel = gumbel
        self.temperature = temperature
        self.entropy_scheduler = entropy_scheduler
        self.detach_logit_log_pi = detach_logit_log_pi

        self.actor = ActorDiscrete(
            obs_shape, action_dim, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, self.gumbel, self.temperature
        ).to(device)

        self.critic = CriticDiscrete(
            obs_shape, action_dim, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, self.gumbel
        ).to(device)

        self.encoders = {}
        # tie encoders between actor and critic
        if self.encoder == 'actor':
            self.critic.encoder.copy_conv_weights_from(self.actor.encoder)
            encoder = self.actor.encoder
            self.encoders['actor'] = encoder
        elif encoder == 'critic':
            self.actor.encoder.copy_conv_weights_from(self.critic.encoder)
            encoder = self.critic.encoder
            self.encoders['critic'] = encoder
        elif encoder == 'both':
            self.encoders['actor'] = self.actor.encoder
            self.encoders['critic'] = self.critic.encoder

        self.critic_target = CriticDiscrete(
            obs_shape, action_dim, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, self.gumbel
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.auto_alpha = auto_alpha
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = self.auto_alpha > 0
        # set target entropy to -|A|
        self.entropy_scheduler.set_target_entropy(0.98 * np.log(action_dim) * self.auto_alpha)
        self.min_alpha = 0.001
        self.log_min_alpha = torch.tensor(np.log(self.min_alpha)).to(device)

        self.decoders = {}
        self.encoder_optimizers = {}
        self.decoder_optimizers = {}
        if decoder_type != 'identity':
            for key in self.encoders.keys():
                # create decoder
                decoder = make_decoder(
                    decoder_type, obs_shape, encoder_feature_dim, num_layers,
                    num_filters
                ).to(device)
                decoder.apply(weight_init)
                self.decoders[key] = decoder

                # optimizer for critic encoder for reconstruction loss
                self.encoder_optimizers[key] = torch.optim.Adam(
                    self.encoders[key].parameters(), lr=encoder_lr
                )

                # optimizer for decoder
                self.decoder_optimizers[key] = torch.optim.Adam(
                    self.decoders[key].parameters(),
                    lr=decoder_lr,
                    weight_decay=decoder_weight_lambda
                )

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        for decoder in self.decoders.values():
            decoder.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            logit, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return logit.max(dim=1).indices.item()

    def sample_action(self, obs, return_entropy=False):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            if self.gumbel != 'none':
                logit, _, log_pi = self.actor(
                    obs, compute_pi=return_entropy, compute_log_pi=return_entropy
                )
                pi = F.softmax(logit, dim=1)
            else:
                _, pi, log_pi = self.actor(obs, compute_log_pi=return_entropy)

            entropy = None
            if return_entropy:
                entropy = -torch.sum(pi * log_pi, dim=1).item()

            action = Categorical(pi).sample()
            return action.item(), entropy

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, pi, log_pi = self.actor(next_obs, detach_encoder=True)
            if self.gumbel != 'none':
                target_Q1, target_Q2 = self.critic_target(next_obs, action=pi, detach_encoder=True)
                target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi.unsqueeze(dim=1)
            else:
                target_Q1, target_Q2 = self.critic_target(next_obs, action=None, detach_encoder=True)
                target_V = torch.sum(pi * (torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi), dim=1, keepdim=True)
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        if self.gumbel != 'none':
            action = F.one_hot(action.squeeze(dim=1).long(), num_classes=self.action_dim).to(torch.float32).to(self.device)
            current_Q1, current_Q2 = self.critic(obs, action=action, detach_encoder=self.encoder == 'actor')
        else:
            current_Q1, current_Q2 = self.critic(obs, action=None, detach_encoder=self.encoder == 'actor')
            current_Q1 = current_Q1.gather(1, action.long())
            current_Q2 = current_Q2.gather(1, action.long())
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        L.log('train_critic/loss', critic_loss, step)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        logit, pi, log_pi = self.actor(
            obs, detach_encoder=self.encoder == 'critic', detach_logit_log_pi=self.detach_logit_log_pi
        )
        if self.gumbel != 'none':
            actor_Q1, actor_Q2 = self.critic(obs, action=pi, detach_encoder=True)
        else:
            actor_Q1, actor_Q2 = self.critic(obs, action=None, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = self.alpha.detach() * log_pi.unsqueeze(dim=1) - actor_Q
        if self.gumbel == 'none':
            actor_loss = pi * actor_loss
            actor_loss = actor_loss.sum(dim=1)
        actor_loss = actor_loss.mean()

        L.log('train_actor/loss', actor_loss, step)
        L.log('train_actor/target_entropy', self.entropy_scheduler.get_target_entropy(), step)
        if self.gumbel != 'none':
            entropy = -torch.sum(F.softmax(logit, dim=1) * F.log_softmax(logit, dim=1), dim=1)
            L.log('train_actor/entropy', -log_pi.mean(), step)
            L.log('train_actor/entropy_orig', entropy.mean(), step)
        else:
            entropy = -torch.sum(pi * log_pi, dim=1)
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        if self.auto_alpha > 0:
            self.log_alpha_optimizer.zero_grad()
            if self.gumbel != 'none':
                alpha_loss = (self.alpha * (-log_pi - self.entropy_scheduler.get_target_entropy()).detach()).mean()
            else:
                alpha_loss = (self.alpha * (entropy - self.entropy_scheduler.get_target_entropy()).detach()).mean()
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            if torch.all(self.log_alpha < self.log_min_alpha).item():
                self.log_alpha = self.log_alpha - self.log_alpha.detach() + self.log_min_alpha

        self.entropy_scheduler.update(entropy.mean().item())
        self.entropy_scheduler.log(L, step)

    def update_decoder(self, obs, target_obs, L, step):
        for key in self.encoders.keys():
            encoder = self.encoders[key]
            decoder = self.decoders[key]
            h = encoder(obs)

            if target_obs.dim() == 4:
                # preprocess images to be in [-0.5, 0.5] range
                target_obs = utils.preprocess_obs(target_obs)
            rec_obs = decoder(h)
            rec_loss = F.mse_loss(target_obs, rec_obs)

            # add L2 penalty on latent representation
            # see https://arxiv.org/pdf/1903.12436.pdf
            latent_loss = (0.5 * h.pow(2).sum(1)).mean()

            loss = rec_loss + self.decoder_latent_lambda * latent_loss
            self.encoder_optimizers[key].zero_grad()
            self.decoder_optimizers[key].zero_grad()
            loss.backward()

            self.encoder_optimizers[key].step()
            self.decoder_optimizers[key].step()
            L.log('train_ae/ae_loss', loss, step)

            decoder.log(L, step, log_freq=LOG_FREQ, tag=key)

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

        if len(self.decoders) > 0 and step % self.decoder_update_freq == 0:
            self.update_decoder(obs, obs, L, step)

    def save(self, model_dir, step):
        if step is not None:
            step_suffix = f'_{step}'
        else:
            step_suffix = ''

        torch.save(
            self.actor.state_dict(), f'{model_dir}/actor{step_suffix}.pt'
        )
        torch.save(
            self.critic.state_dict(), f'{model_dir}/critic{step_suffix}.pt'
        )
        torch.save(
            self.actor_optimizer.state_dict(), f'{model_dir}/actor_optimizer{step_suffix}.pt'
        )
        torch.save(
            self.critic_optimizer.state_dict(), f'{model_dir}/critic_optimizer{step_suffix}.pt'
        )
        torch.save(
            self.log_alpha_optimizer.state_dict(), f'{model_dir}/log_alpha_optimizer{step_suffix}.pt'
        )
        for key in self.encoder_optimizers.keys():
            torch.save(self.encoder_optimizers[key].state_dict(), f'{model_dir}/encoder_optimizer_{key}{step_suffix}.pt')

        for key in self.decoders.keys():
            torch.save(self.decoders[key].state_dict(), f'{model_dir}/decoder_{key}{step_suffix}.pt')
            torch.save(self.decoder_optimizers[key].state_dict(), f'{model_dir}/decoder_optimizer_{key}{step_suffix}.pt')

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        if self.decoder is not None:
            self.decoder.load_state_dict(
                torch.load('%s/decoder_%s.pt' % (model_dir, step))
            )
