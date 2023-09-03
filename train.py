import cv2
import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
import json
import warnings

from gym.wrappers import TimeLimit, GrayScaleObservation
from omegaconf import OmegaConf

from env.cw_envs import CwTargetEnv

try:
    import dmc2gym
except ImportError:
    warnings.warn('dmc2gym is not installed.')

import copy

import wandb
from gym.spaces import Box

import utils
from logger import Logger
from video import VideoRecorder

from sac_ae import SacAeAgent, SacAeAgentDiscrete
import env.shapes2d


class FailOnTimelimit(gym.Wrapper):
    def __init__(self, env):
        super(FailOnTimelimit, self).__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done:
            info['is_success'] = not info.get("TimeLimit.truncated", False)

        return observation, reward, done, info


def str2bool(s):
    """helper function used in order to support boolean command line arguments"""
    if s.lower() in ("true", "t", "1"):
        return True
    elif s.lower() in ("false", "f", "0"):
        return False
    else:
        return s


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--domain_type', choices=['gym', 'dmc', 'cw'], default='dmc')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=1000000, type=int)
    # train
    parser.add_argument('--agent', default='sac_ae', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=10000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    parser.add_argument('--actor_encoder', default=False, choices=[False, True], metavar='False|True', type=str2bool)
    # encoder/decoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--decoder_type', default='pixel', type=str)
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_latent_lambda', default=1e-6, type=float)
    parser.add_argument('--decoder_weight_lambda', default=1e-7, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    parser.add_argument('--auto_alpha', default=True, choices=[False, True], metavar='False|True', type=str2bool)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--run_name', type=str, required=True)

    args = parser.parse_args()
    return args


def evaluate(env, agent, video, num_episodes, L, step):
    for i in range(num_episodes):
        obs = env.reset()
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        info = None
        video.record(env)
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            video.record(env)
            episode_reward += reward

        video.save('%d.mp4' % step)
        L.log('eval/episode_reward', episode_reward, step)
        L.log('eval/success_rate', int(info['is_success']), step)
    L.dump(step, is_eval=True)


def make_agent(obs_shape, action_space, args, device):
    if args.agent == 'sac_ae':
        if isinstance(action_space, gym.spaces.Box):
            return SacAeAgent(
                obs_shape=obs_shape,
                action_shape=action_space.shape,
                device=device,
                hidden_dim=args.hidden_dim,
                discount=args.discount,
                init_temperature=args.init_temperature,
                alpha_lr=args.alpha_lr,
                alpha_beta=args.alpha_beta,
                actor_lr=args.actor_lr,
                actor_beta=args.actor_beta,
                actor_log_std_min=args.actor_log_std_min,
                actor_log_std_max=args.actor_log_std_max,
                actor_update_freq=args.actor_update_freq,
                critic_lr=args.critic_lr,
                critic_beta=args.critic_beta,
                critic_tau=args.critic_tau,
                critic_target_update_freq=args.critic_target_update_freq,
                encoder_type=args.encoder_type,
                encoder_feature_dim=args.encoder_feature_dim,
                encoder_lr=args.encoder_lr,
                encoder_tau=args.encoder_tau,
                decoder_type=args.decoder_type,
                decoder_lr=args.decoder_lr,
                decoder_update_freq=args.decoder_update_freq,
                decoder_latent_lambda=args.decoder_latent_lambda,
                decoder_weight_lambda=args.decoder_weight_lambda,
                num_layers=args.num_layers,
                num_filters=args.num_filters
            )
        elif isinstance(action_space, gym.spaces.Discrete):
            return SacAeAgentDiscrete(
                obs_shape=obs_shape,
                action_dim=action_space.n,
                device=device,
                hidden_dim=args.hidden_dim,
                discount=args.discount,
                init_temperature=args.init_temperature,
                alpha_lr=args.alpha_lr,
                alpha_beta=args.alpha_beta,
                auto_alpha=args.auto_alpha,
                actor_lr=args.actor_lr,
                actor_beta=args.actor_beta,
                actor_update_freq=args.actor_update_freq,
                actor_encoder=args.actor_encoder,
                critic_lr=args.critic_lr,
                critic_beta=args.critic_beta,
                critic_tau=args.critic_tau,
                critic_target_update_freq=args.critic_target_update_freq,
                encoder_type=args.encoder_type,
                encoder_feature_dim=args.encoder_feature_dim,
                encoder_lr=args.encoder_lr,
                encoder_tau=args.encoder_tau,
                decoder_type=args.decoder_type,
                decoder_lr=args.decoder_lr,
                decoder_update_freq=args.decoder_update_freq,
                decoder_latent_lambda=args.decoder_latent_lambda,
                decoder_weight_lambda=args.decoder_weight_lambda,
                num_layers=args.num_layers,
                num_filters=args.num_filters
            )
        else:
            raise ValueError('Unexpected action space type:', type(action_space))
    else:
        assert 'agent is not supported: %s' % args.agent


class LunarLanderImgWrapper(gym.Wrapper):
    def __init__(self, env, shape=(100, 100)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = Box(low=0, high=255, shape=(3, *self.shape), dtype=np.uint8)
        self._max_episode_steps = env._max_episode_steps

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        return self._get_obs(), reward, done, info

    def reset(self, **kwargs):
        _ = self.env.reset(**kwargs)
        return self._get_obs()

    def _get_obs(self):
        image = self.env.render(mode='rgb_array')
        observation = cv2.resize(image, self.shape, interpolation=cv2.INTER_AREA)
        return np.moveaxis(observation, 2, 0)


def make_env(args, is_eval=False):
    seed = args.seed
    if is_eval:
        seed += 1

    channels_first = True
    if args.domain_name == 'lunarlander':
        env = gym.make('LunarLanderContinuous-v2')
        env = LunarLanderImgWrapper(env)
        env.seed(seed)
    elif args.domain_type == 'dmc':
        env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            seed=seed,
            visualize_reward=False,
            from_pixels=(args.encoder_type == 'pixel'),
            height=args.image_size,
            width=args.image_size,
            frame_skip=args.action_repeat
        )
        env.seed(seed)
    elif args.domain_type == 'cw':
        config = OmegaConf.load(f'env/config/{args.domain_name}.yaml')
        env = CwTargetEnv(config, seed)
        env = TimeLimit(env, env.unwrapped._max_episode_length)
        channels_first = False
    elif args.domain_type == 'gym':
        env = FailOnTimelimit(gym.make(args.domain_name))
        env.seed(seed)
        channels_first = False
    else:
        raise ValueError(f'Unknown domain: type={args.domain_type} name={args.domain_name}')

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack, channels_first=channels_first)

    return env


def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)

    run = wandb.init(
        project=args.project,
        sync_tensorboard=False,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        name=args.run_name
    )

    env = make_env(args, is_eval=False)
    eval_env = make_env(args, is_eval=True)

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # the dmc2gym wrapper standardizes actions
    if isinstance(env.action_space, gym.spaces.Box):
        assert env.action_space.low.min() >= -1
        assert env.action_space.high.max() <= 1
        action_shape = env.action_space.shape
    elif isinstance(env.action_space, gym.spaces.Discrete):
        action_shape = (1,)
    else:
        raise ValueError('Unexpected action space type:', type(env.action_space))

    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device
    )

    agent = make_agent(
        obs_shape=env.observation_space.shape,
        action_space=env.action_space,
        args=args,
        device=device
    )

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    eval_step = 0
    eval_episode = 0
    info = {'is_success': 0}
    for step in range(args.num_train_steps):
        if done:
            L.log('train/episode_reward', episode_reward, step)
            L.log('train/success_rate', int(info['is_success']), step)

            # evaluate agent periodically
            if step >= eval_step:
                eval_step += args.eval_freq
                eval_episode += args.num_eval_episodes
                L.log('eval/episode', episode, step)
                evaluate(eval_env, agent, video, args.num_eval_episodes, L, step)
                if args.save_model:
                    agent.save(model_dir, step)
                if args.save_buffer:
                    replay_buffer.save(buffer_dir)
            elif step > 0:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            info = None

            L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = args.init_steps if step == args.init_steps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, info = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward

        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1

    wandb.finish()


if __name__ == '__main__':
    main()
