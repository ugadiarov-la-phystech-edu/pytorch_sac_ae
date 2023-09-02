from gym.envs.registration import register

register(
    'Navigation5x5-v0',
    entry_point='env.shapes2d.shapes2d:Shapes2d',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 1,
        'n_static_boxes': 0,
        'static_goals': True,
        'width': 5,
        'render_scale': 10,
        'channel_wise': False,
        'channels_first': False,
        'ternary_interactions': False,
        'embodied_agent': False,
        'do_reward_push_only': False,
    },
)

register(
    'Navigation10x10-v0',
    entry_point='env.shapes2d.shapes2d:Shapes2d',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 8,
        'n_goals': 1,
        'n_static_boxes': 0,
        'static_goals': True,
        'width': 10,
        'render_scale': 10,
        'channel_wise': False,
        'channels_first': False,
        'ternary_interactions': False,
        'embodied_agent': False,
        'do_reward_push_only': False,
    },
)

register(
    'Pushing7x7-v0',
    entry_point='env.shapes2d.shapes2d:Shapes2d',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 1,
        'n_static_boxes': 0,
        'width': 7,
        'render_scale': 10,
        'channel_wise': False,
        'channels_first': False,
        'ternary_interactions': True,
        'embodied_agent': True,
        'do_reward_push_only': False,
    },
)

register(
    'PushingNoAgent5x5-v0',
    entry_point='env.shapes2d.shapes2d:Shapes2d',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 1,
        'n_static_boxes': 0,
        'width': 5,
        'render_scale': 10,
        'channel_wise': False,
        'ternary_interactions': True,
        'channels_first': False,
        'embodied_agent': False,
        'do_reward_push_only': True,
    },
)
