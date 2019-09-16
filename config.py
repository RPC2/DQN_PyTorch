class AgentConfig:
    frames = 4
    # Learning
    gamma = 0.95
    to_train = True
    train_freq = 1
    start_learning = 21
    epsilon = 1
    memory_size = 1000000
    batch_size = 20
    plot_every = 1000

    epsilon_minimum = 0.01
    epsilon_decay_rate = 0.995

    max_step = 40000000       # 40M steps max
    max_episode_length = 18000  # equivalent of 5 minutes of game play at 60 frames per second

    # Algorithm selection
    train_cartpole = True
    per = False

    double_q_learning = False
    duelling_dqn = False

    gif = False
    gif_every = 9999999


class EnvConfig:
    episode_size = 1
    step_size = 10
    env_name = 'CartPole-v0'
    screen_width = 84
    screen_height = 84
    max_reward = 1
    min_reward = -1
    save_every = 10000
