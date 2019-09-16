class AgentConfig:
    frames = 4
    # Learning
    gamma = 0.99
    to_train = True
    train_freq = 1
    start_learning = 100
    epsilon = 1
    memory_size = 50000
    batch_size = 32
    scale = 10000
    learning_rate = 0.00025
    learning_rate_step = 10 * scale
    learning_rate_minimum = 0.00001
    learning_rate_decay = 0.96
    learning_rate_decay_step = 5 * scale
    reset_step = 10000
    plot_every = 1000

    epsilon_minimum = 0.05
    epsilon_decay = 0.999

    max_step = 40000000       # 40M steps max
    max_episode_length = 18000  # equivalent of 5 minutes of game play at 60 frames per second

    target_update_rate = 0.005

    # Algorithm selection
    soft_update = False
    train_cartpole = True
    per = False

    double_q_learning = True
    duelling_dqn = True

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
    gpu_available = True
