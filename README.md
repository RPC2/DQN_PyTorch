# PyTorch Implementation of DQN

### Result

![img](https://github.com/RPC2/DQN_PyTorch/blob/master/score.png)

OpenAI [defines](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) CartPole as solved "when the average reward is greater than or equal to 195.0 over 100 consecutive trials."

### Hyperparameters Used

gamma = 0.99

train_freq = 1 (step)

start_learning = 10

memory_size = 1000000

batch_size = 32

reset_every = 10 (terminated episode)

epsilon = 1

epsilon_minimum = 0.05

epsilon_decay_rate = 0.9999

learning_rate = 0.001

### Tips for Debugging

1. If you're training your agent with CartPole, set the total episode at a minimum amount (say, 1), then print out all the variables to check if your variables are storing the correct information. If your environment is taking image input (e.g., you're training on Breakout), change the environment to CartPole first (since it's easier to interpret CartPole's observations and outputs).
2. A nice reference to interpret state observation and your agent's actions: OpenAI's [documentation](https://github.com/openai/gym/tree/master/gym/envs). 
3. After checking each variable, see if your program flow is correct. This can be done by inspecting your variables or compare with other working code.
4. If you're sure about variable content and program flow working correctly - congratulations! Adjust your hyperparameters and you'll get a rough sense on how each hyperparameter invokes changes in performance.
5. One key factor of me judging whether my agent is learning effectively is the Q value of each state. At the beginning the Q values might be quite different (since it depends on network initialization), with one action strictly dominate the other. As time goes on, the difference starts to shrink and the performance takes off when the Q values of actions get close to each other. Similarly, try to scrutinize the actual values of your key variables, and you might get insights from it :).

### Original Paper

[Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)