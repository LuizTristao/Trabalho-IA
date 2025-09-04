import numpy as np
import gymnasium as gym

#Hiperparâmetros
alpha = 0.9
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.9995
min_epsilon = 0.01
num_episodes = 10_000
max_steps = 100
seed = 42

#Ambiente de treino (sem render)
env = gym.make('Taxi-v3')
env.reset(seed=seed)

n_states = env.observation_space.n
n_actions = env.action_space.n

q_table = np.zeros((n_states, n_actions), dtype=np.float32)

def choose_action(state, eps):
    #ε-greedy
    if np.random.random() < eps:
        return env.action_space.sample()
    else:
        return int(np.argmax(q_table[state, :]))

#Treinamento
for episode in range(num_episodes):
    state, info = env.reset()
    done = False

    for step in range(max_steps):
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        #Q-learning update
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])
        target = reward if done else (reward + gamma * next_max)
        q_table[state, action] = (1 - alpha) * old_value + alpha * target

        state = next_state
        if done:
            break

    #decaimento do epsilon por episódio
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

env.close()

#Avaliação com renderização
test_env = gym.make('Taxi-v3', render_mode='human')
for ep in range(1, 6):
    state, info = test_env.reset()
    done = False
    total_reward = 0
    steps = 0

    print(f'\nEpisódio {ep}')
    while not done and steps < max_steps:
        #política gulosa (sem exploração)
        action = int(np.argmax(q_table[state, :]))
        next_state, reward, terminated, truncated, info = test_env.step(action)
        test_env.render()  #imprime o grid no terminal
        total_reward += reward
        state = next_state
        steps += 1
        done = terminated or truncated

    print(f'Finalizado com recompensa total = {total_reward} em {steps} passos')

test_env.close()
