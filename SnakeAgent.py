import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from SnakeGame import SnakeGame
from matplotlib import pyplot as plt

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = Linear_QNet(state_size, 128, action_size)  # Adjust hidden_size if needed
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32)
        act_values = self.model(state)
        return np.argmax(act_values.detach().numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        states = np.vstack([each[0] for each in minibatch])
        actions = np.array([each[1] for each in minibatch])
        rewards = np.array([each[2] for each in minibatch], dtype=np.float32)
        next_states = np.vstack([each[3] for each in minibatch])
        dones = np.array([each[4] for each in minibatch], dtype=np.float32)

        states = torch.tensor(states, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long).view(-1, 1)
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)

        pred = self.model(states)
        next_pred = self.model(next_states).detach()
        target = pred.clone()

        batch_index = np.arange(batch_size, dtype=np.int32)
        target[batch_index, actions.squeeze().numpy()] = rewards + self.gamma * torch.max(next_pred, dim=1)[0] * (1 - dones)

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train():
    game = SnakeGame()
    state_size = 11  # Based on the state array length
    action_size = 4  # Left, Right, Up, Down
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    episodes = 200  # Number of games to play

    scores = []  # To store scores for each episode
    losses = []  # To store loss values
    best_score = -np.inf  # Initialize best score to very low value

    for e in range(episodes):
        game.reset()
        state = game.get_state()
        state = np.reshape(state, [1, state_size])
        score = 0

        while not game.game_over:
            action = agent.act(state)
            reward, done = game.play_step(action)
            score = reward
            next_state = game.get_state()
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                scores.append(score)
                print(f"Episode: {e+1}/{episodes}, Score: {score}")
                if score > best_score:
                    best_score = score
                    print(f"Saving best model with score: {score}")
                    agent.model.save("./best_model.pth")  # Save the best model
                break

            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                losses.append(loss)
                agent.update_epsilon()

    return scores, losses



def play_game_with_agent(agent, game):
    # Load the best model
    agent.model.load_state_dict(torch.load("./model/best_model.pth"))
    agent.epsilon = 0  # Ensure agent acts purely based on learned policy

    game.reset()
    state = game.get_state()
    state = np.reshape(state, [1, -1])

    plt.figure(figsize=(12, 8))  # Adjust size as needed

    while not game.game_over:
        with torch.no_grad():  # Ensuring no gradient calculations
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action = agent.act(state_tensor)

        _, done = game.play_step(action)
        state = game.get_state()
        state = np.reshape(state, [1, -1])

        game.plot_state()  # Update the plot for each step

        if done:
            break  # Exit the loop if the game is over

    plt.show()  # Show the final state



if __name__ == "__main__":
    game = SnakeGame()
    agent = DQNAgent(state_size=11, action_size=4) 

    # Training the agent
    scores, losses = train()

    # Plotting scores
    plt.figure(figsize=(10, 5))
    plt.plot(scores, label='Scores')
    plt.title('Scores over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

    # Plotting losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss')
    plt.title('Loss over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Visualization after training
    play_game_with_agent(agent, game)
