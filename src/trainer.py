import torch.optim as optim
import torch.nn.functional as F
from itertools import count
from envManager import GymEnvManager
from strategy import EpsilonGreedyStrategy
from agent import Agent
from configs import *
from replayMemory import *
from network import DQN, QValues


class Trainer:
    def __init__(self, environment_manager: GymEnvManager):
        self.em = environment_manager
        strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

        self.agent = Agent(strategy, self.em.num_actions_available(), device)
        self.memory = ReplayMemory(memory_size)

        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)

        if load_model:
            self.load_policy_from_path('policynet')

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=lr)
        self.episode_durations = []

    def load_policy_from_path(self, path):
        self.policy_net.load_state_dict(torch.load(path))

    def save_policy_to_file(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def train(self):
        for episode in range(num_episodes):
            self.em.reset()
            state = self.em.get_state()

            for timestep in count():
                if self.em.should_render:
                    self.em.render()
                action = self.agent.select_action(state, self.policy_net)
                reward = self.em.take_action(action)
                next_state = self.em.get_state()
                self.memory.push(Experience(state, action, next_state, reward))
                state = next_state

                if self.memory.can_provide_sample(batch_size):
                    experiences = self.memory.sample(batch_size)
                    states, actions, rewards, next_states = extract_tensors(experiences)

                    current_q_values = QValues.get_current(self.policy_net, states, actions)
                    next_q_values = QValues.get_next(self.target_net, next_states)
                    target_q_values = (next_q_values * gamma) + rewards

                    loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if self.em.done:
                    self.episode_durations.append(timestep)
                    # print average time of the last 100 episodes
                    print(sum(self.episode_durations[-100:]) / min(len(self.episode_durations), 100))
                    break

            if episode % target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        # print average time of the last 100 episodes
        print(sum(self.episode_durations[-100:]) / 100)
        self.save_policy_to_file(self.em.env_name)
        self.em.close()