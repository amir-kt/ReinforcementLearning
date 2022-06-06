import gym
import torch
from configs import device


class GymEnvManager:
    def __init__(self, env, should_render, env_name):
        self.device = device
        self.gym_env = env
        self.gym_env.reset()
        self.current_state = None
        self.done = False
        self.should_render = should_render
        self.env_name = env_name

    def reset(self):
        self.current_state = self.gym_env.reset()

    def close(self):
        self.gym_env.close()

    def render(self):
        return self.gym_env.render()

    def num_actions_available(self):
        return self.gym_env.action_space.n

    def take_action(self, action):
        state, reward, self.done, _ = self.gym_env.step(action.item())
        self.current_state = state
        return torch.tensor([reward], device=self.device)

    def get_state(self):
        if self.done:
            return torch.zeros_like(
                torch.tensor(self.current_state), device=self.device
            ).float()
        else:
            return torch.tensor(self.current_state, device=self.device).float()

    def num_state_features(self):
        return self.gym_env.observation_space.shape[0]


class CartPoleManager(GymEnvManager):
    def __init__(self):
        env_name = 'CartPole-v1'
        self.cartpole_env = gym.make(env_name)
        super().__init__(self.cartpole_env, True, env_name)


class PongManager(GymEnvManager):
    def __init__(self):
        env_name = 'ALE/Pong-v5'
        self.pong_env = gym.make(env_name, render_mode='human')
        super().__init__(self.pong_env, False, env_name)
