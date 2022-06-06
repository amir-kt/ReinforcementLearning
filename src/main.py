from trainer import Trainer
from envManager import *


def train_cart_pole():
    env_manager = CartPoleManager()
    env_manager.render()
    trainer = Trainer(env_manager, layers=[('linear', 4, 24), ('linear', 24, 32), ('linear', 32, 2)])
    trainer.train()


def train_pong():
    env_manager = PongManager()
    trainer = Trainer(env_manager, layers=[])
    trainer.train()


if __name__ == '__main__':
    # train_pong()
    train_cart_pole()
