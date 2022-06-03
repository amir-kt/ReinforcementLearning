import torch
import os

batch_size = 256
gamma = 0.99
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = os.path.isfile('policy') and True
