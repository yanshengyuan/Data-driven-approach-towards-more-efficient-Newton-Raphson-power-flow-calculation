import numpy as np
from stable_baselines3 import PPO
import wandb
from CallBack import WandbCallback
from GridEnv import GridEnv
from stable_baselines3.common.monitor import Monitor
import os

# train an RL agent on the environment GridEnv

# Create log dir
lr = 5e-4
total_timesteps = 2e6
log_dir = f"saved_models/log3_PPO_lr_{lr}_timesteps_{total_timesteps}/"
os.makedirs(log_dir, exist_ok=True)

wandb.init(project="grid-env-training-finetuning")

env = GridEnv()
env = Monitor(env, log_dir)


model = PPO("MlpPolicy", env, verbose=1, learning_rate = lr)
model.learn(total_timesteps=total_timesteps, callback = WandbCallback(check_freq=1000, log_dir=log_dir))
model.save(f"saved_models/PPO_{lr=}_{total_timesteps=}")

wandb.finish()

