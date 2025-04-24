import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os
import numpy as np

class WandbCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(WandbCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Check if the episode is done
        if self.locals["dones"][0]:
            # Log the episode return (sum of rewards)
            episode_reward = self.locals["infos"][0].get("episode", {}).get("r", 0)
            episode_length = self.locals["infos"][0].get("episode", {}).get("l", 0)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            wandb.log({"episode_reward": np.mean(self.episode_rewards[:-100]), "episode_length": np.mean(self.episode_lengths[:-100])})
            # print(f"{episode_reward=}")

            if self.n_calls % self.check_freq == 0:
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, np.mean(self.episode_rewards[:-100])))

                    # New best model, you could save the agent here
                if np.mean(self.episode_rewards[:-100]) > self.best_mean_reward:
                    self.best_mean_reward = np.mean(self.episode_rewards[:-100])
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)


        # if self.n_calls % self.check_freq == 0:

        #   # Retrieve training reward
        #   x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        #   if len(x) > 0:
        #       # Mean training reward over the last 100 episodes
        #       mean_reward = np.mean(y[-100:])
        #       if self.verbose > 0:
        #         print("Num timesteps: {}".format(self.num_timesteps))
        #         print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

            #   # New best model, you could save the agent here
            #   if mean_reward > self.best_mean_reward:
            #       self.best_mean_reward = mean_reward
            #       # Example for saving best model
            #       if self.verbose > 0:
            #         print("Saving new best model to {}".format(self.save_path))
            #       self.model.save(self.save_path)

        return True