import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_episode_residual(evaluation_results, save_dir, save, lr, timesteps):
    # Convert results to a numpy array for easier plotting
    V_start_values = np.array([evaluation_results["V_start"]])
    theta_start_values = np.array([evaluation_results["theta_start"]])
    episode_residual = np.array([evaluation_results["cumulative_residual"]])


    plt.figure(figsize=(10,8))
    scatter2 = plt.scatter(theta_start_values, V_start_values, c=episode_residual, cmap='viridis')
    cbar = plt.colorbar(scatter2)
    cbar.set_label("Cumulative episode residual", fontsize=19)
    plt.xlabel(r'$\theta_0$' , fontsize=19)
    plt.ylabel(r'$V_0$', fontsize=19)
    if save:
        plt.savefig(f"{save_dir}episode_residual_{lr=}_{timesteps=}.pdf")
    # plt.show()


def plot_timestep_values(evaluation_results, save_dir, save, lr, timesteps):

    # Convert results to a numpy array for easier plotting
    V_start_values = np.array([evaluation_results["V_start"]])
    theta_start_values = np.array([evaluation_results["theta_start"]])
    timesteps_values = np.array([evaluation_results["timesteps"]])



    # Plotting
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(theta_start_values, V_start_values, c=timesteps_values, cmap='viridis')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Number of agent iterations', fontsize=19)
    plt.xlabel(r'$\theta_0$', fontsize=19)
    plt.ylabel(r'$V_0$', fontsize = 19)

    if save:
        plt.savefig(f"{save_dir}timesteps_agent_{lr=}_{timesteps=}.pdf")
        plt.savefig(f"{save_dir}timesteps_agent_{lr=}_{timesteps=}.png")
    # plt.show()


dir = "saved_files/"
lr=1e-4
timesteps=2e6
evaluation_results = pd.read_csv(f"{dir}evaluation_results_PPO_lr={lr}_timesteps={timesteps}_test.csv")



save_dir = "plotted_results/"
save = True

plot_timestep_values(evaluation_results, save_dir, save, lr, timesteps)

plot_episode_residual(evaluation_results, save_dir, save, lr, timesteps)

