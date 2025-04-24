import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

dir = "saved_training_results/learning_curves/"

# lrs = ["1e-4", "5e-4"]
filenamelr1 = ["PPO_lr1e-4_timesteps2e6_1", "PPO_lr1e-4_timesteps2e6_2", "PPO_lr1e-4_timesteps2e6_3"]
filenamelr5 = ["PPO_lr5e-4_timesteps2e6_1", "PPO_lr5e-4_timesteps2e6_2", "PPO_lr5e-4_timesteps2e6_3"]
timesteps = "2e6"



learning_curve_lr1 = []
for fname in filenamelr1:

    learning_curve = pd.read_csv(f"{dir}{fname}.csv", header=None, skiprows=1)
    learning_curve_lr1.append(learning_curve.iloc[:,1])

mean_learning_curve_lr1 = np.mean(np.asarray(learning_curve_lr1), axis = 0)
std_learning_curve_lr1 = np.std(np.asarray(learning_curve_lr1), axis = 0)



learning_curve_lr5 = []
for fname in filenamelr5:

    learning_curve = pd.read_csv(f"{dir}{fname}.csv", header=None, skiprows= 1)
    learning_curve_lr5.append(learning_curve.iloc[:,1])

mean_learning_curve_lr5 = np.mean(np.asarray(learning_curve_lr5), axis = 0)
std_learning_curve_lr5 = np.std(np.asarray(learning_curve_lr5), axis = 0)




sns.set(style="darkgrid")
plt.figure(figsize=(8,6))
plt.plot(mean_learning_curve_lr1, label = f"lr = 1e-4", linewidth=1.5, color = "royalblue")
plt.fill_between(range(len(mean_learning_curve_lr1)), 
                 mean_learning_curve_lr1 - std_learning_curve_lr1, 
                 mean_learning_curve_lr1 + std_learning_curve_lr1, 
                 alpha=0.3, color = 'royalblue')
plt.plot(mean_learning_curve_lr5, label = f"lr = 5e-4", linewidth=1.5, color = "forestgreen")
plt.fill_between(range(len(mean_learning_curve_lr5)), 
                 mean_learning_curve_lr5 - std_learning_curve_lr5, 
                 mean_learning_curve_lr5 + std_learning_curve_lr5, 
                 alpha=0.3, color = 'forestgreen')
plt.xlabel("Episode", fontsize = 14)
plt.ylabel("Episode reward", fontsize = 14)
plt.legend(fontsize = 12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(f"{dir}PPO_learning_rate_values.pdf")
plt.show()