import numpy as np
from GridEnv import GridEnv
from stable_baselines3 import PPO
import pandas as pd
import matplotlib.pyplot as plt

def plot_a_trace(evaluation_results, index, timesteps, save_dir):
    # Convert results to a numpy array for easier plotting
    V_start_values = np.array([result["V_start"] for result in evaluation_results])
    theta_start_values = np.array([result["theta_start"] for result in evaluation_results])
    timesteps_values = np.array([result["timesteps"] for result in evaluation_results])



    # Plotting
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(theta_start_values, V_start_values, c=timesteps_values, cmap='viridis', alpha=0.1)

    trace = evaluation_results[index]["trace"]


    V_trace = [step["state"][0] for step in trace]

    theta_trace = [step["state"][1] for step in trace]

    k_trace = [step["state"][2] for step in trace]
    residual_trace = [step["reward"] for step in trace]

    plt.plot(theta_trace, V_trace, color="black")
    plt.scatter(theta_trace, V_trace, color="black")

    plt.annotate(f'res={residual_trace[0]:.2f},\n k={k_trace[0]}', (theta_trace[0], V_trace[0]),
                        textcoords="offset points", xytext=(0,5), ha='center', fontsize=13)
    plt.annotate(f'res={residual_trace[-1]:.2f},\n k={k_trace[-1]}', (theta_trace[-1], V_trace[-1]),
                        textcoords="offset points", xytext=(0,5), ha='center', fontsize=13)
      
    if len(list(V_trace))==10:
        plt.scatter(theta_trace[-1], V_trace[-1], color="red")


    cbar = plt.colorbar(scatter)
    cbar.set_label('Number of Timesteps', fontsize=19)
    plt.xlabel(r'$\theta$', fontsize=19)
    plt.ylabel(r'$V$', fontsize=19)

    if save:
        plt.savefig(f"{save_dir}trace_RL_agent_lr={lr}_timesteps={timesteps}_{index=}.png")

def evaluate_model(model, num_evaluations, env, max_timesteps=10):
    
    evaluation_results = []


    for i in range(num_evaluations):
        state, info = env.reset()
        V_start = state[0]
        theta_start = state[1]
        done = False
        timesteps = 0
        trace = []
        cumulative_residual = 0

        for t in range(max_timesteps):

            action, _ = model.predict(state, deterministic=True)

            next_state, reward, done, terminated, info = env.step(action)

            trace.append({
                "state": state.copy(),
                "action": action.copy(),
                "reward": reward.copy(),
                "done": done,
                "terminated": terminated,
                "next_state": next_state.copy()
            })


            state = next_state
            cumulative_residual += reward

            timesteps += 1



            if done:
                V_end = state[0]  # Assuming V_end is the first element of the state
                theta_end = state[1]  # Assuming theta_end is the second element of the state
                break
            if terminated:
                V_end = state[0]  # Assuming V_end is the first element of the state
                theta_end = state[1]  # Assuming theta_end is the second element of the state
                break

        evaluation_results.append({
                "V_start": V_start,
                "theta_start": theta_start,
                "V_end": V_end,
                "theta_end": theta_end,
                "timesteps": timesteps,
                "residual_end": reward,
                "trace": trace,
                "cumulative_residual": cumulative_residual
            })




    return evaluation_results



env = GridEnv()
num_evaluations = 100000
lr = 1e-4
total_timesteps = 2e6
model_dir = f"saved_models/log_PPO_lr_{lr}_timesteps_{total_timesteps}/"
save = True
save_dir = "saved_files/"

model = PPO.load(f"{model_dir}best_model.zip")

evaluation_results= evaluate_model(model, num_evaluations, env)


for i in range(20):
    plot_a_trace(evaluation_results, i, total_timesteps, "plotted_results/")



if save:


    df = pd.DataFrame(evaluation_results)
    df.to_csv(f"{save_dir}evaluation_results_PPO_lr={lr}_timesteps={total_timesteps}.csv", index=False)
