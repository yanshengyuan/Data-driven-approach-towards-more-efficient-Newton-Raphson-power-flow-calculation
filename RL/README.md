# Power Grid RL Agent

## Overview

This repository contains an RL-based approach to power grid simulation and control using a custom environment. The RL agent is trained on an environment that simulates a power grid given an n-bus case. In this project, the 2-bus test case is implemented in `SimpleTwoBus.py`, utilizing `pandapower` for power system simulations.

## Project Structure

```markdown
- `SimpleTwoBus.py` - Defines the 2-bus test case using `pandapower`.
- `GridEnv.py` - Custom environment for RL training.
- `train_RL_agent.py` - Sets up and trains the RL agent using Stable-Baselines3 PPO, specify the learning rate and number of timesteps inside this file please.
- `evaluate_model.py` - Evaluates the trained model with specified parameters,specify the learning rate and number of timesteps inside this file please.
- `plot_training_results.py` - Plots learning curves from training data retrieved via `wandb`.
- `plot_results.py` - Plots evaluation results, specify the learning rate, number of timesteps, and number of evaluations inside this file please.
- `saved_models/` - Stores trained models (e.g., `best_model.zip`).
- `saved_files/` - Stores evaluation results.
- `plotted_results/` - Stores plots of training and evaluation results.
```

## Installation

Ensure you have Python installed, then install the required packages:

```bash
pip install gymnasium pandas stable-baselines3 wandb numpy pandapower seaborn
```

## Training the RL Agent

To train the RL agent, run:

```bash
python train_RL_agent.py
```

Training progress is logged using `wandb`.

## Evaluating the Trained Model

To evaluate the trained model, run:

```bash
python evaluate_model.py
```

Results are saved in `saved_files/`.

## Plotting Training and Evaluation Results

To plot the learning curves:

```bash
python plot_training_results.py
```

To plot the evaluation results:

```bash
python plot_results.py
```


## License

This project is licensed under the MIT License.

