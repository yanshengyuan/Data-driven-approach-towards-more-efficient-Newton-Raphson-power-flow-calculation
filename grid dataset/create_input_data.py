import numpy as np


def generate_input_data_simple_dataset(
    n_samples, INIT_VM_2_PU_MIN=0.85, INIT_VM_2_PU_MAX=1.15, INIT_THETA_2_DEGREE_MAX=-45, INIT_THETA_2_DEGREE_MIN=45
):

    VM_PU_SLACK = 1.0  # Slack bus voltage magnitude
    P_MW = 0.9  # Active power load at bus 2
    Q_MVAR = 0.6  # Reactive power load at bus 2

    R_PU = 0.01  # Resistance of the line
    X_PU = 0.1  # Reactance of the line

    VM_2_PU = np.random.uniform(INIT_VM_2_PU_MIN, INIT_VM_2_PU_MAX, (n_samples, 1))
    THETA_2_RAD = np.random.uniform(INIT_THETA_2_DEGREE_MIN, INIT_THETA_2_DEGREE_MAX, (n_samples, 1))

    input_data = np.hstack(
        (
            np.zeros((n_samples, 1)),  # P1,
            np.ones((n_samples, 1)) * P_MW,  # P2,
            np.zeros((n_samples, 1)),  # Q1,
            np.ones((n_samples, 1)) * Q_MVAR,  # Q2,
            np.ones((n_samples, 4)),  #
            np.ones((n_samples, 4)),
            np.ones((n_samples, 1)) * VM_PU_SLACK,  # VM1,
            VM_2_PU,
            np.zeros((n_samples, 1)),  # THETA_1_RAD, #Theta 1
            THETA_2_RAD,
            np.ones((n_samples, 1)) * R_PU,  # R,
            np.ones((n_samples, 1)) * X_PU,  # X,
        ),
    )

    return input_data


if __name__ == "__main__":
    input_data = generate_input_data_simple_dataset(10)
    print(input_data)
