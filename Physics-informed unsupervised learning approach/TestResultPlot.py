import matplotlib.pyplot as plt
import numpy as np

# Example 1D array (vector)
vector = np.load("Test_sample_N-R_iterations.npy")

# Plot the vector
plt.plot(vector, marker='o', linestyle='-')

# Labels and title
plt.xlabel("Test sample ID")
plt.ylabel("Iterations N-R convergence")

# Show grid
plt.grid(True)

# Show the plot
plt.savefig("Iterations_N-R_convergence_testset.png", dpi=300)
plt.show()