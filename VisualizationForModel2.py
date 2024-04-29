import matplotlib.pyplot as plt
import numpy as np

# Load data from files
data2_500 = np.loadtxt('./OutcomeData/model2_output_500.txt')
data2_1000 = np.loadtxt('./OutcomeData/model2_output_1000.txt')
data2_2000 = np.loadtxt('./OutcomeData/model2_output_2000.txt')

# Create the plot
plt.figure(figsize=(10, 6))

# Plot each dataset
plt.plot(data2_500[:, 0], data2_500[:, 1], label='N=500', marker='o', linestyle='-')
plt.plot(data2_1000[:, 0], data2_1000[:, 1], label='N=1000', marker='s', linestyle='--')
plt.plot(data2_2000[:, 0], data2_2000[:, 1], label='N=2000', marker='^', linestyle=':')

# Label the plot
plt.title('Kuramoto Model Phase Coherence (Model 2)')
plt.xlabel('Coupling Strength')
plt.ylabel('Mean Phase Coherence')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
