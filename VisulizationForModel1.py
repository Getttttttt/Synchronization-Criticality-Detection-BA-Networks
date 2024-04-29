import matplotlib.pyplot as plt
import numpy as np

# Load data from files
data_500 = np.loadtxt('./OutcomeData/model1_output_500.txt')
data_1000 = np.loadtxt('./OutcomeData/model1_output_1000.txt')
data_2000 = np.loadtxt('./OutcomeData/model1_output_2000.txt')

# Create the main plot with an inset
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each dataset on the main axes
ax.plot(data_500[:, 0], data_500[:, 1], label='N=500', marker='o', linestyle='-')
ax.plot(data_1000[:, 0], data_1000[:, 1], label='N=1000', marker='s', linestyle='--')
ax.plot(data_2000[:, 0], data_2000[:, 1], label='N=2000', marker='^', linestyle=':')

# Label the main plot
ax.set_title('Kuramoto Model Phase Coherence')
ax.set_xlabel('Coupling Strength')
ax.set_ylabel('Mean Phase Coherence')
ax.legend()
ax.grid(True)

# Create an inset axis
ax_inset = ax.inset_axes([0.5, 0.5, 0.45, 0.45])

# Plot each dataset in the inset focusing on 0.0 to 0.2 range
mask_500 = data_500[:, 0] <= 0.2
mask_1000 = data_1000[:, 0] <= 0.2
mask_2000 = data_2000[:, 0] <= 0.2

ax_inset.plot(data_500[mask_500, 0], data_500[mask_500, 1], marker='o', linestyle='-')
ax_inset.plot(data_1000[mask_1000, 0], data_1000[mask_1000, 1], marker='s', linestyle='--')
ax_inset.plot(data_2000[mask_2000, 0], data_2000[mask_2000, 1], marker='^', linestyle=':')

# Set inset title and grid
ax_inset.set_title('Zoom: Coupling Strength 0.0-0.2')
ax_inset.grid(True)

# Show the plot with the inset
plt.show()
