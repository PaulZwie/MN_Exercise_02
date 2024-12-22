import numpy as np
import matplotlib.pyplot as plt
import Data.plotSpikeRaster as psr

# Example firing matrix
firingMatrix = np.array([
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
], dtype=bool)

# Example force signal
time = np.arange(firingMatrix.shape[1])
force_signal = np.sin(time / 10) * 10  # Example force signal

# Plotting
fig, ax1 = plt.subplots()

# Plot spike trains
psr(firingMatrix, plot_type='vertline', fig_handle=ax1, vert_spike_height=0.9)

# Plot force signal on secondary y-axis
ax2 = ax1.twinx()
ax2.plot(time, force_signal, 'r-', label='Force Signal')
ax2.set_ylabel('Force (N)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Set labels and title
ax1.set_xlabel('Time (samples)')
ax1.set_ylabel('Motor Units')
ax1.set_title('Spike Trains and Force Signal')

# Show plot
plt.show()