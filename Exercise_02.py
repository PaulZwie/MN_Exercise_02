import os

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def import_data(file_path) -> dict:
    return sp.io.loadmat(file_path)


def MUPulses_to_firing_matrix(MUPulses) -> np.ndarray:
    # find max entry in MUPulses
    maximum = 0
    for array in MUPulses[0]:
        max_array_entry = np.max(array[0])
        if max_array_entry > maximum:
            maximum = max_array_entry

    maximum += 1  # because location counted from 00

    firing_matrix = np.zeros(shape=(MUPulses.shape[1], maximum))

    for array_count, array in enumerate(MUPulses[0]):
        for entry in array[0]:
            firing_matrix[array_count][entry] = 1

    return firing_matrix


def plot_spike_trains_and_force(firing_matrix, force_signal):
    """
    Plots the spike trains of all motor units along with the force signal.

    Args:
        firing_matrix: A 2D NumPy array where rows represent motor units
                      and columns represent time points.
                      A value of 1 indicates a spike, 0 otherwise.
        force_signal: A 1D NumPy array representing the force signal over time.
    """

    num_motor_units, num_time_points = firing_matrix.shape

    # Create a figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot spike trains
    for i in range(num_motor_units):
        spike_times = np.where(firing_matrix[i, :] == 1)[0]
        ax1.scatter(spike_times, np.ones(len(spike_times)) * i, marker='|', color='black')

    # Set y-axis for motor units
    ax1.set_ylabel('Motor Unit Number')
    ax1.set_yticks(range(num_motor_units))
    ax1.set_yticklabels(range(1, num_motor_units + 1))

    # Create a second y-axis for force
    ax2 = ax1.twinx()
    ax2.plot(force_signal, color='red')
    ax2.set_ylabel('Force (N)')

    # Set x-axis label
    ax1.set_xlabel('Time (s)')

    # Set title
    plt.title('Spike Trains and Force Signal')

    # Show the plot
    plt.show()


def main():
    cwd = os.getcwd()
    file_path = os.path.join(cwd, "Data", "iEMG_contraction.mat")

    data = import_data(file_path)
    EMGSig = data['EMGSig']
    MUPulses = data['MUPulses']
    force_signal = data['force_signal']
    fsamp = data['fsamp']
    signal_length_samples = data['signal_length_samples']
    firing_matrix = MUPulses_to_firing_matrix(MUPulses)




    #Plots
    plot_spike_trains_and_force(firing_matrix, force_signal)


if __name__ == "__main__":
    main()
