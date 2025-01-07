import os

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def import_data(file_path):
    return sp.io.loadmat(file_path)


def mu_pulses_to_firing_matrix(mu_pulses) -> np.ndarray:
    # find max entry in MUPulses
    maximum = 0
    for array in mu_pulses:
        max_array_entry = np.max(array[0])
        if max_array_entry > maximum:
            maximum = max_array_entry

    maximum += 1  # because location counted from 00

    firing_matrix = np.zeros(shape=(len(mu_pulses), maximum))

    for array_count, array in enumerate(mu_pulses):
        for entry in array[0]:
            firing_matrix[array_count][entry] = 1

    return firing_matrix


def plot_spike_trains_and_force(firing_matrix: np.ndarray, force_signal, fsamp):
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

    force_sig_len = len(force_signal)
    ax1.set_xticks(
        [0, 0.2 * force_sig_len, 0.4 * force_sig_len, 0.6 * force_sig_len, 0.8 * force_sig_len, force_sig_len - 1],
        [0, int(((0.2 * force_sig_len) / fsamp).item()), int(((0.4 * force_sig_len) / fsamp).item()),
         int(((0.6 * force_sig_len) / fsamp).item()),
         int(((0.8 * force_sig_len) / fsamp).item()), int(((force_sig_len - 1) / fsamp).item())])

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


def sort_mus(mu_pulses: dict):
    mu_sort = sorted(mu_pulses, key=lambda x: x[0][0])

    return mu_sort


def pulses_to_list(mu_pulses):
    array_list = []

    for array in mu_pulses[0]:
        array_list.append(array)

    return array_list


def calculate_idr(mu_pulse, fsamp):
    timediff = np.diff(mu_pulse / fsamp)

    idr = 1 / timediff

    time_stamps = mu_pulse[0][:-1] / fsamp

    return [time_stamps, idr]


def plot_idr_and_force(idr_01, idr_02, mu_nr_01, mu_nr_02, force_signal, fsamp):
    time_vector = np.rot90((np.arange(0, len(force_signal)) / fsamp))

    fig, axs1 = plt.subplots(figsize=(8, 4))

    axs1.set_ylabel("Discharge Rate (Hz)")
    axs1.set_xlabel("Time (s)")
    axs1.scatter(x=idr_01[0], y=idr_01[1], c='blue', label=f"MU #{mu_nr_01}")
    axs1.scatter(x=idr_02[0], y=idr_02[1], c='red', label=f"MU #{mu_nr_02}")

    axs2 = axs1.twinx()
    axs2.set_ylabel("Force Signal (N)")
    axs2.plot(time_vector, force_signal, c='black', label='Force Signal')

    fig.legend()
    plt.title("Instantaneous Discharge Rate with Force Signal")
    plt.tight_layout()
    plt.show()


def main():
    cwd = os.getcwd()
    file_path = os.path.join(cwd, "Data", "iEMG_contraction.mat")

    data = import_data(file_path)

    EMGSig = data['EMGSig']

    mu_pulses = data['MUPulses']
    mu_pulses = pulses_to_list(mu_pulses)
    sorted_mu = sort_mus(mu_pulses)

    force_signal = data['force_signal']

    fsamp = data['fsamp']

    signal_length_samples = data['signal_length_samples']

    firing_matrix = mu_pulses_to_firing_matrix(mu_pulses)
    sorted_firing_matrix = mu_pulses_to_firing_matrix(sorted_mu)

    idr_01 = calculate_idr(mu_pulses[1], fsamp)
    idr_02 = calculate_idr(mu_pulses[2], fsamp)

    # Plots
    # Plot original spike train
    # plot_spike_trains_and_force(firing_matrix=firing_matrix, force_signal=force_signal, fsamp=fsamp)

    # Plot sorted spike train
    # plot_spike_trains_and_force(firing_matrix=sorted_firing_matrix, force_signal=force_signal, fsamp=fsamp)

    plot_idr_and_force(idr_01=idr_01, idr_02=idr_02, mu_nr_01=1, mu_nr_02=2, force_signal=force_signal, fsamp=fsamp)


if __name__ == "__main__":
    main()
