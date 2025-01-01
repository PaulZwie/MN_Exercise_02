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


def plotSpikeRaster(firing_matrix, fsamp):
    fig, ax = plt.subplots(figsize=(8, 4))

    for i, unit in enumerate(firing_matrix):
        spike_times = [t / fsamp for t, spike in enumerate(unit) if spike == 1]
        for spike_time in spike_times:
            ax.vlines(spike_time, i, i + 0.75, color='k')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Motor Units')
    ax.set_yticks(range(len(firing_matrix)))
    ax.set_yticklabels([f'{i + 1}' for i in range(len(firing_matrix))])
    ax.set_xlim(0, (len(firing_matrix[0]) + 0.95 * fsamp) / fsamp)
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

    plotSpikeRaster(firing_matrix, fsamp)


if __name__ == "__main__":
    main()
