import numpy as np

def spike_triggered_averaging(sig, mu_pulses, sta_window, fsamp):
    """
    Compute the spike-triggered averaging (STA) of EMG signals.

    Parameters:
    sig (list of 2D numpy arrays): EMG signal as a list arranged in the grid shape of electrodes.
    mu_pulses (list of 1D numpy arrays): Samples of motor unit (MU) discharges for all MUs.
    sta_window (float): Window length for STA in seconds.
    fsamp (float): Sampling frequency of EMG signal acquisition.

    Returns:
    list of 2D numpy arrays: STA means for all motor units, shaped like the electrode grid.
    """
    # Convert STA window to samples (half-window size)
    sta_window_samples = int(round((sta_window / 2) * fsamp))

    sta_means = []

    for mu_idx, pulses in enumerate(mu_pulses):
        mu_sta_mean = []

        for row in range(len(sig)):
            row_sta_mean = []

            for col in range(len(sig[row])):
                channel_signal = sig[row][col]

                if channel_signal.size > 0:
                    temp_sta = []

                    for spike in pulses:
                        start_idx = spike - sta_window_samples
                        end_idx = spike + sta_window_samples

                        if 0 <= start_idx < len(channel_signal) and end_idx < len(channel_signal):
                            temp_sta.append(channel_signal[start_idx:end_idx])

                    if temp_sta:
                        row_sta_mean.append(np.mean(temp_sta, axis=0))
                    else:
                        row_sta_mean.append(np.array([]))
                else:
                    row_sta_mean.append(np.array([]))

            mu_sta_mean.append(row_sta_mean)

        sta_means.append(mu_sta_mean)

    return sta_means
