import numpy as np


def calculate_sta(emg, mu_sig, mu_num, time, fsamp):
    sample_window = int(((time * fsamp) // 2).item())
    mu = mu_sig[mu_num]

    sta = []

    for y in range(emg.shape[1]):
        for x in range(emg.shape[0]):
            sig = emg[x][y]

            electrode_sta = []

            for mu_idx in mu[0]:
                if sample_window < mu_idx < (sig.shape[1] - sample_window):
                    electrode_sta.append(sig[0][mu_idx - sample_window:mu_idx + sample_window])

            sta.append(np.mean(electrode_sta, axis=0))

    return sta
