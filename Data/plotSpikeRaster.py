import numpy as np
import matplotlib.pyplot as plt

def plot_spike_raster(spikes, plot_type='horzline', fig_handle=None, line_format=None, marker_format=None, auto_label=False, x_lim_for_cell=None, time_per_bin=0.001, spike_duration=0.001, rel_spike_start_time=0, raster_window_offset=None, vert_spike_position=0, vert_spike_height=1):
    if fig_handle is None:
        fig_handle = plt.gca()
    if line_format is None:
        line_format = {'color': [0.2, 0.2, 0.2], 'linewidth': 0.35, 'linestyle': '-'}
    if marker_format is None:
        marker_format = {'markersize': 1, 'color': [0.2, 0.2, 0.2], 'linestyle': 'none'}
    if x_lim_for_cell is None:
        x_lim_for_cell = [np.nan, np.nan]

    if raster_window_offset is not None and rel_spike_start_time == 0:
        rel_spike_start_time = raster_window_offset
    elif raster_window_offset is not None and rel_spike_start_time != 0:
        print('Warning: RasterWindowOffset and RelSpikeStartTime perform the same function. The value set in RasterWindowOffset will be used over RelSpikeStartTime')
        rel_spike_start_time = raster_window_offset

    fig_handle.hold(True)

    if isinstance(spikes, np.ndarray) and spikes.dtype == bool:
        n_trials, n_times = spikes.shape
        spike_duration /= time_per_bin
        rel_spike_start_time /= time_per_bin

        fig_handle.set_xlim([0 + rel_spike_start_time, n_times + 1 + rel_spike_start_time])
        fig_handle.set_ylim([0, n_trials + 1])

        if plot_type == 'horzline':
            trials, timebins = np.where(spikes)
            x_points = np.vstack([timebins + rel_spike_start_time, timebins + rel_spike_start_time + spike_duration, np.nan * np.ones_like(timebins)]).flatten('F')
            y_points = np.vstack([trials + vert_spike_position, trials + vert_spike_position, np.nan * np.ones_like(trials)]).flatten('F')
            fig_handle.plot(x_points, y_points, **line_format)
        elif plot_type == 'vertline':
            trials, timebins = np.where(spikes)
            half_spike_height = vert_spike_height / 2
            x_points = np.vstack([timebins + rel_spike_start_time, timebins + rel_spike_start_time, np.nan * np.ones_like(timebins)]).flatten('F')
            y_points = np.vstack([trials - half_spike_height + vert_spike_position, trials + half_spike_height + vert_spike_position, np.nan * np.ones_like(trials)]).flatten('F')
            fig_handle.plot(x_points, y_points, **line_format)
        elif plot_type == 'scatter':
            trials, timebins = np.where(spikes)
            x_points = timebins + rel_spike_start_time
            y_points = trials
            fig_handle.plot(x_points, y_points, '.', **marker_format)
        elif plot_type == 'imagesc':
            fig_handle.imshow(spikes, aspect='auto', cmap='gray_r')
        else:
            raise ValueError('Invalid plot type. Must be horzline, vertline, scatter, or imagesc')

        fig_handle.set_yticks(np.arange(0.5, n_trials + 1, 1))
        fig_handle.set_yticklabels(np.arange(1, n_trials + 1, 1))
        fig_handle.invert_yaxis()

        if auto_label:
            fig_handle.set_xlabel('Time (ms)')
            fig_handle.set_ylabel('Trial')

    elif isinstance(spikes, list) and all(isinstance(trial, np.ndarray) for trial in spikes):
        n_trials = len(spikes)
        if np.isnan(x_lim_for_cell).any():
            min_time = min(min(trial) for trial in spikes if len(trial) > 0)
            max_time = max(max(trial) for trial in spikes if len(trial) > 0)
            time_range = max_time - min_time
            x_start_offset = rel_spike_start_time - 0.0005 * time_range
            x_end_offset = rel_spike_start_time + 0.0005 * time_range + spike_duration
            x_lim_for_cell = [min_time + x_start_offset, max_time + x_end_offset]
        fig_handle.set_xlim(x_lim_for_cell)
        fig_handle.set_ylim([0, n_trials + 1])

        if plot_type in ['vertline', 'horzline']:
            x_points = []
            y_points = []
            for trial_num, trial in enumerate(spikes):
                if plot_type == 'vertline':
                    half_spike_height = vert_spike_height / 2
                    for spike_time in trial:
                        x_points.extend([spike_time + rel_spike_start_time, spike_time + rel_spike_start_time, np.nan])
                        y_points.extend([trial_num + 1 - half_spike_height + vert_spike_position, trial_num + 1 + half_spike_height + vert_spike_position, np.nan])
                elif plot_type == 'horzline':
                    for spike_time in trial:
                        x_points.extend([spike_time + rel_spike_start_time, spike_time + rel_spike_start_time + spike_duration, np.nan])
                        y_points.extend([trial_num + 1 + vert_spike_position, trial_num + 1 + vert_spike_position, np.nan])
            fig_handle.plot(x_points, y_points, **line_format)
        elif plot_type == 'scatter':
            x_points = [spike_time for trial in spikes for spike_time in trial]
            y_points = [trial_num + 1 for trial_num, trial in enumerate(spikes) for _ in trial]
            fig_handle.plot(x_points, y_points, '.', **marker_format)
        else:
            raise ValueError('Invalid plot type. With cell array of spike times, plot type must be horzline, vertline, or scatter')

        fig_handle.invert_yaxis()

        if auto_label:
            fig_handle.set_xlabel('Time (s)')
            fig_handle.set_ylabel('Trial')

    else:
        raise ValueError('Spikes must be a binary numpy array or a list of numpy arrays')

    fig_handle.hold(False)
    plt.show()