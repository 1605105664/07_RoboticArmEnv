from stable_baselines3.common.results_plotter import *
import matplotlib.pyplot as plt


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, x_axis='cum_timesteps',  y_axis='reward', window=10, label=None):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    data_frame = load_results(log_folder)
    params = {
        'Number of Timesteps': np.cumsum(data_frame.l.values),
        'Episode Count': np.arange(len(data_frame)),
        'Episode Length': data_frame.l.values,
        'Rewards': data_frame.r.values
    }

    x, y = params[x_axis], params[y_axis]
    y = moving_average(y, window)
    # Truncate x
    x = x[len(x) - len(y):]
    title=x_axis+' vs. '+y_axis
    fig = plt.figure(title)
    plt.plot(x, y, label=label)
    if label:
        plt.legend(loc="lower right")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title + " Smoothed, window=" + str(window))

x='Episode Count'
y='Episode Length'
plot_results('./2_0', x_axis=x,  y_axis=y, window=100, label='alpha=0.0')
plot_results('./2_25', x_axis=x,  y_axis=y, window=100, label='alpha=0.25')
plot_results('./2_50', x_axis=x,  y_axis=y, window=100, label='alpha=0.5')
plot_results('./2_75', x_axis=x,  y_axis=y, window=100, label='alpha=0.75')
plot_results('./2_100', x_axis=x,  y_axis=y, window=100, label='alpha=1.0')
plt.show()
