def plot_returns(all_returns, all_evaluation_returns, window_size, evaluate_every, filename):
    """
    Plot the returns of all training episodes and the averaged evaluation return of each evaluation.

    :param all_returns: The returns of all training episodes per run.
    :param all_evaluation_returns: The averaged evaluation return of each evaluation per run.
    :param window_size: The size of the window for the moving average.
    :param evaluate_every: Frequency of evaluation.
    :param filename: The filename to save the plot to.
    :return: None
    """
    # Calculate mean and standard deviation of returns and evaluation returns.
    # Calculate a moving average of mean and standard deviation of the returns.
    all_returns_mean = np.mean(all_returns, axis=0)
    all_returns_std = np.std(all_returns, axis=0)
    all_returns_mean = np.convolve(all_returns_mean, np.ones(window_size) / window_size, mode="valid")
    all_returns_std = np.convolve(all_returns_std, np.ones(window_size) / window_size, mode="valid")
    all_evaluation_returns_mean = np.mean(all_evaluation_returns, axis=0)
    all_evaluation_returns_std = np.std(all_evaluation_returns, axis=0)

    # Plot the returns and evaluation returns with standard deviation
    plt.plot(all_returns_mean, label=f"Training (running average over {window_size} episodes)")
    plt.fill_between(np.arange(len(all_returns_mean)), all_returns_mean - all_returns_std, all_returns_mean + all_returns_std, alpha=0.2)
    plt.plot(np.arange(evaluate_every, all_returns.shape[1] + 1, evaluate_every), all_evaluation_returns_mean,
             label="Evaluation")
    plt.fill_between(np.arange(evaluate_every, all_returns.shape[1] + 1, evaluate_every),
                     all_evaluation_returns_mean - all_evaluation_returns_std, all_evaluation_returns_mean + all_evaluation_returns_std, alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.gcf().set_size_inches(15, 4)
    plt.savefig(filename, dpi=300, bbox_inches="tight")