class EarlyStopping:
    """Early stops the training if test acc doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time test acc improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.test_acc_max = 0

        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, test_acc, test_recall_values):

        score = test_acc

        if self.best_score is None:
            self.best_score = score
            self.update_max_test_acc(test_acc)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}. (Best: {self.test_acc_max:.6f})"
                )
            if self.counter >= self.patience:
                self.trace_func(
                    f"**EarlyStopping Triggered: test accuracy stuck at {self.test_acc_max:.6f} for {self.patience} epoch(es)."
                )
                self.early_stop = True
        else:
            self.best_score = score
            self.update_max_test_acc(test_acc)
            self.counter = 0

    def update_max_test_acc(self, test_acc):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Test accuracy increased ({self.test_acc_max:.6f} --> {test_acc:.6f})."
            )
        self.test_acc_max = test_acc