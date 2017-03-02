import numpy as np
import time

def _str_fmt_time(seconds):
    int_seconds = int(seconds)
    hours = int_seconds//3600
    minutes = (int_seconds%3600)//60
    seconds = int_seconds%60 + (seconds - int_seconds)
    return "%.2dh:%.2dm:%.3fs" % (hours, minutes, seconds)

def _batches_gen(X, y, batch_size, shuffle=False):
    n_samples = len(y)

    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples-batch_size+1, batch_size):
        excerpt = indices[start_idx:start_idx+batch_size]
        yield X[excerpt], y[excerpt]

def _inf_gen():
    n = 0
    while True:
        yield n
        n += 1

def train_loop(
    X_tr, y_tr, tr_f,
    n_epochs=10, batch_size=1,
    X_val=None, y_val=None, val_f=None, val_acc_tol=None,
    max_its=None,
    verbose=2):
    """
    General Training loop.
    Parameters:
    X_tr : numpy ndarray
        Training input.
    y_tr : numpy ndarray
        Training outputs.
    tr_f : callable
        Training function giving loss.
    n_epochs : int or None
        Number of epochs. If None, is infinite.
    batch_size : int
        Batch size.
    X_val : numpy ndarray or None
        Validation input.
    y_val : numpy ndarray or None
        Validation output.
    val_f : callable or None
        Validation function giving a tuple of (loss, accuracy).
    val_acc_tol : float or None
        If difference of curr/last validations < val_acc_tol, stop.
    max_its : int or None
        Maximum number of iterations.
    verbose : int
        Prints nothing if 0, only warnings if 1 and everything if >= 2.
    """

    #info/warning functions
    info = print if verbose >= 2 else lambda *args, **kwargs: None
    warn = print if verbose >= 1 else lambda *args, **kwargs: None

    validation = X_val is not None and y_val is not None and val_f is not None

    if not val_acc_tol and n_epochs is None and max_its is None:
        warn("WARNING: training_loop will never stop since"
            " val_acc_tol, n_epochs and max_its are all None")

    n_tr_batches = len(y_tr)//batch_size
    n_val_batches = max(len(y_val)//batch_size, 1) if validation else None
    last_val_acc = None
    its = 0
    start_time = time.time()

    info("starting training loop...")
    for epoch in _inf_gen():
        if n_epochs is not None and epoch >= n_epochs:
            warn("\nWARNING: maximum number of epochs reached")
            end_reason = "n_epochs"
            return end_reason

        info("epoch %d/%s:" % (epoch+1,
            str(n_epochs) if n_epochs is not None else "?"))

        tr_err = 0
        tr_batch_n = 0
        for batch in _batches_gen(X_tr, y_tr, batch_size, True):
            inputs, targets = batch
            err = tr_f(inputs, targets)
            tr_err += err
            tr_batch_n += 1
            info("\r    [train batch %d/%d] err: %.4g    " %\
                (tr_batch_n, n_tr_batches, err), end="")

            its += 1
            if max_its is not None and its > max_its:
                warn("\nWARNING: maximum number of iterations reached")
                done_looping = True
                end_reason = "max_its"
                return end_reason
        tr_err /= n_tr_batches

        val_err = 0
        val_acc = 0
        val_batch_n = 0
        if validation:
            for batch in _batches_gen(X_val, y_val, batch_size, False):
                inputs, targets = batch
                err, acc = val_f(inputs, targets)
                val_err += err
                val_acc += acc
                val_batch_n += 1
                info("\r    [val batch %d/%d] err: %.4g | acc: %f   " %\
                    (val_batch_n, n_val_batches, err, acc), end="")
            val_err /= n_val_batches
            val_acc /= n_val_batches

            if last_val_acc is not None:
                val_acc_diff = abs(val_acc - last_val_acc)
                if val_acc_tol is not None and val_acc_diff < val_acc_tol:
                    warn("\nWARNING: val_acc_diff=%f < val_acc_tol=%f" %\
                        (val_acc_diff, val_acc_tol))
                    done_looping = True
                    end_reason = "val_acc_tol"
                    return end_reason

            last_val_acc = val_acc

        info("\r" + 64*" ", end="")
        info("\r    elapsed time so far: %s" %\
            _str_fmt_time(time.time() - start_time))
        info("    train loss: %.4g" % (tr_err))
        if validation:
            info("    val loss: %.4g" % val_err, end="")
            info(" | val accuracy: %.2f%%" % (val_acc*100))
