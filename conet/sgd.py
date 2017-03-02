import theano
from theano import tensor
import time
import numpy as np

def _grad(cost, wrt):
    try:
        return tensor.grad(cost=cost, wrt=wrt)
    except theano.gradient.DisconnectedInputError:
        return 0

def _none_or_num(num, after_point_digits=6):
    if num is None:
        return "None"
    ret = str(num)
    if "." in ret:
        b_p, a_p = ret.split(".")
        if num - int(num) == 0:
            return b_p
        else:
            return b_p + "." + a_p[:after_point_digits]
    return ret

def _str_fmt_time(seconds):
    seconds = int(seconds)
    hours = seconds//3600
    minutes = (seconds%3600)//60
    seconds = seconds%60
    return "%.3dh:%.2dm:%.2ds" % (hours, minutes, seconds)

def sgd(est,
        x_tr, y_tr,
        learning_rate=0.1, reg_term=0.1,
        batch_size=1, n_epochs=None,
        max_its=10000, rel_tol=1e-3,
        x=None, y=None,
        verbose=True, print_flush_period=100):
    """
    Stochastic Gradient Descent with regularization.
    Parameters:
    *est: Estimator object. It must have the following attributes:
        .cost(y): cost function (without regularization term)
        .reg: regularization term of cost function
        .params: iterable with model parameters
        .inp: input (only needed if x is None).
    *x_tr: x input matrix (shared).
    *y_tr: y output vector (shared).
    *learning_rate: Learning rate.
    *reg_term: Regularization term.
    *batch_size: Samples for each batch. If None, batch_size = n_samples.
    *n_epochs: Number of epochs to run.
        If None, runs until another stopping criteria.
    *max_its: Maximum number of iterations to run.
    *rel_tol: Relative difference of current/last batch cost. Not used if None.
    *x: Input. If None, x is set to reg.inp.
    *y: Output vector. If None, is set to theano.tensor.ivector.
    *verbose: Print info if True, print nothing otherwise.
    *print_flush_period: Period to flush printed information.
    """

    #info function, only prints something if verbose is True
    info = print if verbose else lambda *args, **kwargs: None

    #setting up x
    if x is None:
        x = est.inp
    if y is None:
        y = tensor.ivector(name="sgd.sgd_y")

    #setting batch size
    n_samples = x_tr.get_value(borrow=True).shape[0]
    if batch_size is None:
        batch_size = n_samples

    #variable for indexing training samples
    index = tensor.lscalar(name="index")

    #parameters gradient of estimator
    grad = [(p, _grad(est.cost(y), p), _grad(est.reg, p)) for p in est.params]

    #updates variables
    updates = [(p, p - learning_rate*(d_cost_p + reg_term*d_reg_p)) \
        for p, d_cost_p, d_reg_p in grad]

    #compiling function
    info("making func...", end="", flush=True)
    train_f = theano.function(
        inputs=[index],
        outputs=est.cost(y),
        updates=updates,
        givens={
            x: x_tr[index*batch_size:(index+1)*batch_size],
            y: y_tr[index*batch_size:(index+1)*batch_size]
        })
    info(" done")

    #start time
    start_time = time.time()
    #counter for number of epochs
    epoch_count = 0
    #total iterations (over batcher) count
    it_count = 0
    #True if its time to stop
    done_looping = False
    #number of batches
    n_batches = n_samples//batch_size
    #last batch cost
    last_batch_cost = None
    #batch cost relative to old
    rel_batch_cost = None
    #criterium that made algorithm stop
    stop_crit = None
    #message of the end
    end_msg = ""

    #main loop
    info("starting optimization")
    while not done_looping:
        batch_cost_sum = 0

        #epochs stopping criteria
        if n_epochs and epoch_count >= n_epochs:
            end_msg = "\nWARNING: reached number of epochs"
            done_looping = True
            stop_crit = "n_epochs"
            continue

        #iterating over batches
        for batch_idx in range(n_batches):
            batch_cost = train_f(batch_idx)
            batch_cost_sum += batch_cost

            elapsed_time = time.time() - start_time
            info(("\r[elapsed_time %s][epoch %d/%s][iter %d/%s][batch %d/%d] "
                "batch_cost: %.4g%s") %\
                (_str_fmt_time(elapsed_time), epoch_count+1,
                _none_or_num(n_epochs), it_count+1, _none_or_num(max_its),
                batch_idx+1, n_batches, batch_cost, 8*" "),
                end="", flush=it_count%print_flush_period == 0)

            #relative cost stopping criteria
            if last_batch_cost and rel_tol:
                rel_batch_cost = abs(1 - last_batch_cost/batch_cost)
                if rel_batch_cost <= rel_tol:
                    end_msg = "\nWARNING: rel_tol criteria matched"
                    done_looping = True
                    stop_crit = "rel_tol"
                    break

            #iterations number stopping criteria
            it_count += 1
            if max_its and it_count >= max_its:
                end_msg = "\nWARNING: maximum iterations reached"
                done_looping = True
                stop_crit = "max_its"
                break

            last_batch_cost = batch_cost

        mean_batch_cost = batch_cost_sum/n_batches
        info(("\n\tnew epoch. mean_batch_cost: %.4g | best_val_score: %f | "
            "rel_val_score: %s") %\
            (mean_batch_cost, best_val_score, _none_or_num(rel_val_score)))
        epoch_count += 1

    info(end_msg)
    elapsed_time = time.time() - start_time
    info("[end of sgd] total elapsed time: %s\n" % _str_fmt_time(elapsed_time))

    return stop_crit

def sgd_with_validation(est, 
        x_tr, y_tr, x_cv, y_cv,
        learning_rate=0.1, reg_term=0.1,
        batch_size=1, n_epochs=10,
        max_its=5000, improv_thresh=0.01, max_its_incr=4,
        rel_val_tol=1e-3, val_freq="auto",
        x=None, y=None,
        verbose=True, print_flush_period=100):
    """
    Stochastic Gradient Descent with regularization using validation set.
    Parameters:
    *est: Estimator object. It must have the following attributes:
        .cost(y): cost function (without regularization term)
        .reg: regularization term of cost function
        .params: iterable with model parameters
        .score(y): accuracy function (in [0, 1])
        .inp: input (only needed if x is None).
    *x_tr: x train input matrix (shared).
    *y_tr: y train output vector (shared).
    *x_cv: x validation input matrix (shared).
    *y_cv: y validation output vector (shared).
    *learning_rate: Learning rate.
    *reg_term: Regularization term.
    *batch_size: Samples for each batch. If None, batch_size = n_samples.
    *n_epochs: Number of epochs to run.
        If None, runs until another stopping criteria.
    *max_its: Maximum number of iterations to run.
    *improv_thresh: Only increments max_its if
        new best val_score >= (1+improv_thresh)*old_best_val_score.
    *max_its_incr: Max iterations to increase when finding a new best val_score.
    *rel_val_tol: If abs(1 - new_val_score/old_val_score) <= rel_val_tol, stop.
    *val_freq: Frequency (in batches) to perform validation.
    *x: Input. If None, x is set to reg.inp.
    *y: Output. If None, y is set to theano.tensor.ivector.
    *verbose: Print info if True, print nothing otherwise.
    *print_flush_period: Period to flush printed information.
    """

    #info function, only prints something if verbose is True
    info = print if verbose else lambda *args, **kwargs: None

    #setting up x
    if x is None:
        x = est.inp
    #setting up y
    if y is None:
        y = tensor.ivector(name="sgd.sgd_with_validation_y")

    #setting batch size
    n_train_samples = x_tr.get_value(borrow=True).shape[0]
    n_val_samples = x_cv.get_value(borrow=True).shape[0]
    if batch_size is None:
        batch_size = n_train_samples

    #variable for indexing training samples
    index = tensor.lscalar(name="index")

    #cost function
    cost = est.cost(y) + reg_term*est.reg
    #updates for variables
    updates = [(p, p - learning_rate*_grad(cost, p)) for p in est.params]

    #compiling functions
    info("making train/val func...", end="", flush=True)
    train_f = theano.function(
        inputs=[index],
        outputs=est.cost(y),
        updates=updates,
        givens={
            x: x_tr[index*batch_size:(index+1)*batch_size],
            y: y_tr[index*batch_size:(index+1)*batch_size]
        })
    val_f = theano.function(
        inputs=[index],
        outputs=est.score(y),
        updates=updates,
        givens={
            x: x_cv[index*batch_size:(index+1)*batch_size],
            y: y_cv[index*batch_size:(index+1)*batch_size]
        })
    info(" done")

    #number of epochs
    epoch_count = 0
    #total iterations (over batcher) count
    it_count = 0
    #true if its time to stop
    done_looping = False
    #number of train batches
    n_train_batches = n_train_samples//batch_size
    #number of validation batches
    n_val_batches = max(n_val_samples//batch_size, 1)
    #validation frequency
    if val_freq == "auto":
        val_freq = min(n_train_batches, max_its//2)
    #best validation score
    best_val_score = None
    #last validation score
    last_val_score = None
    #relative
    rel_val_score = None
    #starting time
    start_time = time.time()
    #criterium that made algorithm stop
    stop_crit = None
    #message of end
    end_msg = ""

    #main loop
    info("starting optimization")
    while not done_looping:
        batch_cost_sum = 0

        #epochs stopping criteria
        if n_epochs and epoch_count >= n_epochs:
            end_msg = "\nWARNING: reached number of epochs"
            done_looping = True
            stop_crit = "n_epochs"
            continue

        #iterating over batches
        for batch_idx in range(n_train_batches):
            batch_cost = train_f(batch_idx)
            batch_cost_sum += batch_cost
             
            if (it_count+1)%val_freq == 0:
                val_score = np.mean([val_f(i) for i in range(n_val_batches)])

                if best_val_score is None or val_score > best_val_score:
                    best_val_score = 0

                    if val_score >= best_val_score*(1+improv_thresh):
                        max_its = max(max_its, it_count+max_its_incr)

                    best_val_score = val_score

                #relative cost stopping criteria
                if last_val_score and rel_val_tol:
                    rel_val_score = abs(1 - last_val_score/val_score)
                    if rel_val_score <= rel_val_tol:
                        end_msg = \
                            "\nWARNING: rel_val_tol criteria matched (%f)" %\
                            rel_val_score
                        done_looping = True
                        stop_crit = "rel_val_tol"
                        break

                last_val_score = val_score

            elapsed_time = time.time() - start_time
            info(("\r[elapsed_time %s][epoch %d/%s][iter %d/%s][batch %d/%d] "
                "batch_cost: %.4g%s") %\
                (_str_fmt_time(elapsed_time), epoch_count+1,
                _none_or_num(n_epochs), it_count+1, _none_or_num(max_its),
                batch_idx+1, n_train_batches, batch_cost, 8*" "),
                end="", flush=it_count%print_flush_period == 0)

            #iterations number stopping criteria
            it_count += 1
            if max_its and it_count >= max_its:
                end_msg = "\nWARNING: maximum iterations reached"
                done_looping = True
                stop_crit = "max_its"
                break

        mean_batch_cost = batch_cost_sum/n_train_batches
        info(("\n\tnew epoch. mean_batch_cost: %.4g | last_val_score: %f | "
            "best_val_score: %f | last_rel_val_score: %s") %\
            (mean_batch_cost, val_score, best_val_score, 
            _none_or_num(rel_val_score)))
        epoch_count += 1

    info(end_msg)
    elapsed_time = time.time() - start_time
    info("[end of sgd_with_validation] elapsed time: %s\n" %\
        _str_fmt_time(elapsed_time))

    return stop_crit
