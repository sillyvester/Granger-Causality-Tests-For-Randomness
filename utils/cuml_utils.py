import statsmodels.api as sm

def fit_cuml_logit_and_ll(X_np, y_np, verbose=False):
    import cupy as cp
    from cuml.linear_model import LogisticRegression as cuLogisticRegression
    """
    Fit a cuML LogisticRegression model on GPU and return:
    - model
    - log-likelihood (float, on CPU)
    """
    # Move to GPU
    X_np = sm.add_constant(X_np)  # trying to get as close to statsmodels as possible
    X_gpu = cp.asarray(X_np, dtype=cp.float64)
    y_gpu = cp.asarray(y_np, dtype=cp.float64)

    # Unpenalized logistic (to mimic statsmodels MLE as closely as possible)
    model = cuLogisticRegression(
        penalty=None,
        fit_intercept=False,
        max_iter=20_000,
        tol=1e-10,
        verbose=0
    )
    model.fit(X_gpu, y_gpu)

    # Predict probabilities for class 1
    probs_gpu = model.predict_proba(X_gpu)[:, 1]

    # Log-likelihood of Bernoulli logistic model
    eps = 1e-12
    ll_gpu = cp.sum(
        y_gpu * cp.log(probs_gpu + eps) +
        (1.0 - y_gpu) * cp.log(1.0 - probs_gpu + eps)
    )

    ll = float(ll_gpu.get())  # back to CPU
    del X_gpu, y_gpu, probs_gpu, ll_gpu, model
    cp.get_default_memory_pool().free_all_blocks()
    if verbose:
        print("  Log-likelihood:", ll)

    return ll