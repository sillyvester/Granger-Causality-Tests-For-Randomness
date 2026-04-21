import numpy as np
import statsmodels.api as sm
from scipy import stats
from dataclasses import dataclass

# Custom Imports
from .data_utils import approximate_null_model, create_training_sets

@dataclass
class GTR_Result():
    pvalue: float
    llr_test_stat: float

@dataclass 
class PST_Result():
    pvalue: float
    llr_test_stat: float

@dataclass 
class PST_Recent_Window_Only():
    pvalue: float
    llr_test_stat: float

# PST Only
# This function exists such that you can run the PST only, which only has the unrestricted model fit, 
# compared to the GTR which fits both the restricted and unrestricted models.
# It also returns the unrestricted model such that you could then call the GTR with this prefit unrestricted model
def pst_only(bitstream, window_size, offset, use_cuml=False, verbose=False):
    _, unrestricted_set, target = create_training_sets(bitstream, window_size, offset, verbose)
    
    # Fit the unrestricted model
    unrestricted_set = sm.add_constant(unrestricted_set)
    unrestricted_model = sm.Logit(target, unrestricted_set)
    unrestricted_result = unrestricted_model.fit()

    pst_result = PST_Result(
        pvalue = unrestricted_result.llr_pvalue,
        llr_test_stat = unrestricted_result.llr
    )

    return pst_result, unrestricted_result


def gtr_and_pst(bitstream, window_size, offset, use_cuml=False, verbose=False, use_prefit_unrestricted=False, prefit_unrestricted=None):
    restricted_set, unrestricted_set, target = create_training_sets(bitstream, window_size, offset, verbose)
    
    if use_cuml:
        return gtr_and_pst_cuml(restricted_set, unrestricted_set, target, verbose)
    else:
        return gtr_and_pst_statsmodel(restricted_set, unrestricted_set, target, verbose, use_prefit_unrestricted, prefit_unrestricted)

def gtr_and_pst_statsmodel(restricted_set, unrestricted_set, target, verbose, use_prefit_unrestricted, prefit_unrestricted):
    
    # Setup Datasets
    restricted_set = sm.add_constant(restricted_set)
    unrestricted_set = sm.add_constant(unrestricted_set)

    # Fit the restricted model
    restricted_model = sm.Logit(target, restricted_set)
    restricted_result = restricted_model.fit()

    # Get or Fit the unrestricted model
    if use_prefit_unrestricted:
        unrestricted_result = prefit_unrestricted
    else:
        unrestricted_model = sm.Logit(target, unrestricted_set)
        unrestricted_result = unrestricted_model.fit()

    if (verbose):
        print("Restricted Model Summary:")
        print(restricted_result.summary())

        print("\nUnrestricted Model Summary:")
        print(unrestricted_result.summary())

    lr = -2 * (restricted_result.llf - unrestricted_result.llf)
    if (verbose):
        print("GTR LR: ", lr)
        print( unrestricted_set.shape[1] - restricted_set.shape[1])
        print("GTR P value: ", stats.chi2.sf(lr, unrestricted_set.shape[1] - restricted_set.shape[1]))
    
    gtr_result = GTR_Result(
        pvalue = stats.chi2.sf(lr, unrestricted_set.shape[1] - restricted_set.shape[1]), 
        llr_test_stat = lr
    )
    pst_result = PST_Result(
        pvalue = unrestricted_result.llr_pvalue,
        llr_test_stat = unrestricted_result.llr
    )
    pst_recent_window_only = PST_Recent_Window_Only(
        pvalue = restricted_result.llr_pvalue,
        llr_test_stat = restricted_result.llr
    )
    return gtr_result, pst_result, pst_recent_window_only


def gtr_and_pst_cuml(restricted_set, unrestricted_set, target, verbose):
    """
    GPU-accelerated Granger-style test using cuML LogisticRegression.
    """
    from .cuml_utils import fit_cuml_logit_and_ll # use a lazy import so that you can run the statsmodel version on a non-GPU device without import errors
    if verbose:
        print("Fitting restricted model (GPU)...")
    ll_restricted = fit_cuml_logit_and_ll(restricted_set, target, verbose=verbose)

    if verbose:
        print("Fitting unrestricted model (GPU)...")
    ll_unrestricted = fit_cuml_logit_and_ll(unrestricted_set, target, verbose=verbose)

    # LR statistic for Granger-style comparison
    lr = -2.0 * (ll_restricted - ll_unrestricted)
    df = unrestricted_set.shape[1] - restricted_set.shape[1]  # extra predictors

    causal_pval = stats.chi2.sf(lr, df)
    gtr_result = GTR_Result(
        pvalue = causal_pval, 
        llr_test_stat = lr
    )

    if verbose:
        print("LR (unrestricted vs restricted):", lr)
        print("df:", df)
        print("P value:", causal_pval)

    # --- Approximate statsmodels' llr_pvalue ---
    ll_null = approximate_null_model(target)

    # Restricted vs null (Unnamed Test but it's the PST equivalent for only looking at recent window)
    df_restricted = restricted_set.shape[1]  # number of predictors
    lr_restricted = -2.0 * (ll_null - ll_restricted)
    restricted_pval = stats.chi2.sf(lr_restricted, df_restricted)

    pst_recent_window_only = PST_Recent_Window_Only(
        pvalue = restricted_pval,
        llr_test_stat = lr_restricted
    )

    # Unrestricted vs null (PST)
    df_unrestricted = unrestricted_set.shape[1]
    lr_unrestricted = -2.0 * (ll_null - ll_unrestricted)
    unrestricted_pval = stats.chi2.sf(lr_unrestricted, df_unrestricted)

    pst_result = PST_Result(
        pvalue = unrestricted_pval,
        llr_test_stat = lr_unrestricted
    )

    return gtr_result, pst_result, pst_recent_window_only