# Granger-Causality-Tests-For-Randomness

This repository contains source files for the implementations of Granger-style tests for randomness. 

**Quick summary**: Run the example runner to compute the GTR and PST tests on a sample bitstream provided in sample_data, or modify to run the tests on your own bitstream. Also included are several pytest tests which validate results against known outputs for `NIST_20000_pulses_chain2_648368_to_668367` provided in `sample_data`.

**Citing**: Please cite our publications on the GTR and PST. 

- Sylvester, Joshua H., Thornton, Micah A., Henderson, Jessie M., Thornton, Mitchell A. and Larson, Eric C.. "GTR: Granger-inspired test for randomness in bitstreams" it - Information Technology. https://doi.org/10.1515/itit-2025-0027

- J. H. Sylvester, Micah A. Thornton, E.C. Larson, Mitch A. Thornton (2026). Granger-inspired Predictive Structure Test for RBG/RNG Evaluation. Dallas Circuit and Systems (DCAS) (Apr 10-12, 2026), DOI Pending

- J. H. Sylvester (2026). *Beyond Time Series: Extending Granger Causality for Clustering, Representation Learning, and Randomness Testing*. Ph.D. Dissertation, Southern Methodist University (unpublished).


**Requirements**
- Python 3.8+
- numpy
- scipy
- statsmodels
- pytest (for running tests)
- Optional: RAPIDS cuML if you want GPU-accelerated logistic regressions and are running in a GPU-accelerated environment

**Running the example**

The main example runner is [example.py](example.py). It expects five CLI arguments:

- `bitstream_filename`: path to a .bin file containing raw bytes (the code unpacks those bytes to bits)
- `window_size`: integer window size `n` (e.g., `31`)
- `offset`: integer number of offset windows (e.g., `6`)
- `use_cuml`: `true` or `false` — use GPU-accelerated cuML implementation when available
- `verbose`: `true` or `false` — print additional model summaries

Example (from the repo root):

```bash
python example.py sample_data/NIST_20000_pulses_chain2_648368_to_668367.bin 31 6 false false
```

The script will print three results blocks:
- `GTR`: Granger-style test result (p-value and log-likelihood-ratio statistic)
- `PST`: PST result comparing unrestricted model to null
- `PST recent window only`: PST-equivalent using only the recent window (restricted model)

**Running the tests**

Tests are provided in [test_statsmodel.py](test_statsmodel.py). From the repository root run:

```bash
pytest -q
```

Notes:
- Tests use `sample_data/NIST_20000_pulses_chain2_648368_to_668367.bin` and expect deterministic numeric results.
- If you see convergence warnings from `statsmodels`, try re-running with `verbose=true` or adjusting your environment; these warnings do not necessarily indicate test failures.
- To run the GPU path, set the fourth argument to `true` and ensure RAPIDS/cuML is properly installed and available.

**Running with CUML**
- We provide an example of how to run the GTR and PST with GPU-acceleration in [Run_CUML_Example.ipynb](Run_CUML_Example.ipynb). This notebook provides an example of how to run in a Google Colab GPU Accelerated environment. Google Colab provides affordable GPU access that allows practically anyone to run GPU-accelerated code. However, if you wish to run on your own GPU-accelerated machine, the notebook [Run_CUML_Example.ipynb](Run_CUML_Example.ipynb) is still applicable with minor modifications.

**Useful files**
- Example runner: [example.py](example.py)
- Test suite: [test_statsmodel.py](test_statsmodel.py)
- Utilities and core logic: [utils/granger_test_utils.py](utils/granger_test_utils.py), [utils/data_utils.py](utils/data_utils.py)
- CUML Example Notebook: [Run_CUML_Example.ipynb](Run_CUML_Example.ipynb)

**Note on NIST-STS Comparisons**
- This implementation of the GTR and PST is intended for practitioners who wish to use them in a standalone manner or integrate them into custom workflows. For users interested in running the GTR and PST alongside the NIST Statistical Test Suite (STS), an implementation of NIST-STS is available within the [STEER Framework](https://github.com/SMU-DDI/steer-framework), which also includes implementations of the GTR and PST.