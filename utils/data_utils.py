import numpy as np

def create_training_sets(bitstream, n, num_of_offset_windows, verbose):
    num_of_offset_bits = num_of_offset_windows * n

    # region 2 creation
    region2_sequences = []
    target_bits = []
    for i in range(num_of_offset_bits, len(bitstream) - n):
        temp_region2_bits = bitstream[i : i + n]
        region2_sequences.append(temp_region2_bits)
        target_bits.append(bitstream[i + n])
    region2_sequences = np.array(region2_sequences)
    target_bits = np.array(target_bits)

    # region 1 creation
    region1_sequences = []
    for i in range(0, len(target_bits)):
        temp_region1_bits = bitstream[i : i + n]
        region1_sequences.append(temp_region1_bits)
    region1_sequences = np.array(region1_sequences)

    restricted_set = region2_sequences
    unrestricted_set = np.concatenate((region1_sequences, region2_sequences), axis=1)

    if (verbose):
        print("restricted set shape after pulling off target: ", np.array(restricted_set).shape)
        print("unrestricted set shape after pulling off target: ", np.array(unrestricted_set).shape)
        print("target shape: ", np.array(target_bits).shape)

    return restricted_set, unrestricted_set, target_bits

def approximate_null_model(target):
    # --- Approximate statsmodels' llr_pvalue for each model ---
    # Null (intercept-only) model log-likelihood: Bernoulli MLE with p = mean(y)
    y = np.asarray(target, dtype=np.float64)
    eps = 1e-12
    p_hat = y.mean()
    ll_null = np.sum(
        y * np.log(p_hat + eps) +
        (1.0 - y) * np.log(1.0 - p_hat + eps)
    )
    return ll_null