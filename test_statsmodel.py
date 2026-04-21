from utils import gtr_and_pst, pst_only
import numpy as np
import pytest

# Run by just doing `pytest -q` in terminal
def test_648368_to_668367_statsmodel():
    bitstream = np.fromfile("sample_data/NIST_20000_pulses_chain2_648368_to_668367.bin", dtype=np.uint8)
    bitstream = np.unpackbits(bitstream)
    bitstream = bitstream[:10000000]
    bitstream = bitstream.astype(np.int64)

    gtr_result, pst_result, pst_recent_window_only = gtr_and_pst(bitstream, 31, 6, False, False)

    assert gtr_result.pvalue == pytest.approx(0.103, abs=1e-3)
    assert pst_result.pvalue == pytest.approx(0.107, abs=1e-3)
    assert pst_recent_window_only.pvalue == pytest.approx(0.290 , abs=1e-3)

    assert gtr_result.llr_test_stat == pytest.approx(41.247, abs=1e-3)
    assert pst_result.llr_test_stat == pytest.approx(76.099, abs=1e-3)
    assert pst_recent_window_only.llr_test_stat == pytest.approx(34.852, abs=1e-3)

def test_648368_to_668367_statsmodel_with_prefit_pst():
    bitstream = np.fromfile("sample_data/NIST_20000_pulses_chain2_648368_to_668367.bin", dtype=np.uint8)
    bitstream = np.unpackbits(bitstream)
    bitstream = bitstream[:10000000]
    bitstream = bitstream.astype(np.int64)
    
    pst_result, pst_model = pst_only(bitstream, 31, 6, False, False)
    assert pst_result.pvalue == pytest.approx(0.107, abs=1e-3)
    assert pst_result.llr_test_stat == pytest.approx(76.099, abs=1e-3)

    gtr_result, pst_result, pst_recent_window_only = gtr_and_pst(bitstream, 31, 6, False, False, True, pst_model)
    assert gtr_result.pvalue == pytest.approx(0.103, abs=1e-3)
    assert gtr_result.llr_test_stat == pytest.approx(41.247, abs=1e-3)
    assert pst_result.pvalue == pytest.approx(0.107, abs=1e-3)
    assert pst_result.llr_test_stat == pytest.approx(76.099, abs=1e-3)

    