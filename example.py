from utils import gtr_and_pst
import sys
import numpy as np

def main(bitstream_filename, window_size, offset, use_cuml, verbose):
    
    # unpack and set up bitstream to be a numpy array
    # EXPECTS A .BIN FILE THAT CAN BE OPENNED WITH INT8
    bitstream = np.fromfile(bitstream_filename, dtype=np.uint8)
    bitstream = np.unpackbits(bitstream)
    bitstream = bitstream[:10000000]
    
    bitstream = bitstream.astype(np.int64)

    gtr_result, pst_result, pst_recent_window_only = gtr_and_pst(bitstream, window_size, offset, use_cuml, verbose)
    
    print("GTR")
    print(gtr_result)
    print()
    print("PST")
    print(pst_result)
    print()
    print("PST recent window only")
    print(pst_recent_window_only)

if __name__ == "__main__":
    if len(sys.argv) == 6:
        bitstream_filename = sys.argv[1]
        window_size = int(sys.argv[2])
        offset = int(sys.argv[3])
        use_cuml = sys.argv[4].lower() == "true"
        verbose = sys.argv[5].lower() == "true"
        main(bitstream_filename, window_size, offset, use_cuml, verbose)
    else:
        print("Must add parameters as CLI arguments. Expects bistream_filename, window_size, offset, use_cuml, and verbose")