"""Keep correctness checks small and independent of GPU simulation jobs."""
import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["MEDIENC_DEVICE"] = "cpu"
os.environ["MEDIENCODER_QUIET_TQDM"] = "1"
