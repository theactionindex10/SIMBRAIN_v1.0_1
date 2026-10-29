# Throwaway check: does a trace value handed to the engine through set_field_flt reach the weights via updbw?
# Run from examples/BCPNN:   python3 notes/test_set_updbw.py
# Expectation (Prj::updbw, BCP rule): w = ln(Pji / (Pi*Pj)) on ACTIVE connections, 0 on silent ones.
# So PJI * 1.1  ->  every ACTIVE weight shifts by exactly ln(1.1) ~ 0.0953; silent weights stay 0.
# [RECONSTRUCTED 2026-10-29 from the session transcript after accidental deletion: skeleton of 2026-10-16 plus the
#  user's sections 2-5 as printed on 2026-10-16 14:01.]

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # examples/BCPNN on the path
import numpy as np
import BcpnnSim as sim
from Model_to_BCPNN import Params, BCPNNMNISTModel, iter_encoded_mnist     # the script's own pieces
from preprocess import MNISTDataset

# ---- 1. build the pyMNIST network as the script does, and present ONE pattern (one gated update) ----
param = Params(trnpat=1, tenpat=1, nepoch_usup=1, nepoch_sup=0)
mnist = MNISTDataset()
model = BCPNNMNISTModel(param)
pat, lbl = next(iter_encoded_mnist(mnist, "train", 1))
model.train_unsupervised_online(pat)
ihprj = model.ihprj

# ---- 2. read the weights and the joint trace, keep copies ----
def _read_weight_trace(prj):
    weight_store = np.asarray(prj.get_field_flt("WJI"))
    pji_store = np.asarray(prj.get_field_flt("PJI"))
    return weight_store, pji_store

# ---- 3. the known change: scale PJI by 1.1 ----
w_before, pji_before = _read_weight_trace(ihprj)
print("Before weights:", w_before) 
print("Before pji:", pji_before) 
p_new = np.asarray(1.1 * pji_before)

# ---- 4. hand it to the engine, rebuild the weights ----
ihprj.set_field_flt("PJI", p_new.astype(np.float32))
ihprj.updbw(force=True)

# ---- 5. read the weights again and compare: active connections +ln(1.1), silent ones unchanged (0) ----
# Hint: the active/silent flags are ihprj.get_field_int("CHJHI"), shape (Hj*denHi,) = (30*784,), one per
#       (hidden HC, slot); a weight at [nj, dni] belongs to slot dhi = dni // Mi of hidden HC hj = nj // Mj.
w_after, pji_after = _read_weight_trace(ihprj)
print("After weights:", w_after)  
print("After pji:", pji_after)
dw = w_after - w_before
print("shift on changed weights: min %.4f max %.4f (expected ln 1.1 = %.4f)" % (dw[dw != 0].min(), dw[dw != 0].max(), np.log(1.1)))
