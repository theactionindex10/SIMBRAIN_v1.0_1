# Timing of the three costs per gated learning step, real ihprj sizes, this machine (CPU).
import sys, time, torch, numpy as np; sys.path.insert(0, '../..'); sys.path.insert(0, '.')
from simbrain.mapping import BCPNNMapping
import BcpnnSim as sim
from Model_to_BCPNN import Params, BCPNNMNISTModel, iter_encoded_mnist
from preprocess import MNISTDataset
T = lambda: time.perf_counter()

# ---- 1. engine alone: one updstate_prns on the real network, gated (prn=1) and ungated ----
model = BCPNNMNISTModel(Params(trnpat=1, tenpat=1, nepoch_usup=1, nepoch_sup=0)); mnist = MNISTDataset()
pat, _ = next(iter_encoded_mnist(mnist, "train", 1)); model.reset_state(); model.set_input(pat)
for prns, label in [([0.0, 0.0], 'engine step, learning gate OFF'), ([1.0, 0.0], 'engine step, learning gate ON ')]:
    ts = []
    for _ in range(5):
        t0 = T(); model.net.updstate_prns(prns); sim.advance(); ts.append(T() - t0)
    print(f'{label}: {np.median(ts)*1e3:7.1f} ms')
ih = model.ihprj

# ---- 2. the shuttle: get and set of the three arrays through the binding ----
for f in ('PI', 'PJ', 'PJI'):
    t0 = T(); a = ih.get_field_flt(f); t1 = T(); ih.set_field_flt(f, np.asarray(a, dtype=np.float32)); t2 = T()
    print(f'get {f:3s} ({a.size:>9,d} values): {(t1-t0)*1e3:6.1f} ms | set: {(t2-t1)*1e3:6.1f} ms')
t0 = T(); ih.updbw(force=True); print(f'updbw(force=True): {(T()-t0)*1e3:6.1f} ms')

# ---- 3. the memristor simulation: write + ADC read of each array, ideal device, 16-bit ----
simp = {'device_structure':'trace','device_name':'ideal','c2c_variation':False,'d2d_variation':0,'stuck_at_fault':False,
 'retention_loss':False,'aging_effect':0,'wire_width':200,'input_bit':1,'batch_interval':3,'CMOS_technode':45,
 'ADC_precision':16,'ADC_setting':4,'ADC_rounding_function':'floor','device_roadmap':'HP','temperature':300,'hardware_estimation':False}
sizes = {'Pi': 30*1568, 'Pj': 3000, 'Pji': 3000*1568}
m = BCPNNMapping(sim_params=simp, shape=sizes); m.voltage_generation_bcpnn('notes/ihprj_golden_traces.npz'); m.set_batch_size_bcpnn(1, True)
for name, n in sizes.items():
    x = torch.zeros(1, n, dtype=torch.float32); d = torch.rand(1, n, dtype=torch.float32) * 0.05
    tw, tr = [], []
    for _ in range(3):
        t0 = T(); x = m.mapping_write_bcpnn(drive=d, x_current=x, trace=name); tw.append(T() - t0)
        t0 = T(); r = m.mapping_read_bcpnn(name); tr.append(T() - t0)
    print(f'memristor {name:3s} ({n:>9,d} cells): write {np.median(tw)*1e3:7.1f} ms | ADC read {np.median(tr)*1e3:7.1f} ms')
