# Step 4: standalone verification of BCPNNMapping at full ihprj size.
# Drive the three memristor arrays with the 200 recorded gated steps and compare, cell by cell, with the
# engine's recorded Pi, Pj and the 1000 recorded Pji columns.  Run from examples/BCPNN:
#   python3 notes/test_full_arrays.py
# [RECONSTRUCTED 2026-10-29 from the session transcript after accidental deletion: skeleton of 2026-10-19, the
#  body as printed on 2026-10-26, the user's ADC read-back edit of 2026-10-27; the final print line re-created to
#  give the same columns as before.  Import path now '.', because the BCPNN SIMBRAIN lives in examples/BCPNN/simbrain.]
import sys, time, torch, numpy as np
sys.path.insert(0, '.')
from simbrain.mapping import BCPNNMapping

sim = {'device_structure':'trace','device_name':'ideal','c2c_variation':False,'d2d_variation':0,'stuck_at_fault':False,
       'retention_loss':False,'aging_effect':0,'wire_width':200,'input_bit':1,'batch_interval':3,'CMOS_technode':45,
       'ADC_precision':16,'ADC_setting':4,'ADC_rounding_function':'floor','device_roadmap':'HP','temperature':300,
       'hardware_estimation':False}

# ---- 1. the recording (same file the calibration used) and the ihprj geometry ----
d = np.load('notes/ihprj_golden_traces.npz')
axoact, hihjhi, zj = d['axoact'], d['hihjhi'], d['zj']          # drives
pi, pj, pji, pji_index = d['pi'], d['pj'], d['pji'], d['pji_index']   # engine's traces (the reference)
T = axoact.shape[0]
Hj, Mj, denHi, Mi = 30, 100, 784, 2
denNi, Nj = denHi * Mi, Hj * Mj
eps = 1e-8

# ---- 2. the mapping at real size: build, calibrate, batch of one ----
# (sizes: Pi = Hj*denNi, Pj = Nj, Pji = Nj*denNi -- write them from the constants above, not as literals)
m = BCPNNMapping(sim_params=sim, shape= {'Pi' : Hj*denNi, 'Pj' : Nj, 'Pji' : Nj*denNi})
m.voltage_generation_bcpnn('notes/ihprj_golden_traces.npz')
m.set_batch_size_bcpnn(1,True)

# ---- 3. rebuild Zi for every cell and every step, as in voltage_generation_bcpnn: shape (T, Hj*denNi) ----
hr = hihjhi.reshape(T, Hj, denHi).astype(int)
zi = np.empty((T,Hj*denNi),dtype=np.float32)

for t in range(T):
    src = (hr[t][:,:,None] * Mi + np.arange(Mi)[None, None, :]).reshape(Hj, denNi)
    zi[t] = (axoact[t][src] * (1-eps) + eps).reshape(-1)
# ---- 4. the per-step drives, as the hook will have to form them each step:
#         Pi  <- zi[t]                                   (Hj*denNi,)
#         Pj  <- zj[t]                                   (Nj,)
#         Pji <- Zi * Zj for every synapse, laid out as the engine does: index nj*denNi + dni, with nj = hj*Mj + mj.
#         Hint: zi[t].reshape(Hj, denNi) and zj[t].reshape(Hj, Mj) broadcast to (Hj, Mj, denNi); flatten.

drives = {'Pi': zi, 'Pj': zj}  
rec = {'Pi': pi, 'Pj': pj, 'Pji': pji} 
            
dev = m.memristor_info_dict['ideal']
print('device thresholds v_off %.3f v_on %.3f' % (dev['v_off'], dev['v_on']))

# ---- 5. run the arrays over the recording.  Remember the recording starts one update in (see the prepend
#         in voltage_generation_bcpnn): all cells start at 0 and the FIRST drive is applied once before step 0.
#         Keep x_pi, x_pj, x_pji as (1, N) tensors fed back each step; record the max |mem - recorded| per step.
t0 = time.time()
for name in ('Pi', 'Pj'):
    x = torch.zeros(1, drives[name].shape[1], dtype=torch.float32)
    out_1 = []
    out_2 = []
    for t in range(T):    
        m.mapping_write_bcpnn(drive=torch.from_numpy(drives[name][t].reshape(1,-1)), x_current=x, trace=name)
        x = m.mapping_read_bcpnn(name).reshape(1,-1)
        out_1.append(np.abs(x[0].numpy() - rec[name][t]).max())
        out_2.append(np.abs(x[0].numpy() - rec[name][t]).mean())
    out_1 = np.array(out_1)
    out_2 = np.array(out_2)
    
    print("Pi median:", np.median(rec['Pi'][-1]))
    print("Pi max:", rec['Pi'][-1].max())
    print("Pj median:", np.median(rec['Pj'][-1]))
    print("Pj max:", rec['Pj'][-1].max())
    
    g=rec[name]
    up = (drives[name][1:] > g[:-1]).mean()
    
# ---- 6. report: per array, max and mean absolute error over all cells and steps, and the same relative to
#         the recorded maximum; for Pji compare only the recorded columns pji_index. Also print wall time. ----
    print(f'{name:4s} pair ({m.v_pair[name]}) | golden range {g.min():.2e}..{g.max():.2e} '
          f'| max|mem-golden| {out_1.max():.2e} | rel {out_1.max()/g.max():.1%} '
          f'| mean|mem-golden| {out_2.mean():.2e} | rel_mean {out_2.mean()/g.mean():.1%} '
          f'| steps needing +pulse {up:.0%}')
print(f'wall {time.time()-t0:.1f} s (arrays only)')
