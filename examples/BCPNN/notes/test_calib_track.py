# Re-simulate each calibrated pair on a 1-cell array over the recorded drive and compare with the golden P.
import sys, torch, numpy as np; sys.path.insert(0, '../..')
from simbrain.mapping import BCPNNMapping
sim = {'device_structure':'trace','device_name':'ideal','c2c_variation':False,'d2d_variation':0,'stuck_at_fault':False,
 'retention_loss':False,'aging_effect':0,'wire_width':200,'input_bit':1,'batch_interval':3,'CMOS_technode':45,
 'ADC_precision':16,'ADC_setting':4,'ADC_rounding_function':'floor','device_roadmap':'HP','temperature':300,'hardware_estimation':False}
m = BCPNNMapping(sim_params=sim, shape={'Pi': 1, 'Pj': 1, 'Pji': 1})
m.voltage_generation_bcpnn('notes/ihprj_golden_traces.npz'); m.set_batch_size_bcpnn(1, True)
# rebuild the same representative drives/goldens the method used
d = np.load('notes/ihprj_golden_traces.npz'); axo, hih, zj, pi, pj, pji, idx = d['axoact'], d['hihjhi'], d['zj'], d['pi'], d['pj'], d['pji'], d['pji_index']
T = axo.shape[0]; Hj, denHi, Mi, denNi, Mj, eps = 30, 784, 2, 1568, 100, 1e-8
hr = hih.reshape(T, Hj, denHi).astype(int)
zi = np.stack([(axo[t][(hr[t][:, :, None]*Mi + np.arange(Mi)[None,None,:]).reshape(Hj, denNi)]*(1-eps)+eps).reshape(-1) for t in range(T)])
i_pi = int(np.argmin(np.abs(pi[-1]-np.median(pi[-1])))); i_pj = int(np.argmin(np.abs(pj[-1]-np.median(pj[-1])))); i_pji = int(np.argmin(np.abs(pji[-1]-np.median(pji[-1]))))
k = idx[i_pji]; nj = k // denNi; dni = k % denNi; hj = nj // Mj
drives = {'Pi': zi[:, i_pi], 'Pj': zj[:, i_pj], 'Pji': zi[:, hj*denNi+dni]*zj[:, nj]}
rec = {'Pi': pi[:, i_pi], 'Pj': pj[:, i_pj], 'Pji': pji[:, i_pji]}
dev = m.memristor_info_dict['ideal']; print('device thresholds v_off %.3f v_on %.3f' % (dev['v_off'], dev['v_on']))
for name in ('Pi', 'Pj', 'Pji'):
    x = torch.full((1, 1), float(rec[name][0]), dtype=torch.float64)      # start where the recording starts
    out = [x.item()]
    for t in range(1, T):
        x = m.mapping_write_bcpnn(drive=torch.tensor([[drives[name][t]]]), x_current=x, trace=name).reshape(1, 1)
        out.append(x.item())
    out = np.array(out); g = rec[name]
    up = (drives[name][1:] > g[:-1]).mean()
    print(f'{name:4s} pair ({float(m.v_pair[name][0]):.4f}, {float(m.v_pair[name][1]):.4f}) | golden range {g.min():.2e}..{g.max():.2e} | max|mem-golden| {np.abs(out-g).max():.2e} | rel {np.abs(out-g).max()/g.max():.1%} | steps needing +pulse {up:.0%}')
