# Calibrate the three ihprj pairs from the recorded engine traces (ideal device) and report tracking error.
import sys, time, torch, numpy as np; sys.path.insert(0, '../..')
from simbrain.mapping import BCPNNMapping
sim = {'device_structure':'trace','device_name':'ideal','c2c_variation':False,'d2d_variation':0,'stuck_at_fault':False,
 'retention_loss':False,'aging_effect':0,'wire_width':200,'input_bit':1,'batch_interval':3,'CMOS_technode':45,
 'ADC_precision':16,'ADC_setting':4,'ADC_rounding_function':'floor','device_roadmap':'HP','temperature':300,'hardware_estimation':False}
m = BCPNNMapping(sim_params=sim, shape={'Pi': 10, 'Pj': 10, 'Pji': 10})   # small arrays: only the calibration is under test here
t0 = time.time(); m.voltage_generation_bcpnn('notes/ihprj_golden_traces.npz', taupdt=0.001, plot=False)
print('calibration time %.1f s' % (time.time()-t0))
for k, v in m.v_pair.items(): print(f'{k:4s} pair (v_pos, v_neg) = ({float(v[0]):.4f}, {float(v[1]):.4f})')
