# Test: area of a two-array BCPNN mapping (shared periphery) vs the arrays' own areas.
import sys, types, torch
for m in ['cv2','torchvision','torchvision.transforms','bindsnet.datasets']: sys.modules[m]=types.ModuleType(m)
sys.path.insert(0,'.'); sys.path.insert(0,'../..')
from simbrain.mapping import BCPNNMapping
sim = {'device_structure':'trace','device_name':'ideal','c2c_variation':False,'d2d_variation':0,'stuck_at_fault':False,
 'retention_loss':False,'aging_effect':0,'wire_width':200,'input_bit':1,'batch_interval':250,'CMOS_technode':45,
 'ADC_precision':16,'ADC_setting':4,'ADC_rounding_function':'floor','device_roadmap':'HP','temperature':300,'hardware_estimation':True}
m = BCPNNMapping(sim_params=sim, shape=(10,10), traces=('Zj','Pj'))      # a (10,10) layer -> (1,100) row per array
m.total_area_calculation()
arrays = {k: float(a.area.array_area) for k, a in m.mem_arrays.items()}
print('per-array area (m^2):', {k: f'{v:.3e}' for k, v in arrays.items()}, '| sum', f'{sum(arrays.values()):.3e}')
print('sim_area:', {k: (f'{v:.3e}' if isinstance(v,(int,float)) or hasattr(v,'item') else v) for k, v in m.sim_area.items()})
print('periphery:', {k: f'{float(v):.3e}' for k, v in m.sim_periph_area.items()})
print('array share of total: %.1f%%' % (100*float(m.sim_area['sim_mem_area'])/float(m.sim_area['sim_total_area'])))
