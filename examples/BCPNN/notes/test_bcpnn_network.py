# Test: build BCPNNNetwork exactly as a script would, run it through Network.run, check traces/weights.
import sys, types, torch
for m in ['cv2', 'torchvision', 'torchvision.transforms', 'bindsnet.datasets']:
    sys.modules[m] = types.ModuleType(m)
sys.path.insert(0, '.'); sys.path.insert(0, '../..')
from bindsnet.models import BCPNNNetwork

HW = True  # hardware_estimation on: Network.run prints the energy totals
sim = {'device_structure': 'trace', 'device_name': 'ideal', 'c2c_variation': False, 'd2d_variation': 0,
       'stuck_at_fault': False, 'retention_loss': False, 'aging_effect': 0, 'wire_width': 200, 'input_bit': 1,
       'batch_interval': 250, 'CMOS_technode': 45, 'ADC_precision': 16, 'ADC_setting': 4,
       'ADC_rounding_function': 'floor', 'device_roadmap': 'HP', 'temperature': 300, 'hardware_estimation': HW}

torch.manual_seed(0)
B, T, eps = 2, 300, 0.01
net = BCPNNNetwork(n_input=12, n_hc=2, n_mc=2, input_shape=(6, 2), sim_params=sim, batch_size=B, dt=1.0, eps=eps)
X, Y, C = net.layers['X'], net.layers['Y'], net.connections[('X', 'Y')]
print('layers:', {k: tuple(v.shape) for k, v in net.layers.items()}, '| rule:', type(C.update_rule).__name__,
      '| conn arrays:', list(C.transform.mem_arrays))

# Bernoulli input spikes, 10% rate, shape (time, batch, *input_shape) as Network.run expects
spikes = (torch.rand(T, B, 6, 2) < 0.10)
net.run(inputs={'X': spikes}, time=T)

# golden: recompute Zi/Pi from the same input spikes, Zj/Pj from the spikes Y actually emitted is not
# recorded by run(), so check the rule against the traces it had access to.
pi = X.p.view(B, -1).mean(0); pj = Y.p.view(B, -1).mean(0); pij = C.pij.mean(0)
w_ref = torch.log(pij + eps**2) - torch.log(pi + eps).unsqueeze(1) - torch.log(pj + eps).unsqueeze(0)
b_ref = torch.log(pj + eps)
kz, kp = float(X.kzi), float(X.kp); Zg = torch.zeros(B, 6, 2); Pg = torch.zeros(B, 6, 2)
for t in range(T):
    Pg = Pg * (1 - kp) + Zg * kp; Zg = Zg * (1 - kz) + spikes[t].float() * kz
print(f'ran {T} steps | X: Z err {(X.x - Zg).abs().max():.1e}, P err {(X.p - Pg).abs().max():.4f}')
print(f'Y spikes/step {Y.s.float().sum().item():.0f} | a sums per HC {[round(v,2) for v in Y.a.sum(-1)[0].tolist()]}')
print(f'w {tuple(C.w.shape)} finite {bool(torch.isfinite(C.w).all())} | max|w-ref| {(C.w - w_ref).abs().max():.4f} '
      f'| max|bias-ref| {(Y.bias.flatten() - b_ref).abs().max():.4f} | w range [{C.w.min():.2f}, {C.w.max():.2f}]')
net.reset_state_variables(); print('reset ok | pij sum', float(C.pij.sum()), '| X.p sum', float(X.p.sum()))
