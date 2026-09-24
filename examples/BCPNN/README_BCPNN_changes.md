# BCPNN traces on SIMBRAIN: what was changed and why

This file documents the edits made on 2026-09-23 and 2026-09-24 to put the BCPNN
Z and E traces onto SIMBRAIN memristors. It has two parts:

- **Part A** explains each change in plain language, for a reader who knows the
  science but is not an advanced Python user.
- **Part B** lists the exact line-level changes (unified diffs).

The references used are Wang et al. 2021, *Mapping the BCPNN Learning Rule to a
Memristor Model* (Front. Neurosci. 15:750458), equations 6 to 9 and 16 to 18, and
the SIMBRAIN `STDPMapping` class, which was the template. The analog circuit in
Wang (sample-and-hold, multiplier, log circuits) is **not** recreated; only the
algorithmic mapping is used.

Files touched:

| File | What it is |
|---|---|
| `simbrain/mapping.py` | The SIMBRAIN library. Only the `BCPNNMapping` class changed. `STDPMapping`, `MLPMapping`, `CNNMapping` are untouched. |
| `examples/BCPNN/bindsnet/network/nodes.py` | BindsNET neuron layers (BCPNN copy only). |
| `examples/BCPNN/bindsnet/network/network.py` | BindsNET network driver (BCPNN copy only). Hardware-estimation block only. |
| `examples/BCPNN/bindsnet/learning/learning.py` | BindsNET learning rule (BCPNN copy only). Two call sites. |
| `examples/BCPNN/bindsnet/models/models.py` | The `IncreasingInhibitionNetwork` builder (BCPNN copy only). |

The `examples/STDP/bindsnet` copy is untouched, so the STDP experiments still run
exactly as before.

---

## Part A. The changes, explained

### A.1 The big picture

In SIMBRAIN a trace is not stored as a number in Python. The trace *is* the
internal state `x` of a memristor, between 0 and 1. Writing the trace means
applying a voltage pulse that moves `x`; reading it means measuring conductance
and converting back to `x`.

`STDPMapping` handles one trace per neuron, driven by that neuron's spikes.
BCPNN needs, per neuron, a cascade of three: Z (spike-driven), E (driven by Z)
and P (driven by E). So each layer now owns **three memristor rows**, one per
trace, and the Mapping class knows how to write each of them. P_ij, the
per-synapse trace, is not part of a layer and is not done yet (see A.15).

### A.2 Where the parameters live (and why they are not inside the Mapping)

SIMBRAIN's convention: the Mapping object only knows about the *device* and the
*circuit* (it reads those from `sim_params`). Everything about the *network*,
such as time constants, lives on the BindsNET side and is handed in when needed.
That is how `STDPMapping` gets its `trace_decay`: BindsNET's `Nodes` layer
computes it from `tc_trace` and passes it into `voltage_generation`.

So the hard-coded lines that were in `BCPNNMapping.__init__`

```python
self.kzi = 1/11
self.kzj = 1/11
self.kp = 1/500
self.epsilon = 0.01
```

were removed. Instead the `Nodes` layer in BindsNET has four new time constants,
`tau_zi`, `tau_zj`, `tau_e`, `tau_p` (defaults 11, 11, 50, 500), and converts
them once, in `compute_decays`, with Wang eq. 9:

```python
self.kzi = self.dt / self.tau_zi
self.kzj = self.dt / self.tau_zj
self.ke  = self.dt / self.tau_e
self.kp  = self.dt / self.tau_p
```

These are stored as PyTorch *buffers* (`register_buffer`). A buffer is a number
that belongs to the model and moves to the GPU with it, but is not a trainable
weight. `tc_trace` was already a buffer, so the new ones are stored the same way.

### A.3 One voltage pair per trace: `self.v_pair`

Before, the Mapping kept two loose numbers, `self.vpos` and `self.vneg`. With
several traces you need a pair *per trace*, so they are now kept in a Python
dictionary (a lookup table):

```python
self.v_pair['Zi']  ->  (v_pos, v_neg) for the Z_i trace
self.v_pair['Zj']  ->  (v_pos, v_neg) for the Z_j trace
self.v_pair['Ei'], ['Ej'], ['Pi'], ['Pj']  ->  likewise for E and P
```

Adding a trace later means adding a key; no other code has to change.

### A.4 Calibration is factored into a helper: `_calibrate_z`

`STDPMapping.voltage_generation` did two jobs in one block: build a reference
("golden") trace, then find the voltages that make a memristor follow it. To do
that for two Z traces you would have to copy the block twice. Instead the
search part was moved into a helper method:

```python
def _calibrate_z(self, kz, spike, golden, test_array, plot, name):
    ...
    return v_pos, v_neg
```

and `voltage_generation` now just builds the two golden Z traces (Wang eq. 6)
and calls the helper twice:

```python
self.v_pair['Zi'] = self._calibrate_z(kzi, spike_i, Zi_trace, ...)
self.v_pair['Zj'] = self._calibrate_z(kzj, spike_j, Zj_trace, ...)
```

The leading underscore in `_calibrate_z` is a Python convention meaning
"internal helper, not meant to be called from outside the class".

### A.5 The two closed-form voltages, and the sign fix

Inside `_calibrate_z`, when the device's window exponents are 1
(`P_off == 1`, `P_on == 1`, true for the `'ideal'` device), the voltages are
solved directly instead of searched.

**Positive pulse (spike).** Wang eq. 6 with a spike gives
`Z = Z(1 - kz) + kz = Z + kz(1 - Z)`: Z moves toward 1 by a fraction `kz` of the
remaining distance. VTEAM with exponent 1 moves `x` by
`dt*k_off*(v/v_off - 1)^alpha * (1 - x)`. Equating the two fractions and solving
for `v`:

```python
v_pos = v_off * ((kz / (dt * k_off)) ** (1/alpha_off) + 1)
```

The STDP original has a `1` where `kz` now is, because the STDP trace jumps all
the way to 1 on a spike. So this is the same formula with the jump size changed.

**Negative pulse (silence).** Wang eq. 16 gives `Z = (1 - kz) Z`. VTEAM gives
`x = (1 + dt*k_on*(v/v_on - 1)^alpha) x`. Equating the multipliers, the two `1`s
cancel and the bracket must equal `-kz`:

```python
v_neg = v_on * ((-kz / (dt * k_on)) ** (1/alpha_on) + 1)
```

An earlier draft had `(1 - kz)` here instead of `-kz`. Since `k_on` is negative,
that puts a negative number under a fractional root, and Python stops with
`math domain error`. The paper's equation is right; the slip was in solving it
for `v`. The STDP original writes the same thing as `trace_decay - 1`.

Both closed forms are exact only when the window exponent is 1. For any other
device the existing batched search over 500 candidate `v_neg` values is used,
unchanged, now driven by the passed-in `spike` and `golden` arrays.

### A.6 Writing Z: `mapping_write_z(s, trace)`

This is the STDP write path for the single-row (`'trace'`) structure, with one
addition: it looks up the pair for the trace it is writing.

```python
v_pos, v_neg = self.v_pair[trace]
self.mem_v = self.s.float()
self.mem_v[self.mem_v == 0] = v_neg      # silent -> negative pulse
self.mem_v[self.mem_v == 1] = v_pos      # spike  -> positive pulse
mem_c = self.mem_arrays['Z'].memristor_write(...)
self.x = (mem_c - self.Gon) * self.trans_ratio   # conductance -> trace
```

Every memristor gets one pulse every timestep, exactly as in STDP.

### A.7 Which pair does a layer use? `z_trace`

The input layer holds Z_i, the excitatory layer holds Z_j. The layer is the only
object that knows its own role, so each `Nodes` layer carries a small label,
`z_trace`, which is `'Zi'` or `'Zj'`. `models.py` assigns it: layer X gets
`'Zi'`, layer Y gets `'Zj'`. `Nodes.forward` then calls

```python
self.x = self.transform.mapping_write_z(s=self.s, trace=self.z_trace)
```

### A.8 Writing E: `mapping_write_e(z_prev, trace)`, a fixed pair found by search

Wang eq. 7: `E(t) = E(t-1)(1 - ke) + Z(t-1) ke`, i.e. `E += ke (Z - E)`.

E follows the same pattern as Z and as the original STDP code: **one fixed
`(v_pos, v_neg)` pair per trace, found once at calibration, then plain pulses
every timestep.** Two things are different from Z because E is driven by an
analog value rather than by spikes:

1. **The pulse rule.** Z uses "spike -> positive pulse, no spike -> negative".
   E has no spike, so the rule is: positive pulse where the driving Z is above
   the memristor's current E state, negative pulse elsewhere. That pushes E
   toward Z, which is what `E += ke (Z - E)` does.

2. **How the pair is found.** For Z, `v_pos` has a closed form and only
   `v_neg` is searched. For E neither has a closed form, so `_calibrate_driven`
   searches both: a 100 x 100 grid of candidate amplitudes (from the threshold
   up to 2.5 x the threshold, positive and negative) is simulated in parallel
   as one batch of 10,000 memristors on the `(1, 1)` test array, each driven by
   the golden Z with the pulse rule above, and the pair with the lowest mean
   squared error against the golden E wins. This is the STDP `v_neg` search
   extended to two dimensions; it takes about 0.2 s.

The golden E is built in `voltage_generation` from the golden Z with `ke`
passed in from BindsNET.

**P_i and P_j** use exactly the same machinery one level down the cascade:
golden P from golden E with `kp` (Wang eq. 8), pair found by
`_calibrate_driven`, written by `mapping_write_p` with the rule "positive pulse
where E is above the current P state". Internally `mapping_write_e` and
`mapping_write_p` are thin wrappers around one shared `_write_driven(kind,
drive, trace)` so the code is not duplicated. The third memristor row is
`'P'` in `trace_kinds`.

**Accuracy.** With fixed amplitudes the step size cannot scale with
`ke (Z - E)`, so the E memristor only follows the golden E roughly. That is an
inherent property of the fixed-pair approach, on every device including
`ideal`, and is part of the non-ideality picture rather than a bug. Numbers are
in A.13.

An earlier draft solved a fresh voltage per memristor per step instead. It was
exact on all devices but departs from the SIMBRAIN pattern (calibrate once,
then pulse), so it was replaced by the search-based fixed pair.

**The calibration spike train matters a lot for E and P.** The STDP code
calibrates on a single spike at step 10. For Z that is fine. For E and
especially P, a single spike produces golden traces that are tiny (P peaks
around 1e-4), so the search settles on pulses that barely move the memristor,
and in a real run P never grows. `voltage_generation` therefore has three
optional arguments, `calib_rate`, `calib_points`, `calib_seed`. With
`calib_rate=None` (the default) it reproduces the STDP single-spike setup. With
`calib_rate=0.15, calib_points=400` it calibrates on a random spike train at
that rate, which is close to the Poisson-encoded MNIST input. Measured effect on
`ideal`, evaluated on an independent 15 % train:

| calibration train | E mean err | E peak (golden 0.198) | P mean err | P peak (golden 0.073) |
|---|---|---|---|---|
| single spike (default) | 0.039 | 0.144 | 0.035 | 0.005 |
| 15 %, 150 steps | 0.006 | 0.188 | 0.016 | 0.044 |
| 15 %, 400 steps | 0.006 | 0.186 | 0.010 | 0.058 |
| 30 %, 400 steps | 0.023 | 0.228 | 0.038 | 0.130 |

The rate should match the operating rate; too high overshoots. The BindsNET
side does not pass these arguments yet, so the default is in force until the
spike-handling step wires it up.

### A.9 Order of updates inside one timestep

Wang eq. 7 and 8 use the previous step's value of the driving trace. In
`Nodes.forward` the previous values are still stored when the step begins, so
the cascade is written from the top down, P first, then E, then Z:

```python
self.p = self.transform.mapping_write_p(e_prev=self.e, trace=self.p_trace)  # uses E(t-1)
self.e = self.transform.mapping_write_e(z_prev=self.x, trace=self.e_trace)  # uses Z(t-1)
self.x = self.transform.mapping_write_z(s=self.s, trace=self.z_trace)       # now Z(t)
```

The software path (`--memristor_device trace`, no memristors) does the same two
updates with plain arithmetic, so software and memristor runs compute the same
traces.

### A.10 One array per trace: `mem_arrays`, `DAC_modules`, `ADC_modules`

`STDPMapping` had one `self.mem_array` with one DAC and one ADC. `BCPNNMapping`
now has dictionaries keyed by trace kind:

```python
self.trace_kinds = ['Z', 'E', 'P']
self.mem_arrays  = {'Z': MemristorArray, 'E': MemristorArray, 'P': MemristorArray}
self.DAC_modules = {'Z': DAC_Module,     'E': DAC_Module,     'P': DAC_Module}
self.ADC_modules = {'Z': ADC_Module,     'E': ADC_Module,     'P': ADC_Module}
```

They are `torch.nn.ModuleDict`, which is a dictionary PyTorch knows how to move
to the GPU and save. Everything that used to act on the single array
(`set_batch_size_stdp`, `reset_memristor_variables`, `mem_t_update`,
`update_SAF_mask`, and the hardware-estimation block in `network.py`) now loops
over `trace_kinds`.

### A.11 Reading a trace: `mapping_read(s, kind)`

Same as `STDPMapping.mapping_read_stdp`, with a `kind` argument that selects the
row (`'Z'`, `'E'` or `'P'`). `learning.py` was changed to call
`mapping_read(..., kind='Z')` in the two places it used to call
`mapping_read_stdp`. As before, batch entries with no spike are not read and
return 0.

### A.12 Helper methods copied from STDP

BindsNET calls `reset_memristor_variables()` between images, `mem_t_update()`
between batches and `update_SAF_mask()` when stuck-at faults are on. These were
only defined on `STDPMapping`, so `BCPNNMapping` now has its own copies, looping
over all trace rows.

### A.13 What was verified

Run on the `'ideal'` device in this WSL Python:

- A single memristor written by `mapping_write_z` follows the golden Z trace
  (random 15 % spike train, 200 steps, `kzi = 1/11`, `kzj = 1/7`) to within
  about 1e-7.
- `mapping_read` returns the written Z and E values.
- Z and E write accuracy per device (same random 15 % spike train, 200 steps,
  `kzi = 1/11`, `ke = 1/50`; golden Z peaks at 0.37, golden E at 0.20; for the
  E test the golden Z drives the E row so the E write is measured on its own):

  | device | `P_on` | `P_off` | Z max err | Z mean err | E max err | E mean err | E peak (golden 0.198) |
  |---|---|---|---|---|---|---|---|
  | `ideal` | 1.00 | 1.00 | 1e-7 | 4e-8 | 0.074 | 0.035 | 0.137 |
  | `ferro` | 1.80 | 0.90 | 0.20 | 0.08 | 0.114 | 0.051 | 0.108 |
  | `MF` | 0.94 | 0.72 | 0.04 | 0.015 | 0.079 | 0.042 | 0.130 |
  | `CMS` | 0.15 | 0.51 | 0.73 | 0.37 | 0.085 | 0.048 | 0.125 |

  Z is exact on `ideal` and degrades with the window exponent; on `CMS` the
  fixed-pair Z is unusable as it stands. E is approximate everywhere by
  construction (fixed pulse sizes), following the shape of the golden E at
  roughly 60 to 70 % of its amplitude.
- P write with the default single-spike calibration, golden E driving the P row,
  400-step 15 % train: P mean error about 0.034, memristor peak 0.005 against a
  golden peak 0.073, on `ideal`, `MF` and `CMS` alike. This is the calibration
  problem described in A.8; with a 15 % calibration train the peak is 0.058.
- A full `IncreasingInhibitionNetwork` (784 to 100 neurons, batch 4, 50 steps)
  runs end to end: both layers hold Z, E and P on separate rows, all six pairs
  are calibrated, the learning rule reads Z from the memristor, weights change,
  and `reset_state_variables` clears all rows.

Not verified here: `hardware_estimation=True` (no run).

### A.14 Known limitations and open items

- The Z closed-form `v_pos` assumes `P_off = 1`. On devices with `P_off != 1`
  (`ferro`, `MF`, `CMS`) the positive Z step is approximate; `v_neg` falls back
  to the batched search when `P_on != 1`. See the accuracy table in A.13.
- `mapping_read`, like the STDP original, fails for `batch_size == 1` because of
  a `squeeze` on the spike-sum tensor. Training uses 50.
- `voltage_generation` calibrates both `'Zi'` and `'Zj'` pairs in every layer,
  although each layer only uses one. Harmless, calibration runs once.
- P_i, P_j (per neuron, driven by E) can reuse the E scheme (fixed pair by
  search, pulse rule "positive if E > P") once `kp` is plumbed. P_ij is connection-shaped (N_i x N_j) and driven by `Z_i * Z_j`, so
  it needs a Mapping owned by the `Connection`, not by a layer.

---

## Part B. Exact line changes

Unified diffs. Lines starting with `-` were removed, `+` were added. For
`mapping.py` the diff is against the last git commit (`git diff`). For the
BindsNET files the diff is against the untouched STDP copy in
`examples/STDP/bindsnet/`, since the BCPNN copy is not yet in git.

### B.1 `simbrain/mapping.py`

```diff
@@ -133,5 +133,5 @@ class STDPMapping(Mapping):
             shape=shape
         )
-	#Submodules
+        #Submodules
         self.mem_array = MemristorArray(sim_params=sim_params, shape=self.shape,
                                         memristor_info_dict=self.memristor_info_dict)
@@ -216,5 +216,5 @@ class STDPMapping(Mapping):
             if ori_trace[i + 1] > 1:
                 ori_trace[i + 1] = 1
-			
+
         # Memristor-based Trace
         # Do not count in non-idealities
@@ -471,5 +471,6 @@ class STDPMapping(Mapping):
 
 
-'''        
+
+'''--------------BCPNN mapping class ------------'''       
 class BCPNNMapping(Mapping):
     # language=rst
@@ -489,7 +490,5 @@ class BCPNNMapping(Mapping):
         :param sim_params: Memristor device to be used in learning.
         :param shape: The dimensionality of the memristor array.
-        :param vneg: Negative voltage applied to the memristor.
-        :param vpos: Positive voltage applied to the memristor.
-        :param trace_decay: The original trace drop per time step for each of the BCPNN traces
+        Each layer holds one memristor row per trace kind ('Z', 'E', 'P'); see ``self.mem_arrays``.
         """
         super().__init__(
@@ -498,10 +497,19 @@ class BCPNNMapping(Mapping):
         )
 
-        self.mem_array = MemristorArray(sim_params=sim_params, shape=self.shape,
-                                        memristor_info_dict=self.memristor_info_dict)
-        self.DAC_module = DAC_Module(sim_params=sim_params, shape=self.shape,
-                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
-        self.ADC_module = ADC_Module(sim_params=sim_params, shape=self.shape,
-                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
+        # One memristor array (plus its DAC and ADC) per trace kind held by this layer:
+        #   'Z' - spike-driven Z trace, written with a calibrated (v_pos, v_neg) pair
+        #   'E' - Z-driven E trace, written with a calibrated (v_pos, v_neg) pair (positive pulse if Z > E)
+        #   'P' - E-driven P trace, written with a calibrated (v_pos, v_neg) pair (positive pulse if E > P)
+        self.trace_kinds = ['Z', 'E', 'P']
+        self.mem_arrays = torch.nn.ModuleDict()
+        self.DAC_modules = torch.nn.ModuleDict()
+        self.ADC_modules = torch.nn.ModuleDict()
+        for kind in self.trace_kinds:
+            self.mem_arrays[kind] = MemristorArray(sim_params=sim_params, shape=self.shape,
+                                                   memristor_info_dict=self.memristor_info_dict)
+            self.DAC_modules[kind] = DAC_Module(sim_params=sim_params, shape=self.shape,
+                                                CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
+            self.ADC_modules[kind] = ADC_Module(sim_params=sim_params, shape=self.shape,
+                                                CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
         if 'clipping' in sim_params.keys():
             self.clipping = Clipping(sim_params=sim_params, shape=self.shape, memristor_info_dict=self.memristor_info_dict)
@@ -517,12 +525,7 @@ class BCPNNMapping(Mapping):
         self.register_buffer("x", torch.Tensor())
         self.register_buffer("s", torch.Tensor())
-        self.vneg = 0
-        self.vpos = 0
-        self.trace_decay = 0
-        #BCPNN parameters
-        self.kzi = 1/11
-	self.kzj = 1/11
-	self.kp = 1/500
-	self.epsilon = 0.01
+        self.register_buffer("e", torch.Tensor())
+        self.register_buffer("p", torch.Tensor())
+        self.v_pair = {}  # trace name ('Zi','Zj','Ei','Ej','Pi','Pj') -> (v_pos, v_neg), filled by voltage_generation
 
     def set_batch_size_stdp(self, batch_size, learning) -> None:
@@ -536,10 +539,13 @@ class BCPNNMapping(Mapping):
         self.learning = learning
         self.set_batch_size(batch_size)
-        self.mem_array.set_batch_size(batch_size=self.batch_size)
-        self.DAC_module.set_batch_size(batch_size=batch_size)
-        self.ADC_module.set_batch_size(batch_size=batch_size)
+        for kind in self.trace_kinds:
+            self.mem_arrays[kind].set_batch_size(batch_size=self.batch_size)
+            self.DAC_modules[kind].set_batch_size(batch_size=batch_size)
+            self.ADC_modules[kind].set_batch_size(batch_size=batch_size)
 
         self.mem_v_read = torch.zeros(1, batch_size, 1, self.shape[0], device=self.mem_v_read.device)
         self.x = torch.zeros(batch_size, *self.shape, device=self.x.device)
+        self.e = torch.zeros(batch_size, *self.shape, device=self.e.device)
+        self.p = torch.zeros(batch_size, *self.shape, device=self.p.device)
         self.s = torch.zeros(batch_size, *self.shape, device=self.s.device)
 
@@ -553,54 +559,58 @@ class BCPNNMapping(Mapping):
             self.mem_wr_t.fill_(torch.min(self.mem_wr_t_batch_update[:]))
             
-        self.mem_array.mem_t = self.mem_t
-        self.mem_array.mem_wr_t = self.mem_wr_t
+        for kind in self.trace_kinds:
+            self.mem_arrays[kind].mem_t = self.mem_t.clone()
+            self.mem_arrays[kind].mem_wr_t = self.mem_wr_t.clone()
         
         
         
-    def voltage_generation(self, trace_decay, plot) -> None:
+    def voltage_generation(self, kzi, kzj, ke, kp, plot, calib_rate=None, calib_points=150, calib_seed=0) -> None:
         # language=rst
         """
-        Sets mini-batch size. Called when memristor is used to mapping trace-STDP.
-    
-        :param trace_decay: The original trace drop per time step.
+        Calibrates one (v_pos, v_neg) write-voltage pair per BCPNN trace. Called from Nodes.compute_decays.
+        Pairs are stored in ``self.v_pair`` keyed by trace name ('Zi', 'Zj', 'Ei', 'Ej', 'Pi', 'Pj').
+
+        :param kzi: Z_i update constant, kzi = dt / tau_zi (Wang et al. 2021, Eq. 9). Supplied by the Nodes layer.
+        :param kzj: Z_j update constant, kzj = dt / tau_zj.
+        :param ke: E update constant, ke = dt / tau_e.
+        :param kp: P update constant, kp = dt / tau_p (times the learning-rate kappa, if used).
         :param plot: A boolean flag to determine whether to plot the results.
+        :param calib_rate: Spike probability per step of the calibration spike trains. ``None`` (default)
+            reproduces the STDPMapping setup: one spike at step 10 (pre) and step 20 (post). For the
+            analog-driven E and P traces a train near the operating rate gives a far better fit, because
+            the golden E and P from a single spike are too small for the search to see.
+        :param calib_points: Length of the calibration spike trains.
+        :param calib_seed: Seed for the random calibration trains, so calibration is reproducible.
         """
         # Simulation Setup
-        points = 150
-        spike_i = torch.zeros(points)
-        spike_j = torch.zeros(points)
+        points = calib_points
+        if calib_rate is None:
+            spike_i = torch.zeros(points)
+            spike_j = torch.zeros(points)
+            spike_i[10] = 1
+            spike_j[20] = 1
+        else:
+            g = torch.Generator().manual_seed(calib_seed)
+            spike_i = (torch.rand(points, generator=g) < calib_rate).float()
+            spike_j = (torch.rand(points, generator=g) < calib_rate).float()
         Zi_trace = torch.zeros(points)
         Zj_trace = torch.zeros(points)
+        Ei_trace = torch.zeros(points)
+        Ej_trace = torch.zeros(points)
         Pi_trace = torch.zeros(points)
         Pj_trace = torch.zeros(points)
-        Pij_trace = torch.zeros(points)
-        mem_x = torch.zeros(points)
-        
-        
-        
 
-        # BCPNN Setup - generate from MNIST dataset
-        spike_i[10] = 1
-        spike_j[20] = 1
+        # Golden BCPNN Z, E and P traces (Wang et al. 2021, Eq. 6, 7, 8)
+        for i in range(points - 1):
+            Pi_trace[i + 1] = Pi_trace[i] * (1 - kp) + Ei_trace[i] * kp
+            Pj_trace[i + 1] = Pj_trace[i] * (1 - kp) + Ej_trace[i] * kp
+            Ei_trace[i + 1] = Ei_trace[i] * (1 - ke) + Zi_trace[i] * ke
+            Ej_trace[i + 1] = Ej_trace[i] * (1 - ke) + Zj_trace[i] * ke
+            Zi_trace[i + 1] = Zi_trace[i] * (1 - kzi) + spike_i[i] * kzi
+            Zj_trace[i + 1] = Zj_trace[i] * (1 - kzj) + spike_j[i] * kzj
 
-        # Original Trace
-        for i in range(len(spike) - 1):
-            ori_trace[i + 1] = ori_trace[i] * trace_decay + spike[i]
-
-            if ori_trace[i + 1] > 1:
-                ori_trace[i + 1] = 1
+        # TODO (P_ij): connection-shaped (N_i x N_j), driven by Zi*Zj; belongs to a Mapping owned
+        # by the Connection, not by this layer.
 
-
-	# Original Simplified BCPNN traces
-	for i in range(len(spike) -1):
-	    Zi_trace[i + 1] = Zi_trace[i] * (1-kzi) + spike_i[i]*kzi
-	    Zj_trace[i + 1] = Zj_trace[i] * (1-kzj) + spike_j[i]*kzj
-	    Pi_trace[i+1] = (1-kp)*Pi_trace[i] + Zi_trace[i]*kp
-	    Pj_trace[i+1] = (1-kp)*Pj_trace[i] + Zj_trace[i]*kp
-	    Pij_trace[i+1] = (1-kp)*Pij_trace[i] + Zi_trace[i]*Zj_trace[i]*kp
-
-	
-  
-  
         # Memristor-based Trace
         # Do not count in non-idealities
@@ -623,6 +633,31 @@ class BCPNNMapping(Mapping):
                            'hardware_estimation': self.sim_params['hardware_estimation']}
         test_array = MemristorArray(sim_params=test_sim_params, shape=(1, 1), memristor_info_dict=self.memristor_info_dict)
-        test_array.set_batch_size(batch_size=1)
+
+        self.v_pair = {}
+        self.v_pair['Zi'] = self._calibrate_z(kzi, spike_i, Zi_trace, test_array, plot, 'Zi')
+        self.v_pair['Zj'] = self._calibrate_z(kzj, spike_j, Zj_trace, test_array, plot, 'Zj')
+        self.v_pair['Ei'] = self._calibrate_driven(Zi_trace, Ei_trace, test_array, plot, 'Ei')
+        self.v_pair['Ej'] = self._calibrate_driven(Zj_trace, Ej_trace, test_array, plot, 'Ej')
+        self.v_pair['Pi'] = self._calibrate_driven(Ei_trace, Pi_trace, test_array, plot, 'Pi')
+        self.v_pair['Pj'] = self._calibrate_driven(Ej_trace, Pj_trace, test_array, plot, 'Pj')
+
+
+    def _calibrate_z(self, kz, spike, golden, test_array, plot, name):
+        # language=rst
+        """
+        Finds the (v_pos, v_neg) pair that makes a single memristor follow one spike-driven Z trace.
+        Same procedure as STDPMapping.voltage_generation, factored so it can be run once per trace.
+
+        :param kz: Z update constant for this trace.
+        :param spike: Binary spike train used for calibration, shape [points].
+        :param golden: Reference Z trace for that spike train, shape [points].
+        :param test_array: A (1, 1) ideal MemristorArray used for the search.
+        :param plot: Whether to plot golden vs. memristor trace.
+        :param name: Trace name, used for the plot file.
+        :return: (v_pos, v_neg)
+        """
+        points = len(spike)
         mem_info = self.memristor_info_dict[self.device_name]
+        write_time = 1  # 'trace' structure: one write cycle per timestep
 
         dt = mem_info['delta_t'] * mem_info['duty_ratio']
@@ -630,10 +665,6 @@ class BCPNNMapping(Mapping):
         v_off = mem_info['v_off']
         alpha_off = mem_info['alpha_off']
-        v_pos = v_off * (math.pow(1 / (dt * k_off), 1.0 / alpha_off) + 1)
-
-#        if self.device_structure == 'STDP_crossbar':
-#            write_time = 2
-#        elif self.device_structure == 'trace':
-#             write_time = 1
+        # On a spike Z moves by kz*(1-Z); with P_off == 1 VTEAM moves x by dt*k_off*(v/v_off-1)^alpha*(1-x).
+        v_pos = v_off * (math.pow(kz / (dt * k_off), 1.0 / alpha_off) + 1)
 
         if mem_info['P_on'] == 1:
@@ -641,11 +672,6 @@ class BCPNNMapping(Mapping):
             v_on = mem_info['v_on']
             alpha_on = mem_info['alpha_on']
-            v_neg = v_on * (math.pow((trace_decay - 1) / (dt * k_on), 1.0 / alpha_on) + 1)
-        elif mem_info['P_off'] == 1:
-            k_off = mem_info['k_off']
-            v_off = mem_info['v_off']
-            
-            
-        #Turn into a function      
+            # Silent step: Z -> Z*(1-kz); with P_on == 1 VTEAM gives x -> x*(1 + dt*k_on*(v/v_on-1)^alpha), so the bracket = -kz.
+            v_neg = v_on * (math.pow(-kz / (dt * k_on), 1.0 / alpha_on) + 1)
         else:
             # Enable batch processing for searching the best v_neg
@@ -661,9 +687,9 @@ class BCPNNMapping(Mapping):
                 mem_v = v_tensor if mem_s == 0 else torch.tensor(v_pos).expand(n_test)
 
-                mem_c = test_array.memristor_write(mem_v=mem_v.unsqueeze(1).unsqueeze(2), write_time=1, mem_v_amp=[0,0])
+                mem_c = test_array.memristor_write(mem_v=mem_v.unsqueeze(1).unsqueeze(2), write_time=write_time, mem_v_amp=[0,0])
                 test_x[t + 1] = (mem_c - self.Gon) * self.trans_ratio
 
             # Compare results
-            golden_x = ori_trace.reshape(points, 1, 1, 1).expand(-1, n_test, -1, -1)
+            golden_x = golden.reshape(points, 1, 1, 1).expand(-1, n_test, -1, -1)
             mse = torch.zeros(n_test)
             for i in range(n_test):
@@ -675,4 +701,5 @@ class BCPNNMapping(Mapping):
             blue = (47 / 255, 130 / 255, 189 / 255)
             green = (98 / 255, 149 / 255, 61 / 255)
+            mem_x = torch.zeros(points)
 
             plt.figure(figsize=(13, 4.5))
@@ -693,19 +720,238 @@ class BCPNNMapping(Mapping):
                 mem_x[t+1] = temp_x.squeeze()
 
-            # Plot the original trace and memristor trace
-            plot_x = range(points)
-            # Original
-            ax.plot(ori_trace, color=blue, label='Original Trace')
-            ax.plot(mem_x, color=green, label='Memristor Trace')
+            ax.plot(golden, color=blue, label='Original ' + name)
+            ax.plot(mem_x, color=green, label='Memristor ' + name)
             ax.legend(frameon=False)
 
             plt.tight_layout()
-            plt.savefig('voltage_generation.png', dpi=300, bbox_inches='tight')
+            plt.savefig('voltage_generation_' + name + '.png', dpi=300, bbox_inches='tight')
             plt.show()
 
-        self.vneg = v_neg
-        self.vpos = v_pos    
-        
-'''
+        return v_pos, v_neg
+
+
+    def mapping_write_z(self, s, trace):
+        # language=rst
+        """
+        Writes one timestep of a spike-driven Z trace into the memristor array ('trace' structure only).
+        Spikes are mapped to the (v_pos, v_neg) pair calibrated for ``trace``.
+
+        :param s: Input spikes of this layer.
+        :param trace: Which trace this layer holds, 'Zi' or 'Zj'.
+        :return: Trace values read back from the written conductance.
+        """
+        v_pos, v_neg = self.v_pair[trace]
+        write_time = 1
+        if s.dim() == 4:
+            self.s = s.flatten(2, 3)
+        elif s.dim() == 2:
+            self.s = torch.unsqueeze(s, 1)
+
+        # nn to mem
+        self.mem_v = self.s.float()
+        self.mem_v[self.mem_v == 0] = v_neg
+        self.mem_v[self.mem_v == 1] = v_pos
+
+        self.DAC_modules['Z'].DAC_write(mem_v=self.mem_v, mem_v_amp=[v_pos, v_neg])
+
+        mem_c = self.mem_arrays['Z'].memristor_write(mem_v=self.mem_v, write_time=write_time, mem_v_amp=[v_pos, v_neg])
+
+        # mem to nn
+        self.x = (mem_c - self.Gon) * self.trans_ratio
+
+        if s.dim() == 4:
+            self.x = self.x.reshape(s.size(0), s.size(1), s.size(2), s.size(3))
+        elif s.dim() == 2:
+            self.x = self.x.squeeze()
+
+        return self.x
+
+
+    def _calibrate_driven(self, golden_drive, golden_target, test_array, plot, name):
+        # language=rst
+        """
+        Finds the (v_pos, v_neg) pair that makes a single memristor best follow one analog-driven trace
+        (E driven by Z, or P driven by E), by batched search. Same idea as the v_neg search in
+        STDPMapping.voltage_generation, but over both voltages, because these traces have no saturating
+        jump that fixes v_pos in closed form.
+
+        Pulse rule used both here and in ``_write_driven``: at each step the memristor gets the positive
+        pulse if the driving trace is above its current state, otherwise the negative pulse. With fixed
+        amplitudes the step size cannot scale with k*(drive - state), so the fit is approximate on every
+        device; the search picks the pair with the lowest MSE against the golden target.
+
+        :param golden_drive: Reference trace that drives the target (Z for E, E for P), shape [points].
+        :param golden_target: Reference target trace for that drive, shape [points].
+        :param test_array: A (1, 1) ideal MemristorArray used for the search.
+        :param plot: Whether to plot golden vs. memristor trace.
+        :param name: Trace name, used for the plot file.
+        :return: (v_pos, v_neg)
+        """
+        golden_z, golden_e = golden_drive, golden_target
+        points = len(golden_z)
+        mem_info = self.memristor_info_dict[self.device_name]
+        write_time = 1
+        v_off, v_on = mem_info['v_off'], mem_info['v_on']
+
+        # Candidate grid: n_p positive amplitudes from v_off upward, n_n negative from v_on downward
+        # (in multiples of the threshold), all simulated in parallel as one batch of n_p*n_n memristors.
+        n_p, n_n = 100, 100
+        pos_cands = v_off * (1 + torch.linspace(0, 1.5, n_p))
+        neg_cands = v_on * (1 + torch.linspace(0, 1.5, n_n))
+        vp_grid = pos_cands.repeat_interleave(n_n)   # [n_p*n_n]
+        vn_grid = neg_cands.repeat(n_p)              # [n_p*n_n]
+        n_test = n_p * n_n
+
+        test_array.set_batch_size(batch_size=n_test)
+        test_x = torch.zeros(points, n_test, 1, 1)
+        x = torch.zeros(n_test)
+        for t in range(points - 1):
+            mem_v = torch.where(golden_z[t] > x, vp_grid, vn_grid)
+            mem_c = test_array.memristor_write(mem_v=mem_v.unsqueeze(1).unsqueeze(2), write_time=write_time, mem_v_amp=[0, 0])
+            x = ((mem_c - self.Gon) * self.trans_ratio).reshape(n_test)
+            test_x[t + 1, :, 0, 0] = x
+
+        golden_x = golden_e.reshape(points, 1)
+        mse = ((test_x[:, :, 0, 0] - golden_x) ** 2).mean(dim=0)
+        min_index = int(torch.argmin(mse))
+        v_pos, v_neg = float(vp_grid[min_index]), float(vn_grid[min_index])
+
+        if plot:
+            blue = (47 / 255, 130 / 255, 189 / 255)
+            green = (98 / 255, 149 / 255, 61 / 255)
+            plt.figure(figsize=(13, 4.5))
+            grid = plt.GridSpec(14, 17, wspace=0.5, hspace=0.5)
+            ax = plt.subplot(grid[0:14, 0:17])
+            ax.plot(golden_e, color=blue, label='Original ' + name)
+            ax.plot(test_x[:, min_index, 0, 0], color=green, label='Memristor ' + name)
+            ax.legend(frameon=False)
+            plt.tight_layout()
+            plt.savefig('voltage_generation_' + name + '.png', dpi=300, bbox_inches='tight')
+            plt.show()
+
+        return v_pos, v_neg
+
+
+    def _write_driven(self, kind, drive, trace):
+        # language=rst
+        """
+        Writes one timestep of an analog-driven trace row ('E' driven by Z, 'P' driven by E) with the fixed
+        (v_pos, v_neg) pair calibrated for ``trace`` ('trace' structure only).
+        Pulse rule: positive pulse where the driving value is above the row's current state, negative elsewhere.
+
+        :param kind: Memristor row to write, 'E' or 'P'.
+        :param drive: Driving trace from the previous timestep, same shape as the layer output.
+        :param trace: Trace name whose pair to use, e.g. 'Ei', 'Ej', 'Pi', 'Pj'.
+        :return: Trace values read back from the written conductance, same shape as ``drive``.
+        """
+        v_pos, v_neg = self.v_pair[trace]
+        write_time = 1
+        orig_shape = drive.shape
+        if drive.dim() == 4:
+            d = drive.flatten(2, 3)
+        elif drive.dim() == 2:
+            d = torch.unsqueeze(drive, 1)
+        else:
+            d = drive
+        d = d.float()
+
+        arr = self.mem_arrays[kind]
+        # nn to mem: compare the driving trace with the current state to pick the pulse polarity
+        mem_v = torch.where(d > arr.mem_x, torch.tensor(v_pos, dtype=d.dtype), torch.tensor(v_neg, dtype=d.dtype))
+
+        self.DAC_modules[kind].DAC_write(mem_v=mem_v, mem_v_amp=[v_pos, v_neg])
+        mem_c = arr.memristor_write(mem_v=mem_v, write_time=write_time, mem_v_amp=[v_pos, v_neg])
+
+        # mem to nn
+        out = (mem_c - self.Gon) * self.trans_ratio
+        return out.reshape(orig_shape)
+
+
+    def mapping_write_e(self, z_prev, trace):
+        # language=rst
+        """
+        Writes one timestep of the E trace, E(t) = E(t-1)*(1-ke) + Z(t-1)*ke (Wang et al. 2021, Eq. 7).
+
+        :param z_prev: Z trace from the previous timestep.
+        :param trace: 'Ei' or 'Ej'.
+        """
+        self.e = self._write_driven('E', z_prev, trace)
+        return self.e
+
+
+    def mapping_write_p(self, e_prev, trace):
+        # language=rst
+        """
+        Writes one timestep of the P trace, P(t) = P(t-1)*(1-kp) + E(t-1)*kp (Wang et al. 2021, Eq. 8).
+
+        :param e_prev: E trace from the previous timestep.
+        :param trace: 'Pi' or 'Pj'.
+        """
+        self.p = self._write_driven('P', e_prev, trace)
+        return self.p
+
+
+    def mapping_read(self, s, kind='Z'):
+        # language=rst
+        """
+        Reads one trace row from its memristor array ('trace' structure only). Same as STDPMapping.mapping_read_stdp.
+
+        :param s: Spikes of the other layer; batch entries with no spike are not read.
+        :param kind: Which trace row to read, 'Z', 'E' or 'P'.
+        """
+        if s.dim() == 4:
+            s = s.flatten(2, 3)
+        elif s.dim() == 2:
+            s = torch.unsqueeze(s, 1)
+
+        # Read Voltage generation
+        # For every batch, read is not necessary when there is no spike s
+        s_sum = torch.sum(s, dim=2).squeeze()
+        s_sum = torch.unsqueeze(s_sum, 1)
+
+        self.mem_v_read.zero_()
+        self.mem_v_read[0, s_sum.bool()] = 1
+
+        self.mem_v_read = self.DAC_modules[kind].DAC_read(mem_v=self.mem_v_read, sgn=None)
+
+        mem_i, _ = self.mem_arrays[kind].memristor_read(mem_v=self.mem_v_read, read_time=1)
+        ADC_mem_c = 1 / (1 / self.Goff + self.mem_arrays[kind].total_wire_resistance)
+        mem_i = self.ADC_modules[kind].ADC_read(mem_i_sequence=mem_i, mem_c=ADC_mem_c, high_cut_ratio=1)
+
+        if 'clipping' in self.sim_params.keys():
+            mem_i = self.clipping.clipping_function(mem_i_origin=mem_i)
+
+        # current to trace
+        self.mem_x_read = (mem_i/self.v_read - self.Gon) * self.trans_ratio
+        self.mem_x_read[~s_sum.bool()] = 0
+
+        return self.mem_x_read
+
+
+    def reset_memristor_variables(self) -> None:
+        # language=rst
+        """
+        Resets the memristor array with the LUT reset voltage. Same as STDPMapping.
+        """
+        v_reset = self.memristor_luts[self.device_name]['V_reset']
+        self.mem_v.fill_(v_reset)
+        for kind in self.trace_kinds:
+            self.DAC_modules[kind].DAC_reset(mem_v=self.mem_v)
+            self.mem_arrays[kind].memristor_reset(mem_v=self.mem_v)
+
+
+    def mem_t_update(self) -> None:
+        # language=rst
+        """
+        Updates the timing parameters for the memristor array. Same as STDPMapping.
+        """
+        for kind in self.trace_kinds:
+            self.mem_arrays[kind].mem_t += self.batch_interval * (self.batch_size - 1)
+            self.mem_arrays[kind].mem_wr_t += self.write_batch_interval * (self.batch_size - 1)
+
+    def update_SAF_mask(self) -> None:
+        for kind in self.trace_kinds:
+            self.mem_arrays[kind].update_SAF_mask()
+
 
 class MLPMapping(Mapping):
```

### B.2 `examples/BCPNN/bindsnet/network/nodes.py` (vs. STDP copy)

```diff
@@ -3,5 +3,5 @@
 from operator import mul
 from typing import Iterable, Optional, Union
-from simbrain.mapping import STDPMapping
+from simbrain.mapping import BCPNNMapping
 import torch
 
@@ -21,4 +21,9 @@
         tc_trace: Union[float, torch.Tensor] = 20.0,
         trace_scale: Union[float, torch.Tensor] = 1.0,
+        tau_zi: Union[float, torch.Tensor] = 11.0,
+        tau_zj: Union[float, torch.Tensor] = 11.0,
+        tau_e: Union[float, torch.Tensor] = 50.0,
+        tau_p: Union[float, torch.Tensor] = 500.0,
+        z_trace: str = 'Zi',
         sum_input: bool = False,
         learning: bool = True,
@@ -36,4 +41,9 @@
         :param tc_trace: Time constant of spike trace decay.
         :param trace_scale: Scaling factor for spike trace.
+        :param tau_zi: Time constant of the BCPNN Z_i trace; kzi = dt / tau_zi (Wang et al. 2021, Eq. 9).
+        :param tau_zj: Time constant of the BCPNN Z_j trace; kzj = dt / tau_zj.
+        :param tau_e: Time constant of the BCPNN E trace; ke = dt / tau_e (Wang et al. 2021, Eq. 9).
+        :param tau_p: Time constant of the BCPNN P trace; kp = dt / tau_p.
+        :param z_trace: Which Z trace this layer's memristors hold: 'Zi' (presynaptic) or 'Zj' (postsynaptic).
         :param sum_input: Whether to sum all inputs.
         :param learning: Whether to be in learning or testing.
@@ -82,6 +92,19 @@
                 "trace_decay", torch.empty_like(self.tc_trace)
             )  # Set in compute_decays.
+            self.register_buffer("tau_zi", torch.tensor(tau_zi))  # Time constant of Z_i.
+            self.register_buffer("tau_zj", torch.tensor(tau_zj))  # Time constant of Z_j.
+            self.register_buffer("kzi", torch.empty_like(self.tau_zi))  # Set in compute_decays.
+            self.register_buffer("kzj", torch.empty_like(self.tau_zj))  # Set in compute_decays.
+            self.register_buffer("tau_e", torch.tensor(tau_e))  # Time constant of E.
+            self.register_buffer("ke", torch.empty_like(self.tau_e))  # Set in compute_decays.
+            self.register_buffer("e", torch.Tensor())  # E trace (software path or memristor read-back).
+            self.register_buffer("tau_p", torch.tensor(tau_p))  # Time constant of P.
+            self.register_buffer("kp", torch.empty_like(self.tau_p))  # Set in compute_decays.
+            self.register_buffer("p", torch.Tensor())  # P trace (software path or memristor read-back).
+            self.z_trace = z_trace  # Which Z trace this layer writes: 'Zi' or 'Zj'.
+            self.e_trace = 'Ei' if z_trace == 'Zi' else 'Ej'  # Matching E trace.
+            self.p_trace = 'Pi' if z_trace == 'Zi' else 'Pj'  # Matching P trace.
             if self.device_name != 'trace':
-                self.transform = STDPMapping(sim_params=sim_params, shape=self.shape)
+                self.transform = BCPNNMapping(sim_params=sim_params, shape=self.shape)
    
         if self.sum_input:
@@ -104,14 +127,17 @@
         if self.traces:
             if (self.device_name != 'trace' and self.learning == True):
-                self.x = self.transform.mapping_write_stdp(s=self.s)
+                # Cascade order P, E, Z: each uses the previous step's value of the trace that drives it
+                # (Wang et al. 2021, Eq. 6-8), which is still stored when the step begins.
+                self.p = self.transform.mapping_write_p(e_prev=self.e, trace=self.p_trace)
+                self.e = self.transform.mapping_write_e(z_prev=self.x, trace=self.e_trace)
+                self.x = self.transform.mapping_write_z(s=self.s, trace=self.z_trace)
 
             else:
-                # Decay and set spike traces.
-                self.x *= self.trace_decay
-
-                if self.traces_additive:
-                    self.x += self.trace_scale * self.s.float()
-                else:
-                    self.x.masked_fill_(self.s.bool(), self.trace_scale)
+                # Same cascade order in software.
+                self.p = self.p * (1 - self.kp) + self.e * self.kp
+                self.e = self.e * (1 - self.ke) + self.x * self.ke
+                # Z: BCPNN form, Z(t) = Z(t-1)*(1-kz) + S(t-1)*kz (Wang et al. 2021, Eq. 6).
+                kz = self.kzi if self.z_trace == 'Zi' else self.kzj
+                self.x = self.x * (1 - kz) + self.s.float() * kz
 
         if self.sum_input:
@@ -129,4 +155,6 @@
         if self.traces:
             self.x.zero_()  # Spike traces.
+            self.e.zero_()  # E traces.
+            self.p.zero_()  # P traces.
             if (self.device_name != 'trace' and self.learning == True):
                 self.transform.reset_memristor_variables()
@@ -152,6 +180,11 @@
             )  # Spike trace decay (per timestep).
 
+            self.kzi = self.dt / self.tau_zi  # BCPNN Z-trace update constants.
+            self.kzj = self.dt / self.tau_zj
+            self.ke = self.dt / self.tau_e  # BCPNN E-trace update constant.
+            self.kp = self.dt / self.tau_p  # BCPNN P-trace update constant.
+
             if (self.device_name != 'trace' and self.learning == True):
-                self.transform.voltage_generation(self.trace_decay, plot=False)
+                self.transform.voltage_generation(self.kzi, self.kzj, self.ke, self.kp, plot=False)
 
 
@@ -170,4 +203,6 @@
         if self.traces:
             self.x = torch.zeros(batch_size, *self.shape, device=self.x.device)
+            self.e = torch.zeros(batch_size, *self.shape, device=self.e.device)
+            self.p = torch.zeros(batch_size, *self.shape, device=self.p.device)
             if (self.device_name != 'trace' and self.learning == True and init_batch_sign == True):
                 self.transform.set_batch_size_stdp(batch_size=self.batch_size, learning=self.learning)
@@ -211,4 +246,9 @@
         tc_trace: Union[float, torch.Tensor] = 20.0,
         trace_scale: Union[float, torch.Tensor] = 1,
+        tau_zi: Union[float, torch.Tensor] = 11.0,
+        tau_zj: Union[float, torch.Tensor] = 11.0,
+        tau_e: Union[float, torch.Tensor] = 50.0,
+        tau_p: Union[float, torch.Tensor] = 500.0,
+        z_trace: str = 'Zi',
         sum_input: bool = False,
         sim_params: dict = {},
@@ -235,4 +275,9 @@
             tc_trace=tc_trace,
             trace_scale=trace_scale,
+            tau_zi=tau_zi,
+            tau_zj=tau_zj,
+            tau_e=tau_e,
+            tau_p=tau_p,
+            z_trace=z_trace,
             sum_input=sum_input,
             sim_params=sim_params
@@ -1024,4 +1069,9 @@
         tc_trace: Union[float, torch.Tensor] = 20.0,
         trace_scale: Union[float, torch.Tensor] = 1,
+        tau_zi: Union[float, torch.Tensor] = 11.0,
+        tau_zj: Union[float, torch.Tensor] = 11.0,
+        tau_e: Union[float, torch.Tensor] = 50.0,
+        tau_p: Union[float, torch.Tensor] = 500.0,
+        z_trace: str = 'Zi',
         sum_input: bool = False,
         sim_params: dict = {},
@@ -1066,4 +1116,9 @@
             tc_trace=tc_trace,
             trace_scale=trace_scale,
+            tau_zi=tau_zi,
+            tau_zj=tau_zj,
+            tau_e=tau_e,
+            tau_p=tau_p,
+            z_trace=z_trace,
             sum_input=sum_input,
             sim_params=sim_params
```

### B.3 `examples/BCPNN/bindsnet/network/network.py` (vs. STDP copy)

```diff
@@ -436,18 +436,18 @@
             self.periph_average_power = 0
             for l in self.layers:
-                self.layers[l].transform.mem_array.total_energy_calculation()
-                self.layers[l].transform.DAC_module.DAC_energy_calculation(
-                    mem_t=self.layers[l].transform.mem_array.mem_t)
-                self.layers[l].transform.ADC_module.ADC_energy_calculation(
-                    mem_t=self.layers[l].transform.mem_array.mem_t)
-                self.sim_power = self.layers[l].transform.mem_array.power.sim_power
-                self.sim_DAC_module_power = self.layers[l].transform.DAC_module.DAC_module_power.sim_power
-                self.sim_ADC_module_power = self.layers[l].transform.ADC_module.ADC_module_power.sim_power
-                self.total_energy += self.sim_power['total_energy']
-                self.average_power += self.sim_power['average_power']
-                self.periph_total_energy += self.sim_DAC_module_power['DAC_total_energy'] + self.sim_ADC_module_power[
-                    'ADC_total_energy']
-                self.periph_average_power += self.sim_DAC_module_power['DAC_average_power'] + self.sim_ADC_module_power[
-                    'ADC_average_power']
+                tf = self.layers[l].transform
+                for kind in tf.trace_kinds:  # one memristor row per trace ('Z', 'E', ...)
+                    tf.mem_arrays[kind].total_energy_calculation()
+                    tf.DAC_modules[kind].DAC_energy_calculation(mem_t=tf.mem_arrays[kind].mem_t)
+                    tf.ADC_modules[kind].ADC_energy_calculation(mem_t=tf.mem_arrays[kind].mem_t)
+                    self.sim_power = tf.mem_arrays[kind].power.sim_power
+                    self.sim_DAC_module_power = tf.DAC_modules[kind].DAC_module_power.sim_power
+                    self.sim_ADC_module_power = tf.ADC_modules[kind].ADC_module_power.sim_power
+                    self.total_energy += self.sim_power['total_energy']
+                    self.average_power += self.sim_power['average_power']
+                    self.periph_total_energy += self.sim_DAC_module_power['DAC_total_energy'] + self.sim_ADC_module_power[
+                        'ADC_total_energy']
+                    self.periph_average_power += self.sim_DAC_module_power['DAC_average_power'] + self.sim_ADC_module_power[
+                        'ADC_average_power']
             print("total_energy=", self.total_energy)
             print("average_power=", self.average_power)
```

### B.4 `examples/BCPNN/bindsnet/learning/learning.py` (vs. STDP copy)

```diff
@@ -195,5 +195,5 @@
             if self.target.traces and self.target.learning and self.target.device_name != 'trace':
                 # Memristor Read
-                target_x = self.target.transform.mapping_read_stdp(s=self.source.s)
+                target_x = self.target.transform.mapping_read(s=self.source.s, kind='Z')
                 target_x = target_x.view(batch_size, -1).unsqueeze(1) * self.nu[0]                
 
@@ -213,5 +213,5 @@
             if self.source.traces and self.source.learning and self.source.device_name != 'trace':
                 # Memristor Read
-                source_x = self.source.transform.mapping_read_stdp(s=self.target.s)
+                source_x = self.source.transform.mapping_read(s=self.target.s, kind='Z')
                 source_x = source_x.view(batch_size, -1).unsqueeze(2)               
             else:                
```

### B.5 `examples/BCPNN/bindsnet/models/models.py` (vs. STDP copy)

```diff
@@ -326,4 +326,8 @@
         tc_theta_decay: float = 1e7,
         inpt_shape: Optional[Iterable[int]] = None,
+        tau_zi: float = 11.0,
+        tau_zj: float = 11.0,
+        tau_e: float = 50.0,
+        tau_p: float = 500.0,
     ) -> None:
         # language=rst
@@ -349,4 +353,8 @@
             potential decay.
         :param inpt_shape: The dimensionality of the input layer.
+        :param tau_zi: Time constant of the BCPNN Z_i trace (input layer).
+        :param tau_zj: Time constant of the BCPNN Z_j trace (output layer).
+        :param tau_e: Time constant of the BCPNN E traces (both layers).
+        :param tau_p: Time constant of the BCPNN P traces (both layers).
         """
         super().__init__(dt=dt, sim_params=sim_params, batch_size=batch_size)
@@ -361,5 +369,6 @@
 
         input_layer = Input(
-            n=self.n_input, shape=self.inpt_shape, traces=True, tc_trace=20.0, sim_params=self.sim_params
+            n=self.n_input, shape=self.inpt_shape, traces=True, tc_trace=20.0, sim_params=self.sim_params,
+            tau_zi=tau_zi, tau_zj=tau_zj, tau_e=tau_e, tau_p=tau_p, z_trace='Zi'
         )
         self.add_layer(input_layer, name="X")
@@ -376,5 +385,6 @@
             theta_plus=theta_plus,
             tc_theta_decay=tc_theta_decay,
-            sim_params=self.sim_params
+            sim_params=self.sim_params,
+            tau_zi=tau_zi, tau_zj=tau_zj, tau_e=tau_e, tau_p=tau_p, z_trace='Zj'
         )
         self.add_layer(output_layer, name="Y")
```

