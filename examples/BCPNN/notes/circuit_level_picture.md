# Circuit-level picture: memristor P-trace pipeline as simulated

_Written 2026-11-10 by Claude from four source readers (SIMBRAIN peripherals, device model, our mapping/engine kernels, SIMBRAIN and Wang 2023 papers), then adversarially checked against the code; the checker's corrections are applied. Line references are to the files as of 2026-11-10. Measured results in section 5a are from progress_log.md._

Scope: BCPNN P traces (Pi, Pj, Pji) stored one memristor per trace cell. Settings: SIMBRAIN `device_structure='trace'`, device `'ideal'`, 45 nm HP CMOS, `wire_width=200`. This note describes only what the code and the two papers state. Where something is not modelled, or exists only as energy/area bookkeeping, the note says so. Paper statements ([S], [W]) were not re-checked against the code in this pass.

**Path abbreviations** (relative to the repo root unless absolute):

| Tag | File |
|---|---|
| M | `examples/BCPNN/simbrain/mapping.py` |
| MA | `examples/BCPNN/simbrain/memarray.py` |
| PC | `simbrain/periphcircuit.py` |
| PP | `simbrain/periphpower.py` |
| PA | `simbrain/peripharea.py` |
| MP | `simbrain/mempower.py` |
| E | `examples/BCPNN/Model_to_BCPNN_engine.py` |
| CU | `~/PhD_Work/BCPNNSim2.1_memristor/libsrc_cu/src/Prj_cu.cu` |
| CPP | `.../libsrc_cu/src/Prj.cpp` |
| B | `~/PhD_Work/BCPNNSim2.1_memristor/python_api/bindings.cpp` |

PC, PP, PA, MP, memarea.py and memarray.py are byte-identical to their copies under `examples/BCPNN/simbrain/` (or `simbrain/`).

**Paper abbreviations:**
- [S] Xu et al., SIMBRAIN, Neurocomputing 665 (2026). Page numbers are printed pages.
- [W] Wang et al., TBioCAS 17(5) 2023. "pN" means PDF page N.

---

## 1. Summary

There are two code paths:
- **Path S:** SIMBRAIN Python, `BCPNNMapping` (class M:449; write/read M:762-850). Default pulse mode there is 'pair' (M:450).
- **Path X:** the CUDA engine copy used for the reported runs. It runs only with `--memristor_inference 1` (E:40, E:171-183). Default pulse mode in E is 'gap' (E:43).

In Path X, SIMBRAIN's MemristorArray, DAC, ADC, Power and Area code is not executed (E imports no SIMBRAIN module). The engine re-implements the device law and the read/ADC path in kernels (CU:430-549). SIMBRAIN contributes the device and wire constants and the calibration npz (E:192-205). In gap mode the npz (band labels, vpos/vneg) is loaded but has no functional effect: `vpos[b]`/`vneg[b]` are used only when `d.gap == 0` (CU:484, 499, 518).

What each block is, per gated step and per cell:
- **Controller:** digital.
- **Write "DAC":** an ideal, continuous float32 voltage. Write voltages are not quantised in either path.
- **Device:** VTEAM with linear windows (P = 1), the only analog element that is physically modelled.
- **Read:** ohmic, with a lumped series wire resistance.
- **ADC:** an ideal unipolar floor quantiser, with a per-cell full scale tied to a G_off cell.
- **BCPNN:** sees only the ADC-quantised value, floored at eps.

SIMBRAIN's peripheral models (switch matrix, SAR ADC, shift-add, DFF) are NeuroSim-derived energy and area estimators. They are off by default in the BCPNN scripts (`hardware_estimation=False`, E:39) and never change signal values. The 'ideal' device has every non-ideality magnitude at zero.

---

## 2. Signal chain for one trace cell, per gated step (Path X, CU:479-525)

**When a step happens.** A gated step runs only if `prntaupdt = prn*taupdt > 0` (CU:535, 543-547).
- Unsupervised: one step per pattern, on the last of `nloop = 3` (E:82, E:257).
- Supervised: `prns = [0.0, ...]`, so ihprj gets prn = 0 and no pulses (E:276).
- Inference: `net.updstate()` (E:244, E:289) -> B:404-405 -> `Netw::updstate(float prn = 0, ...)` (Netw.h:87; Netw.cpp:256-260), so no pulses.
- Order: Z is updated before the P kernels, then P runs in the order Pj, Pi, Pji (CU:541-547).

| # | Stage | What the code does | Class |
|---|---|---|---|
| 1 | Drive | Pj: `Zj[nj]`. Pi: `Zi[k]`. Pji: `float drive = Zi[(long)hj * denNi + ni] * Zj[nj];` (CU:516). This is a digital multiply; there is no analog multiplier. | [digital] |
| 2 | Last reading | `mr[c]` holds the previous ADC reconstruction (CU:489, 504, 523). It is initialised to `memread(0, c, hc)` (CPP:786-788, 803-814). The controller never sees the analog `x`. | [digital] |
| 3 | Step target (gap mode) | `step = (drive - xread) * d.taupdt`, with `taupdt = 0.001` hard-coded (E:205; CU:450) | [digital] |
| 4a | VTEAM inversion, SET | `pos = step / (d.dtdr * d.koff * powf(fmaxf(1.0f - x, 1e-12f), d.Poff))`; `v = fminf(d.voff * (1.0f + powf(fmaxf(pos, 0.0f), 1.0f / d.aoff)), 1.5f * d.voff)` (CU:448-460) | [digital] |
| 4b | VTEAM inversion, RESET | The symmetric expression with `fabsf(d.kon)` and `Pon`, clamped at `1.5f * d.von` (−1.5 V) (CU:455-457). If `step == 0`, or `x == 0` on the negative side, `v = 0` (no pulse). | [digital] |
| 4' | Pair mode (alternative) | `(drive > mr[c]) ? vpos[b] : vneg[b]` (CU:518). This is a per-band float32 table from the calibration npz (CPP:782-785). A tie applies vneg. | [digital] |
| 5 | Write DAC | `v` is applied as an exact float32 value. There is no resolution, settling, write-line IR drop or pulse shape. In Path S, `DAC_write` is energy only and returns None (PC:109-131). | [not modelled] |
| 6 | Device update | `if (v >= d.voff) x += d.dtdr * d.koff * powf(v / d.voff - 1.0f, d.aoff) * powf(1.0f - x, d.Poff);` plus the `von` branch, then clamp to [0,1] (CU:462-469). This is the same law as MA:263-271. One pulse of width `dtdr = delta_t*duty_ratio = 1e-3 s` (E:202). | [analog, modelled] |
| 7 | Conductance | `G = Gon + x*(Goff-Gon)` (CU:435; MA:308). Linear I-V. | [analog, modelled] |
| 8 | Read voltage | Path S: a unit mask (`mem_v_read.fill_(1)`, M:831) passes through `DAC_read` unchanged (PC:71-75; M:836); `v_read = 0.1 V` is multiplied in inside `memristor_read` (MA:327, 349). Path X: v_read cancels and is not used. | [analog, modelled] (ideal source) |
| 9 | Series wire | `Geff = 1.0 / (1.0 / G + Rw)`, with `Rw = wuc*(cell+1) + wur` (CU:433-437; MA:104-105, 334-336) | [analog, modelled] |
| 10 | Cell current | Path S: `self.mem_i = torch.matmul(mem_v * v_read, mem_c)` (MA:349). For a (1,N) row this is a per-cell multiply. | [analog, modelled] |
| 11 | ADC full scale | `ADC_mem_c = 1 / (1 / self.Goff + total_wire_resistance)` (M:839); `mem_i_max = high_cut_ratio * torch.sum(self.read_v_amp*mem_c, dim=0)` (PC:229). Per cell: I_FS = high_cut × v_read × 1/(1/Goff + Rw). Engine: `Gmax = hc / (1/Goff + Rw)` (CU:438). | [converter, modelled] |
| 12 | Codes | `mem_i_step = (mem_i_max - mem_i_min) / (2**self.ADC_precision)`, with `mem_i_min = 0` (PC:230-231). Then `clamp(floor(index), 0, 2**ADC_precision-1)` (PC:235-236). Engine: `idx = floor(Geff/step)`, clamped to [0, `levels-1`] (CU:439-443). | [converter, modelled] |
| 13 | Back to trace units | `(idx*step - Gon)/(Goff-Gon)`, cast to float32 (CU:444). Path S: `(mem_i/self.v_read - self.Gon) * self.trans_ratio` (M:846; trans_ratio M:88). Wire resistance is not compensated. At idx = 0 the result is ≈ -0.0101. | [digital] |
| 14 | eps floor | `mr[c] = r;` and `P[c] = fmaxf(r, eps);` with eps = 1e-8 (CU:489-490, 504-505, 523-524; E:85, E:219). The golden path has no such floor (CU:362-370, 380-402). | [digital] |
| 15 | Hand to engine | Weights and bias are computed from the floored, quantised P. `mx[c]` is not passed to BCPNN; it is readable only via `get_mem_x` (B:249 -> CPP:816-823); `mr` via `get_mem_read` (B:250). | [digital] |

**Not in this chain:**
- Peripheral energy and area: [bookkeeping only], in Path S only, and only with `hardware_estimation` (PC:42-48, 190-201).
- c2c, d2d, SAF, retention and aging: absent in Path X. The flags exist only in an unused `sim_params` (E:24-28, E:54-61).

---

## 3. Block diagram (one per trace array; arrays are independent)

```
        +--------------------------- digital controller (float32) -----------------------------+
        | drive = Z (Pi,Pj) or Zi*Zj (Pji)    mr[c] = last ADC reading                         |
        | gap:  step=(drive-mr)*taupdt -> VTEAM inverse -> v in [-1.5,+1.5] V (or 0)           |
        | pair: drive>mr ? vpos[band] : vneg                                                   |
        +-------------------------------------------+------------------------------------------+
                                                    | v (exact float; DAC NOT modelled)
                                                    v
   [write "DAC"] --> [ cell c: VTEAM x, G=Gon+x(Goff-Gon) ]   one device per cell, no selector/1T1R
                                     |
            v_read=0.1 V (ideal) --> | --> Rw(c) series (lumped) --> I = v_read*Geff
                                                    |
                                                    v
   [ADC, 1 per cell, ideal floor, 0..I_FS(c)=hc*v_read/(1/Goff+Rw(c)), 2^N codes]
                                                    |
                                                    v
                    x_hat = (code*LSB - Gon)/(Goff-Gon) -> mr[c];  P[c] = max(x_hat, 1e-8) -> BCPNN
```

| Array | Shape | N | hc (full-scale fraction) | Rw at the last cell (derived) |
|---|---|---|---|---|
| Pi | 1 × 47,040 (Hj·denNi = 30·1568) | 47,040 | 1.0 | ≈ 5.9e3 Ω |
| Pj | 1 × 3,000 | 3,000 | 0.2 | ≈ 378 Ω |
| Pji | 1 × 4,704,000 | 4,704,000 | 0.2 | ≈ 5.9e5 Ω |

Sources: sizes M:460 (checked at CPP:771-772; denNi = (nactHi+nsilHi)·Mi, CPP:137-138); hc M:451 and E:200 (E:42 default 0.2). Rw is derived from 0.126 Ω per pitch, compared with 1/Goff = 1e8 Ω.

**'trace' structure.**
- Each trace is its own MemristorArray of shape (1,N) (M:480-482).
- The read contraction runs over a single row, so there is no MAC (M:836-838, MA:349). The engine has no matmul at all.
- In Path S, one DAC_Module and one ADC_Module are shared by all traces. They are sized to the largest trace, (1, 4,704,000) (M:475, 483-486; shape built by the trace branch M:57-61).

**STDP_crossbar / [S] trace crossbar (alternative; paper, not re-checked).** This is a real 2-D passive crossbar written row by row with ½V half-select (consistent with the ½·V row-write term, PP:221). It uses two write cycles per row (+VH for spike, −VL otherwise) and a row-wise read into per-column SAR ADCs ([S] p4, Fig. 3).

---

## 4. What SIMBRAIN's peripheral models contain, and what they do not

### 4a. Modelled, all as energy/area bookkeeping only

**"DAC" = transmission-gate switch matrix + DFFs** (PC:9-151; PP:81-238; PA:111-218)

| Item | Formula / behaviour | Cite |
|---|---|---|
| Read TG sizing | `resTg = (1/Goff)/shape[1]*IR_DROP_TOLERANCE`; `width_tg = R_on(technode)*technode/(resTg*2)` | PP:81-110 |
| Column-write TG sizing | `resTg = (1/Goff)/shape[0]/2` | PP:124-160 |
| Read energy | `(C_tg_drain*3)*v_read^2*activity*shape[0]*read_times`, plus gate cap × vdd², plus DFF energy | PP:184-195 |
| Column-write energy (used for trace) | `C_drain*3*Vpos_max^2*n_pos + C_drain*3*Vneg_min^2*n_neg + gate + DFF`. Only the max and min of mem_v are used, so in gap mode every positive cell is charged at the global max. | PP:198-212 |
| Reset energy | `C_drain*3*min(mem_v)^2*batch*shape[1] + gate + DFF` | PP:228-238 |
| Area | Row mode `(wTg*2)*minCellHeight*shape[0] + Dff_area`; column mode `hTg*2*minCellWidth*shape[1] + Dff_area` | PA:147, 189 |

The switch matrix does not generate the write voltage. No voltage-DAC circuit (R-string or C-DAC) appears in function, energy or area. In [S], "DAC resolution" means input-bit quantisation ([S] p7, p10). In trace mode, `DAC_read` returns `mem_v` unchanged and does no bit-slicing (PC:71-75).

**SAR ADC** (PC:154-259; PP:399-479; PA:362-386)
- Energy, fitted from Cadence at Vread = 0.5 V (PP:406). HP 45 nm:
  `E = ((2.1843*p + 11.931)e-6 + 0.097754*exp(-2.296*log10(mem_r))) * (1+1.3e-3*(T-300)) * (p+1)*1e-9`, with `mem_r = 0.5/I` (PP:409-423, 477-478).
  - The model is quadratic in p ((a·p + b)·(p + 1)) with no upper bound, so it extrapolates above the fitted range.
  - One conversion is charged per element per read.
- Area: `areaUnit = (hN*wN)*(269 + (p-1)*109) + (hP*wP)*(209 + (log2(levelOutput)-1)*73)`; `SarADC_area = areaUnit*shape[1]` (PA:375-380).
  - This means one ADC per column with no multiplexing. In trace mode that is one ADC per memristor of the largest trace.
  - **Defect:** `levelOutput = 2 ^ self.ADC_precision` is XOR in Python (PA:375), so the PMOS term does not scale with p as intended (p = 16 gives levelOutput = 18; p = 2 gives 0 and `math.log2(0)` raises ValueError, PA:378).
- Only the floor function is allowed when hardware estimation is on (PC:190-201).

**Shift-add / DFF**
- Ripple adder built from NANDs; DFF = 12 minimum inverters (PP:482-528; PA:268-359, DFF at PA:334-341).
- In trace mode, shift-add *energy* is not booked: the call is gated to crossbar at PC:247-248. Shift-add *area* is still counted (PA:389-401; M:891-892).
- Functionally, the shift-add sum PC:242-243 still runs in trace mode, over input_bit = 1 (E:31), which is an identity.
- [S] says trace mode bypasses ShiftAdd ([S] p10).

**Decoders, sense amplifiers, TIA, integrators, current mirrors:** not modelled anywhere (grep of simbrain/*.py and examples/BCPNN/simbrain/*.py). Current mirrors appear only in an area comment (PA:384). The word "decoder" does not appear in [S] (paper, not re-checked).

**Array energy (MP).**
- Wire capacitance: 0.2 fF/µm (MP:51-52).
- Trace write: dynamic `sum(mem_v^2*wire_cap_col)` + static `sum(mem_v^2*dr*dt*G_eff)`, with G_eff = 1/(1/(½(G+G_pre)) + Rw) and no half-select term (MP:107-116).
- Read: dynamic `v_read^2*v_sum*wire_cap_row` (MP:86) + static `v_read^2*sum(v·G_eff)*dt*duty` (MP:88-92).
- Array area = `length_row*length_col`, with no access-transistor or selector area (memarea.py:37-42).

### 4b. Technology inputs

| Parameter | Value | Cite |
|---|---|---|
| CMOS node / roadmap | 45 nm, HP | E:36-37 |
| HP/45 | vdd 1.0 V, vth 0.171 V, pn_size_ratio 1.176, R multiplier 1.54 | CMOS_tech_info.json |
| Constants (units of F) | MIN_NMOS_SIZE 2, MAX_TRANSISTOR_HEIGHT 28, IR_DROP_TOLERANCE 0.1 | CMOS_tech_info.json |
| wire_width 200 | AR 2.1, Rho 2.42e-8 Ω·m | wire_tech_info.json |
| Unit wire R | `relax*mem_size*Rho/(AR*w^2*1e-18)` ≈ 0.126 Ω per pitch (derived) | MA:101-102; E:198-199 |
| Temperature | 300 K; affects peripherals only | E:38; PP:477 |

The CMOS values feed only PP and PA. They do not affect the device state or the read current.

### 4c. Device 'ideal' (memristor_device_info.json)

| v_off / v_on | k_off / k_on | α_off/on | P_off/on | G_on | G_off | delta_t × duty | v_read | mem_size | relax |
|---|---|---|---|---|---|---|---|---|---|
| +1 / −1 V | +100 / −100 | 5 / 5 | 1 / 1 | 1e-10 S | 1e-8 S | 0.002 s × 0.5 = 1 ms | 0.1 V | 350 nm | 1.25 |

- All non-ideality magnitudes are 0: sigma_relative, sigma_absolute, Gon/Goff/Pon/Poff_sigma, SAF_lambda, SAF_delta, retention_loss_tau_reciprocal, Aging_on/off. SAF_ratio = 9 and retention_loss_beta = 1 are shape parameters, inert because their magnitudes are 0.
- x = 0 ↔ G_on (low conductance) and x = 1 ↔ G_off (high). The ratio is 100.
- At 1.5·v_off the maximum step is Δx = 3.125e-3·(1−x) per pulse (derived from MA:263-271).

### 4d. Not modelled (functional signal path)

| Item | Status | Cite |
|---|---|---|
| Write-DAC quantisation, INL, settling, pulse shape | absent; mem_v reaches the device exactly | PC:109-131; CU:462-469 |
| Switch-matrix TG R_on in series with the cell | absent; used only to size TGs | PP:81-160 |
| IR drop on write | absent; Rw used only in write energy | MA:263-271, 310-311; CU:462-469 |
| IR drop on read | lumped series R per cell, no nodal solve | MA:334-336; CU:436-437 |
| Sneak paths, half-select disturb | absent (single row, trace mode) | M:480-482; MA:349; MP:107-116 |
| Read disturb | absent; only `assert v_on <= v_read <= v_off` | MA:330-331 |
| Read, thermal or 1/f noise; device temperature | absent | MA:316-394; E:38 |
| Nonlinear I-V, selector, 1T1R | absent | MA:349; memarea.py:37-42 |
| ADC offset, gain error, INL/DNL, noise/ENOB, metastability, conversion time | absent; ideal quantiser | PC:229-239; CU:433-445 |
| ADC multiplexing / time-sharing | absent; one ADC per cell functionally | PC:229; PA:380 |
| `ADC_setting` | stored but unused in trace mode | M:46, M:653 |
| Decoders, SA, TIA, current mirrors | absent | grep; PA:384 |
| Energy/area in Path X | absent | E (imports no SIMBRAIN module) |

---

## 5. Where the two worrying assumptions sit

### 5a. ADC precision (~20 bits at full scale 0.2·Goff)

- **Location:** stages 11-13. Pj and Pji use `hc = 0.2`, so x above ≈ 0.1919 saturates. Pi uses `hc = 1.0` (M:451; E:200).
- **Measured** (engine copy, gap pulses, cheddar, test accuracy; progress_log.md 2026-11-09/10):
  - golden: 89.38 ± 0.29 (4 seeds)
  - 16 bits: 87.70 ± 1.54 (4 seeds, range 0.275); single runs 83.0 to 88.1 across ranges
  - 20 bits: 89.62 ± 0.64 (range 0.275) and 89.33 ± 0.50 (range 0.2), 4 seeds each
  - 32 bits: 89.82 ± 0.68 (4 seeds), difference to golden +0.45 ± 0.37, not significant
  - The knee lies between 16 and 20 bits; at 20 bits and above accuracy equals the golden within the seed spread. Structural-plasticity swaps differ systematically: +3.5% (16 bits), −2.7% (20), −0.7% (32).
- **[S]** uses 16-bit floor SAR, one per column ([S] p10-11). It does not define the full scale or the code-to-trace mapping ([S] A4). (Paper, not re-checked.)

Derived physical scale (ideal device, v_read = 0.1 V, Rw ignored):

| Trace | I_FS | LSB at 16 bit | LSB at 20 bit | LSB in x units at 20 bit |
|---|---|---|---|---|
| Pi (hc 1.0) | 1e-9 A | 15 fA | ≈ 0.95 fA | ≈ 9.6e-7 |
| Pj / Pji (hc 0.2) | 2e-10 A | 3.1 fA | ≈ 0.19 fA | ≈ 1.9e-7 |

Over a 1 ms window, 0.19 fA is about 1.2 electrons (derived: 0.19e-15 × 1e-3 / 1.6e-19). At this G_off and v_read, a 20-bit current-domain conversion is therefore below the charge granularity of a ms-scale integration, before any noise is considered. A scaled device (larger G, higher v_read) moves these numbers but does not change the ratio of 2^20.

**What 20 bits implies physically.** A SAR at 20 bits needs a comparator and capacitor DAC matched to about 1 ppm. A delta-sigma reaches 20+ ENOB only at low sample rates. Figures supplied by the student for the TI ADS1262 (not verified in these notes): 25 ENOB at 2.5 SPS and 15.6 ENOB at 38.4 kSPS. The pipeline needs one conversion per cell per gated step: 4.75 M cells × patterns. The SIMBRAIN energy model is quadratic in p and has no ENOB or noise term (PP:422-423, 477-478), so it cannot show this cost.

### 5b. Computed (gap-dependent) pulse and the DAC it implies

- **Location:** stages 3-5.
- The amplitude is a continuous per-cell value `v = voff*(1 + pos^(1/αoff))`, clamped to ±1.5 V (CU:448-460; M:808-822). The DAC that delivers it is not modelled (stage 5).
- **Sensitivity:** Δx ∝ (v/v_off − 1)^5 (α = 5), so δΔx/Δx ≈ 5·δv/(v − v_off).
- **Example:** 8-bit DAC over 0.5 V (the [1.0, 1.5] V positive range) gives LSB ≈ 1.95 mV. At v − 1 = 0.16 V, the error is 5 × 1.95e-3 / 0.16 ≈ 6% of the step.
- This error is largest for small steps (v near v_off). Whether most P updates sit there is not yet measured (no step-size histogram in these notes).
- The controller also needs a fifth root and a window inversion per cell per step, in digital arithmetic.

### Contrasts (paper columns not re-checked against the PDFs here)

| | This pipeline (Path X) | Wang 2023 [W] | SIMBRAIN STDP trace [S] |
|---|---|---|---|
| Write drive | per-cell computed amplitude (gap) or per-band pair compared against the ADC reading | per-trace pulse whose amplitude sets k and τ; Z traces "drive" P, but the Z → voltage mechanism is not given ([W] p4-5) | two fixed amplitudes, VH = 2.90 V and VL = 2.15 V, decided by spike / no spike ([S] p4, Table 6) |
| Readout | ideal ADC per cell, floor | constant current → C1 → C2 → op-amp → C3 (R2T), analog, no ADC ([W] p6, Fig. 5) | row read 0.5 V, 16-bit floor SAR per column |
| Arithmetic | digital (Zi·Zj, log, eps) | analog adder, Gilbert multiplier, log amp ([W] Fig. 6) | digital after ADC |
| Timing / energy | 1 ms pulse (ideal device); no energy in Path X | 100 ns per update, 180 nm, 51.49 pJ per update (multiplier 27 pJ, memristor 4.91 pJ) ([W] p8, Table III) | NeuroSim-derived estimates; the array dominates ([S] p11) |
| Accuracy reported | network accuracy vs golden | SPICE vs reference RRMSE: Pi 1.82%, Pj 2.35%, Pij 3.29%, wij 2.06% ([W] Table II) | 92.57 ± 0.50% (trace crossbar) ([S] p10) |

**Why closed-form voltages suffice for STDP.** The STDP trace (dC/dt = −C/τ + (1 − C)Σδ) maps directly onto the device: decay is a fixed −VL depression and the jump is a fixed +VH potentiation ([S] Eq. 1, p4). The decision is binary (spike or not) and needs no reading of the stored state. The BCPNN P update (drive − P)·taupdt has a magnitude that depends on the stored P, so the controller must read P (ADC) and compute an amplitude (DAC).

[W]'s chain is analog end to end. It reports 2-3% RRMSE on P traces, but it does not describe how the analog Z becomes a P-memristor write voltage.

---

## 6. Case for memristors: what the ideal-device results do and do not show

**What they show.**
- With an ideal VTEAM device, a gap-computed pulse, and a fine ADC (32 bits), the memristor P store reproduces the golden BCPNN run within the seed spread at 20 bits or more (89.6 ± 0.6% vs 89.4 ± 0.3%, 4 seeds); 16 bits loses 1.7 ± 1.5 points.
- So the device law, invertible in closed form, can act as the storage element of a digital control loop.
- The loop is: read P through the ADC, compute Δ in digital, invert VTEAM, pulse.

**What they do not show.**
- Every non-ideality is off: c2c, d2d, SAF, retention, aging, read noise, write IR drop, DAC quantisation, ADC noise/INL.
- In Path X the energy and area are not estimated at all.
- The arithmetic (drive, Zi·Zj, step, inversion, log, eps) is digital. The memristor is therefore a non-volatile register written through a feedback loop, not an analog compute element.
- The ideal device itself is generic (1 ms pulse, 0.1–10 nS) and does not match a fabricated device.

**Where the precision requirement comes from.** The precision requirement belongs to the BCPNN algorithm, not to the memristor. The student's digital-engine quantisation and eps-sweep results (progress log; not reproduced or verified here) show that the same degradation appears when P is quantised in a purely digital engine. The ~20-bit readout requirement and the fifth-power DAC sensitivity both follow from small P values and small per-step updates (taupdt = 1e-3). They would apply to any storage technology read through a converter.

**What the next experiments decide.**
1. **DAC bits on the gap voltage.** Does a realistic (8-12 bit) write DAC hold accuracy, or does the 5·δv/(v − v_off) error dominate?
2. **Non-idealities on realistic devices** (c2c and d2d first; then retention and SAF). Does device noise exceed the ADC LSB that the algorithm needs?
3. **Bits curve between 16 and 20** (16 and 20 are measured; 17 to 19 would place the knee exactly) for comparison with the achievable ENOB at the required conversion rate.

If the knee stays near 20 bits and realistic DAC or noise budgets break accuracy, the memristor's advantage must come from density or non-volatility, not from in-memory computation. If the requirement can be relaxed (P scaling, larger taupdt, log-domain storage), the converter budget changes accordingly.
