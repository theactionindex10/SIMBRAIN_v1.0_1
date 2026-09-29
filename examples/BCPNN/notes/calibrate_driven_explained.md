# `_calibrate_driven` explained line by line

`simbrain/mapping.py`, class `BCPNNMapping`. Written 2026-09-29. Companion to
`_calibrate_one`, which calibrates the spike-driven Z traces; this method
calibrates the P traces, which are driven by Z rather than by spikes.

Read `_calibrate_one` first. Everything here is either identical to it or is
one of two deliberate differences. Both differences come from the P update
rule and nothing else.

---

## 1. Why P cannot reuse `_calibrate_one`

**Z update** (Wang et al. 2021 eq. 6; Ravichandran et al. eq. 1), per timestep:

```
spike:     Z <- Z + kz * (1 - Z)     move UP by a fixed fraction of the distance to 1
no spike:  Z <- Z - kz * Z           move DOWN by a fixed fraction of the distance to 0
```

Two facts: the *direction* is decided by the spike, and the *size* depends only
on Z itself.

**Memristor with a fixed voltage** (Wang eq. 17 to 18, VTEAM with window
exponent 1):

```
positive v:  x <- x + D * (1 - x)    D = dt * k_off * (v/v_off - 1)^alpha
negative v:  x <- x - (1-E) * x      1-E = -dt * k_on * (v/v_on - 1)^alpha
```

Same shape. So one `v_pos` makes `D = kz` for every spike, one `v_neg` makes
`1-E = kz` for every silence, and the match is exact. That is the closed form
in `_calibrate_one`. It needs exponent 1, hence the `if mem_info['P_on'] == 1`
test there; when the exponent is not 1 that helper falls back to a search
over `v_neg` only.

**P update** (Wang eq. 10; Ravichandran eq. 2):

```
P <- P + kp * (Z - P)                move toward Z by a fixed fraction of the GAP
```

Now the *direction* is "up if Z is above P, down if below": there is no spike
to read it from, it has to come from comparing Z with P. And the *size* is a
fraction of `(Z - P)`, a quantity the memristor knows nothing about; a fixed
voltage always moves it by a fraction of `(1 - x)` or of `x`. So no voltage
reproduces the P step exactly, for any window exponent, and there is nothing
to solve.

Consequences for the code:

| | `_calibrate_one` (Z) | `_calibrate_driven` (P) |
|---|---|---|
| what decides positive vs negative pulse | the spike, same for all candidates | `Z > state`, per candidate |
| `v_pos` | closed form | searched |
| `v_neg` | closed form if exponent is 1, else searched | searched |
| `P_on == 1` test | yes, chooses closed form vs search | no: there is no closed form to choose |
| candidates | 500 values of `v_neg` (1-D) | 100 x 100 pairs of (`v_pos`, `v_neg`) (2-D) |
| result | exact on the ideal device | best fit, approximate everywhere |

Wang's own circuit avoids the approximation by turning the sampled Z into the
*input voltage* of the P memristor each step (his Fig. 5 and 7). We chose not to
reproduce that circuit and to keep SIMBRAIN's calibrate-once-then-pulse
pattern, which is exactly why P gets a comparison rule and a search.

---

## 2. The method, line by line

Signature:

```python
def _calibrate_driven(self, drive, trace, test_array, plot=False, name=''):
```

- `drive`: the golden Z (150 values). It is what P is compared against.
- `trace`: the golden P (150 values). It is what the memristor should follow.
- `test_array`: the ideal (1,1) `MemristorArray` built once in
  `voltage_generation` and shared by all four calibrations.
- `plot`, `name`: as in `_calibrate_one`.

`kp` is *not* a parameter: it was used when the golden P was built in
`voltage_generation`, and the search only compares against that finished
curve. (`_calibrate_one` does take `kz` because its closed forms need the
number itself.)

### Setup

```python
points = len(trace)                                    # number of timesteps, 150
mem_info = self.memristor_info_dict[self.device_name]  # the device's constants (JSON)
write_time = 1                                         # one write cycle per step ('trace' structure)
```

### Candidate voltages

```python
v_off, v_on = mem_info['v_off'], mem_info['v_on']
n_p, n_n = 100, 100
pos_cands = v_off * (1 + torch.linspace(0, 1.5, n_p))
neg_cands = v_on  * (1 + torch.linspace(0, 1.5, n_n))
```

A memristor does not move unless the pulse is above its threshold, `v_off`
for positive pulses and `v_on` (negative) for negative ones. So every
candidate is "threshold x factor", factor at least 1.

- `n_p`, `n_n`: how many positive and how many negative candidates. Plain
  counts, 100 each. Named so the grid maths below reads `n_p * n_n`.
- `torch.linspace(0, 1.5, 100)`: 100 evenly spaced numbers from 0.0 to 1.5.
  `1 + ...` makes them 1.0 to 2.5. Times `v_off` gives, for the ideal device,
  100 voltages from 1.0 V to 2.5 V. `neg_cands` is the same with `v_on`:
  -1.0 V to -2.5 V.

`_calibrate_one` builds its equivalent list as
`torch.arange(v_on, v_on - 0.01*500, -0.01)`, 500 negative values in 0.01 V
steps. Same idea; multiples of threshold are used here so the range scales to
devices with very different thresholds (CMS: 0.2 V).

### Every pair as one flat list

```python
vp_grid = pos_cands.repeat_interleave(n_n)   # [10000]
vn_grid = neg_cands.repeat(n_p)              # [10000]
n_test  = n_p * n_n                          # 10000
```

`_calibrate_one` searches one voltage, so its candidates are one list and the
batch is 500 memristors. Here every *pair* must be tried: 100 x 100 = 10,000.
A batch is one-dimensional, so the pairs are laid out as two vectors of
length 10,000 that line up: entry `k` of `vp_grid` with entry `k` of
`vn_grid` is pair number `k`.

Miniature, 3 positive x 2 negative:

```
pos_cands                             [1.0, 1.5, 2.0]
neg_cands                             [-1.0, -2.0]
vp_grid = pos_cands.repeat_interleave(2)   [1.0, 1.0, 1.5, 1.5, 2.0, 2.0]   each element stretched in place
vn_grid = neg_cands.repeat(3)              [-1.0,-2.0,-1.0,-2.0,-1.0,-2.0]  whole list tiled
pair k = (vp_grid[k], vn_grid[k]):    (1.0,-1.0) (1.0,-2.0) (1.5,-1.0) (1.5,-2.0) (2.0,-1.0) (2.0,-2.0)
```

That is all six combinations, once each. With 100 and 100 it is all 10,000.
`n_test` is the resulting batch size, playing the role of `n_test = 500` in
`_calibrate_one`.

### The simulation: all 10,000 memristors in parallel

```python
test_array.set_batch_size(batch_size=n_test)   # 10,000 independent memristors
test_x = torch.zeros(points, n_test)           # record: trace of every memristor at every step
x = torch.zeros(n_test)                        # current trace of every memristor
for t in range(points - 1):
    up = drive[t] > x                          # (a) pulse rule
    mem_v = torch.where(up, vp_grid, vn_grid)  # (b) one voltage per memristor
    mem_c = test_array.memristor_write(mem_v=mem_v.unsqueeze(1).unsqueeze(2),
                                       write_time=write_time, mem_v_amp=[0, 0])   # (c)
    x = ((mem_c - self.Gon) * self.trans_ratio).reshape(n_test)                  # (d)
    test_x[t + 1] = x                                                            # (e)
```

(a) `drive[t]` is the golden Z at this step, one number. `x` is 10,000 numbers.
`drive[t] > x` compares the one against all and gives 10,000 True/False:
True where that memristor is currently below Z (needs pushing up), False
where above. This is the P rule "move toward Z". In `_calibrate_one` the
corresponding line is `mem_s == 0`, one yes/no from the spike that applies to
every candidate alike.

(b) `torch.where(condition, a, b)` picks, elementwise, `a` where the condition
is True and `b` where False. So each memristor gets *its own* `v_pos` if it is
below Z, *its own* `v_neg` if above. Result: 10,000 voltages. In
`_calibrate_one` this is `v_tensor if mem_s == 0 else v_pos`: one decision for
all. (`torch.where` is the same construct `memarray.py` uses to apply VTEAM
above and below threshold.)

(c) The write, identical to `_calibrate_one`. `unsqueeze(1).unsqueeze(2)`
turns the `[10000]` voltage vector into `[10000, 1, 1]`, the
`[batch, rows, cols]` shape the array expects.

(d) Conductance back to trace, the same `(G - Gon) * trans_ratio` as
everywhere. `reshape(n_test)` flattens `[10000, 1, 1]` to `[10000]` so the
next step's comparison in (a) works.

(e) Store this step's values.

### Pick the winner

```python
mse = ((test_x - trace.reshape(points, 1)) ** 2).mean(dim=0)   # [10000], one error per memristor
min_index = int(torch.argmin(mse))                             # which memristor was closest
v_pos, v_neg = float(vp_grid[min_index]), float(vn_grid[min_index])
```

`trace.reshape(points, 1)` makes the golden P a column so that one
subtraction applies it to all 10,000 columns of `test_x`. Square, average over
time (`dim=0`), and there is one number per memristor. `argmin` is the index
of the smallest; the two grids give that memristor's voltages. `int()` and
`float()` turn the tensor results into plain Python numbers, so `v_pair`
stores ordinary floats like the STDP `vpos`/`vneg`.

`_calibrate_one` computes the same thing with a Python loop,
`for i in range(n_test): mse[i] = F.mse_loss(...)`. For 500 candidates that
is fine; for 10,000 the single tensor expression is much faster and reads as
one formula.

### Plot and return

```python
if plot:
    ...
    ax.plot(trace, ...)                    # golden P
    ax.plot(test_x[:, min_index], ...)     # the winning memristor's recorded trace
    plt.savefig(f'voltage_generation_{name}.png', ...)
return v_pos, v_neg
```

`_calibrate_one` re-simulates the winner at batch size 1 for the plot. Here
the winner's curve is already column `min_index` of `test_x`, so it is sliced
out. Return is the same as `_calibrate_one`.

---

## 3. What to expect from it

- Because the step size cannot follow `kp * (Z - P)`, the fit is a best
  compromise, not exact, on every device including `ideal`.
- The fit depends heavily on the golden traces it is given. With the STDP
  single spike at step 10, golden P is of order 1e-3 and the chosen pulses are
  too weak for real activity (memristor P peaks at 0.023 against a golden
  0.077 on a 15 % spike train). A calibration spike train near the operating
  rate is the fix; this is decided in the spike-handling step.
- If a winning voltage lands on the edge of the candidate range (exactly 1.0 x
  or 2.5 x threshold), the range should be widened. `v_neg` for P tends to sit
  at the 1.0 x edge, i.e. "never pull down", because golden P decays so slowly.
