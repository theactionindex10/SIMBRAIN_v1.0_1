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

The code is written to mirror `_calibrate_one` statement for statement; the
two places it must differ are marked `DIFFERENCE 1` and `DIFFERENCE 2` in the
source. Read it next to `_calibrate_one`.

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
curve. (`_calibrate_one` takes `kz` because its closed forms need the number.)

### Setup (identical to `_calibrate_one`)

```python
points = len(trace)
mem_info = self.memristor_info_dict[self.device_name]
write_time = 1
```

### DIFFERENCE 2: candidate lists for both voltages

`_calibrate_one` has `v_pos` from a closed form and one list,
`v_tensor = torch.arange(v_on, v_on - 0.01*500, -0.01)`, of 500 `v_neg`
candidates. Here there is no closed form, so both voltages get a list:

```python
v_off = mem_info['v_off']
v_on = mem_info['v_on']
n_pos = 100
n_neg = 100
v_pos_tensor = v_off * (1 + torch.linspace(0, 1.5, n_pos))   # 1.0x .. 2.5x threshold, e.g. ideal 1.0 .. 2.5 V
v_neg_tensor = v_on  * (1 + torch.linspace(0, 1.5, n_neg))   # ideal: -1.0 .. -2.5 V
```

A memristor does not move unless the pulse is above its threshold (`v_off`
positive, `v_on` negative), so candidates start at 1x threshold.
`torch.linspace(0, 1.5, 100)` is 100 evenly spaced numbers 0.0 to 1.5;
`1 + ...` makes them 1.0 to 2.5. Multiples of threshold are used instead of a
fixed 0.01 V step so the range scales to devices with very different
thresholds (CMS: 0.2 V).

Every *pair* must be tried, 100 x 100 = 10,000. A batch is one list, so the
pairs are laid out as two lists of length 10,000 that line up: entry `k` of
each, taken together, is pair `k`:

```python
n_test = n_pos * n_neg                                   # 10,000, the batch size
v_pos_grid = v_pos_tensor.repeat_interleave(n_neg)       # each v_pos repeated n_neg times in place
v_neg_grid = v_neg_tensor.repeat(n_pos)                  # the whole v_neg list tiled n_pos times
```

Miniature, 3 positive x 2 negative:

```
v_pos_tensor                           [1.0, 1.5, 2.0]
v_neg_tensor                           [-1.0, -2.0]
v_pos_grid = ....repeat_interleave(2)  [1.0, 1.0, 1.5, 1.5, 2.0, 2.0]
v_neg_grid = ....repeat(3)             [-1.0,-2.0,-1.0,-2.0,-1.0,-2.0]
pair k:                                (1.0,-1.0) (1.0,-2.0) (1.5,-1.0) (1.5,-2.0) (2.0,-1.0) (2.0,-2.0)
```

All six combinations, once each.

### The batch simulation (same structure as `_calibrate_one`)

```python
test_array.set_batch_size(batch_size=n_test)
test_x = torch.zeros(points, n_test, 1, 1)        # every memristor's trace at every step

for t in range(points - 1):
    # DIFFERENCE 1: polarity from comparing the drive with each memristor's current state
    mem_d = torch.tensor(drive[t], dtype=torch.float64)
    mem_up = mem_d > test_x[t, :, 0, 0]
    mem_v = torch.zeros(n_test)
    mem_v[mem_up] = v_pos_grid[mem_up]
    mem_v[~mem_up] = v_neg_grid[~mem_up]

    mem_c = test_array.memristor_write(mem_v=mem_v.unsqueeze(1).unsqueeze(2), write_time=write_time, mem_v_amp=[0,0])
    test_x[t + 1] = (mem_c - self.Gon) * self.trans_ratio
```

Compare with `_calibrate_one`'s loop body:

```python
    mem_s = torch.tensor(spike[t], dtype=torch.float64)
    mem_v = v_tensor if mem_s == 0 else torch.tensor(v_pos).expand(n_test)
```

There, `mem_s` is the spike, one yes/no, and the whole batch gets `v_pos` or
the `v_neg` candidates together. Here, `mem_d` is the golden Z at step `t`,
one number, and `mem_up` compares it with all 10,000 current states
(`test_x[t, :, 0, 0]`), giving 10,000 True/False: True where that memristor
is below Z and must be pushed up. The next three lines are the same
boolean-mask assignment `_calibrate_one` uses in its plot block
(`mem_v[mem_v == 0] = v_neg`): memristors with `mem_up` True get their own
`v_pos` from the grid, the rest get their own `v_neg`. The write and the
conductance-to-trace line are identical to `_calibrate_one`.

### Compare results (identical to `_calibrate_one`)

```python
golden_x = trace.reshape(points, 1, 1, 1).expand(-1, n_test, -1, -1)
mse = torch.zeros(n_test)
for i in range(n_test):
    mse[i] = F.mse_loss(test_x[:, i, :, :], golden_x[:, i, :, :])
min_mse, min_index = torch.min(mse, 0)
v_pos = float(v_pos_grid[min_index])
v_neg = float(v_neg_grid[min_index])
```

Word for word the same as `_calibrate_one`, except that the winner's index
reads *two* voltages out of the two grids instead of one out of `v_tensor`.

### Plot and return (identical structure)

The plot block re-simulates the winning pair on a single memristor into
`mem_x`, as `_calibrate_one` does, with the polarity at each step from
`mem_d > mem_x[t]` instead of from the spike, then plots `trace` against
`mem_x` and saves `voltage_generation_{name}.png`. `return v_pos, v_neg`.

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
