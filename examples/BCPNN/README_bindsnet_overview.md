# BindsNET in the BCPNN example: what each module does and where SIMBRAIN plugs in

This is a map of the BindsNET copy under `examples/BCPNN/bindsnet/`, written for
someone who will be modifying it. It covers only the modules the BCPNN example
actually uses. Line numbers refer to the BCPNN copy as of 2026-09-24.

BindsNET is an open-source PyTorch library for spiking neural networks
(Hazan et al. 2018, UMass Amherst). The copy in this repo is **not** the stock
package: the SIMBRAIN authors forked it to hook their memristor `Mapping` into
the trace machinery, and the BCPNN copy carries a second layer of edits on top
of that (documented in `README_BCPNN_changes.md`). The STDP copy under
`examples/STDP/bindsnet/` is the SIMBRAIN authors' version, untouched by us.

---

## 1. How a simulation flows through the modules

```
Model_to_BCPNN.py                       (experiment script)
   |
   |-- bindsnet.datasets.MNIST + bindsnet.encoding.PoissonEncoder
   |        -> dict{"encoded_image": [time, 1, 28, 28] spikes, "label": int}
   |
   |-- bindsnet.models.IncreasingInhibitionNetwork(...)          builds:
   |        Network                        (network/network.py)
   |          layers["X"] = Input          (network/nodes.py)   784 neurons
   |          layers["Y"] = DiehlAndCookNodes                    625 neurons
   |          connections[("X","Y")] = Connection(PostPre)      (network/topology.py, learning/learning.py)
   |          connections[("Y","Y")] = Connection (inhibitory, no learning)
   |
   |-- Monitor(...)                        (network/monitors.py)  records spikes / voltages
   |
   |-- network.run(inputs, time)           one image, `time` timesteps:
   |        each step:  Connection.compute  -> input currents
   |                    Nodes.forward       -> spikes, membrane, TRACES  <-- SIMBRAIN memristor writes
   |                    LearningRule.update -> weight change            <-- SIMBRAIN memristor reads
   |                    Monitor.record
   |
   |-- bindsnet.evaluation.assign_labels / all_activity     accuracy
   |
   '-- network.reset_state_variables(); network.mem_t_update()   between images/batches
```

---

## 2. Module by module

### 2.1 `network/network.py`: the simulation driver

`class Network` (line 29). Holds `self.layers` (dict of `Nodes`),
`self.connections` (dict keyed by `(source_name, target_name)`) and
`self.monitors`. Everything else is orchestration.

| Method | Line | What it does |
|---|---|---|
| `__init__(dt, sim_params, batch_size, learning)` | 84 | Stores `dt` and `sim_params`. `sim_params` is the SIMBRAIN device/circuit config dict; it is not in upstream BindsNET. |
| `add_layer(layer, name)` | 123 | Registers a `Nodes`; calls `layer.train`, `layer.compute_decays(dt)` (which triggers memristor **calibration**), `layer.set_batch_size`. |
| `add_connection(conn, source, target)` | 139 | Registers a `Connection`; sets its `dt`. |
| `_get_inputs()` | 219 | For every connection, `connection.compute(source.s)` and sums the result into the target's input. |
| `run(inputs, time, init_batch_sign, ...)` | 260 | The timestep loop. Per step: `_get_inputs`, then `layer.forward` for every layer, then `connection.update` for every connection, then monitors. After the loop: `connection.normalize`. If `hardware_estimation` is on and memristors are in use, prints energy from every layer's `transform` (lines 432 to 455, edited by us to loop over trace rows). |
| `reset_state_variables()` | 458 | Calls `reset_state_variables` on every layer, connection and monitor. Between images. |
| `mem_t_update()` | 474 | SIMBRAIN addition: advances the memristor time counters on every layer's `transform`. Called by the script between batches. |
| `train(mode)` | 480 | Sets `self.learning` on the network and (via `torch.nn.Module.train`) on all layers and connections. |

**SIMBRAIN hooks here:** `sim_params` threading, the `hardware_estimation`
printout, `mem_t_update`. Our edit: the printout loops over `trace_kinds`.

### 2.2 `network/nodes.py`: neuron populations and the traces

`class Nodes(torch.nn.Module)` (line 9) is the base class. It owns the spike
tensor `s`, the traces, and, when memristors are in use, the SIMBRAIN Mapping
object `self.transform`. All the specific neuron models subclass it and call
`super().forward(x)` at the end of their own `forward`, which is where the trace
update runs.

| Method | Line | What it does |
|---|---|---|
| `__init__(n, shape, traces, tc_trace, ..., sim_params)` | 15 | Registers buffers `s`, `x` (trace), `tc_trace`, `trace_decay`; with memristors, creates `self.transform = BCPNNMapping(sim_params, shape)`. Our additions: `tau_zi`, `tau_zj`, `tau_e`, `tau_p`, `kzi`, `kzj`, `ke`, `kp`, buffers `e`, `p`, and the role labels `z_trace`, `e_trace`, `p_trace`. |
| `forward(x)` | 120 | Called by subclasses after they have set `self.s`. Memristor path: writes P, E, Z through `transform.mapping_write_p/e/z`. Software path (`device_name == 'trace'`): same arithmetic in plain tensors. |
| `reset_state_variables()` | 148 | Zeros `s`, `x`, `e`, `p`; calls `transform.reset_memristor_variables()`. |
| `update_SAF_mask()` | 165 | SIMBRAIN addition; forwards to the transform. |
| `compute_decays(dt)` | 170 | Computes `trace_decay`, `kzi`, `kzj`, `ke`, `kp` and calls `transform.voltage_generation(...)`, the one-time memristor **calibration**. Runs from `Network.add_layer`. |
| `set_batch_size(batch_size, init_batch_sign)` | 191 | Allocates `s`, `x`, `e`, `p` for the batch; on first call, `transform.set_batch_size_stdp`. |
| `train(mode)` | 215 | Sets `self.learning`; the memristor path is only taken when `learning` is True. |

Subclasses used by the BCPNN model:

- `class Input(Nodes)` (line 234): `forward` just sets `self.s = x`, the spikes
  handed in from the dataset. This is layer **X**, the presynaptic population.
- `class DiehlAndCookNodes(Nodes)` (line 1056): leaky integrate-and-fire with
  an adaptive threshold `theta` that grows by `theta_plus` on every spike and
  decays with `tc_theta_decay`, plus a "one spike per timestep" winner rule.
  This is layer **Y**, the postsynaptic population.

Other neuron models in the file (`LIFNodes`, `IzhikevichNodes`, `SRM0Nodes`,
...) are upstream BindsNET and unused here. They do not accept `sim_params`, so
they would need the same treatment as `Input` and `DiehlAndCookNodes` before
they could hold memristor traces.

**SIMBRAIN hooks here:** `self.transform`, the memristor branch in `forward`,
`compute_decays` calling `voltage_generation`, `update_SAF_mask`. This is the
file with most of our BCPNN edits.

### 2.3 `network/topology.py`: connections (synapses)

`class AbstractConnection(ABC, Module)` (line 13) and `class Connection`
(line 125). A `Connection` holds the weight matrix `w` of shape
`(source.n, target.n)` as a non-trainable `Parameter`, optional bias `b`, the
bounds `wmin`, `wmax`, a normalisation constant `norm`, and an instance of a
learning rule created from the `update_rule` kwarg.

| Method | Line | What it does |
|---|---|---|
| `compute(s)` | 186 | `s @ w`: turns source spikes into input current for the target. |
| `update(**kwargs)` | 230 | Delegates to `self.update_rule.update(...)` if `learning` is True, then applies an optional mask. |
| `normalize()` | 237 | Rescales `w` so every column sums to `norm` (78.4 in the model). Called after every image. |

Other connection types (`Conv2dConnection`, `LocalConnection`, ...) are
upstream and unused here.

**SIMBRAIN hooks here:** none in the current code. This is the natural home
for the P_ij memristor array, since the connection is the only object that
holds both `source` and `target` and has the `(N_i, N_j)` shape.

### 2.4 `learning/learning.py`: learning rules

`class LearningRule(ABC)` (line 19) is the base: stores the connection, the
learning rates `nu` (a pair, pre and post), a batch-reduction function, and
weight decay. Its `update()` (line 79) applies decay and clamps `w` to
`[wmin, wmax]`.

`class PostPre(LearningRule)` (line 136) is the STDP rule used by the model.
`_connection_update` (line 183):

- pre-synaptic term: `w -= nu[0] * outer(source.s, target_trace)`
- post-synaptic term: `w += nu[1] * outer(source_trace, target.s)`

where the traces come from `transform.mapping_read(..., kind='Z')` on the
memristor path (lines 197 and 215) or from `layer.x` on the software path.

Other rules in the file (`Hebbian`, `WeightDependentPostPre`, `MSTDP`,
`MSTDPET`, `Rmax`, `NoOp`) are upstream and unused. `NoOp` is what the
inhibitory `Y_to_Y` connection uses implicitly (no `update_rule` given).

**SIMBRAIN hooks here:** the two `mapping_read` calls. This is where a BCPNN
rule would replace `PostPre`: read P_i, P_j, P_ij and set
`w = log(P_ij / (P_i P_j))` instead of the STDP increments.

### 2.5 `models/models.py`: ready-made network builders

`class IncreasingInhibitionNetwork(Network)` (line 304) is the one the example
uses. It builds:

- `Input(n=784, shape=(1,28,28), traces=True, ...)` as `"X"`,
- `DiehlAndCookNodes(n=n_neurons, traces=True, thresh=-52, rest=-65, reset=-60, refrac=5, tc_decay=100, theta_plus, tc_theta_decay, ...)` as `"Y"`,
- `Connection(X, Y, w=0.3*rand, update_rule=PostPre, nu, wmin=0, wmax=1, norm=78.4)`,
- `Connection(Y, Y, w=...)`: fixed lateral inhibition whose strength grows with
  the distance between neurons on a square grid (`start_inhib`, `max_inhib`);
  the experiment script makes it stronger over time.

Our edit: it takes `tau_zi`, `tau_zj`, `tau_e`, `tau_p` and passes them to both
layers, with `z_trace='Zi'` on X and `z_trace='Zj'` on Y.

The other models in the file (`TwoLayerNetwork`, `DiehlAndCook2015`, ...) are
upstream and do not pass `sim_params`, so they cannot use memristors as-is.

### 2.6 `network/monitors.py`: recording

`class Monitor` (line 22): given an object and a list of state-variable names
(`"s"`, `"v"`, ...), appends a copy of each after every timestep; `get(var)`
returns the stacked recording `[time, batch, ...]`. The script uses one on Y's
spikes for classification and one on Y's voltage. Unmodified upstream code.

### 2.7 `encoding/`: turning images into spikes

`encodings.poisson(datum, time, dt)` (encodings.py line 99): for each pixel
intensity (treated as a rate in Hz after the script multiplies by
`intensity=64`), samples inter-spike intervals from a Poisson distribution and
returns a `[time, ...]` binary spike tensor. `PoissonEncoder` (encoders.py
line 88) is the callable wrapper the dataset applies to every image. Other
encoders (`bernoulli`, `rank_order`, `single`, `repeat`) are available and
unused. Unmodified upstream code.

### 2.8 `datasets/`: MNIST loading

`datasets/__init__.py` wraps every `torchvision` dataset so that `__getitem__`
returns a dict with `"image"`, `"label"`, `"encoded_image"` (the encoder
applied to the image) and `"encoded_label"`. `MNIST` is
`create_torchvision_dataset_wrapper("MNIST")` (torchvision_wrapper.py). The
other files in the folder (`alov300.py`, `davis.py`, `spoken_mnist.py`,
`preprocess.py`) are video/audio datasets not used here; `preprocess.py`
imports `cv2` and `torchvision` at package load, which is why the package
does not import on a machine without them. Unmodified upstream code.

### 2.9 `evaluation/evaluation.py`: classification from spikes

- `assign_labels(spikes, labels, n_labels, rates)` (line 8): for each Y neuron,
  its average firing rate per digit class over the last `update_interval`
  images; the neuron is assigned the class it fires most for.
- `all_activity(spikes, assignments, n_labels)` (line 96): for a test image,
  the predicted class is the one whose assigned neurons fired most on average.
- `proportion_weighting` (line 131): same, weighted by each neuron's class
  proportions. Used by the dynamic-train script, not the early-stop one.

Unmodified upstream code. These are the same "assignment" readout the paper
uses for the SNN accuracy numbers.

### 2.10 `utils.py`, `learning/reward.py`, `__init__.py`

`utils.py`: reshaping helpers for plotting weights (`get_square_weights`,
`get_square_assignments`) and `im2col_indices` for convolutional rules.
`reward.py`: reward-modulated learning support, unused. The `__init__.py`
files just re-export. All unmodified.

---

## 3. Where SIMBRAIN and BCPNN touch BindsNET, in one table

| File | SIMBRAIN authors' hook | Our BCPNN edit |
|---|---|---|
| `network/network.py` | `sim_params`, `mem_t_update`, energy printout | printout loops over trace rows |
| `network/nodes.py` | `transform`, memristor branch in `forward`, `compute_decays` calibration, `update_SAF_mask` | `BCPNNMapping`; `tau_*`, `k*`; `e`, `p` buffers; `z/e/p_trace`; P, E, Z cascade in `forward` and software path |
| `network/topology.py` | none | none yet (P_ij goes here) |
| `learning/learning.py` | `mapping_read_stdp` in `PostPre` | renamed to `mapping_read(..., kind='Z')` (BCPNN rule goes here) |
| `models/models.py` | `sim_params`, `batch_size` on `IncreasingInhibitionNetwork` | `tau_zi`, `tau_zj`, `tau_e`, `tau_p`, `z_trace` per layer |
| `monitors.py`, `encoding/`, `datasets/`, `evaluation/`, `utils.py` | none | none |

---

## 4. Things to know when editing

- **Layer order matters.** `Network.run` calls `forward` on layers in the order
  they were added (X then Y) and connections likewise. Within one timestep Y
  sees X's spikes from the *same* step through `_get_inputs`, which is computed
  before any `forward`.
- **`learning` is a switch on three levels.** `network.train(False)` turns it
  off on the network, every layer and every connection. With it off, layers
  take the software trace path and connections skip their learning rule.
  Memristor writes only happen while `learning` is True.
- **Batch is a dimension everywhere.** `s`, `x`, `e`, `p` are `[batch, *shape]`;
  the memristor arrays are `[batch, 1, N]`, one independent physical row per
  batch entry. `reduction=torch.sum` in the learning rule sums weight updates
  over the batch.
- **Two copies of BindsNET exist.** Edit only `examples/BCPNN/bindsnet`. The
  `sys.path.append('../../')` in the scripts finds `simbrain`, and the local
  `bindsnet` folder shadows any installed one.
- **Running locally in WSL** needs `opencv-python` and `torchvision`, or stubs
  for `cv2`, `torchvision` and `bindsnet.datasets` in `sys.modules` if only the
  network is being tested.
