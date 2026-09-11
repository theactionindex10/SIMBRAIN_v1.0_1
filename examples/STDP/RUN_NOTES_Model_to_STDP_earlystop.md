# Run notes: `Model_to_STDP_earlystop.py` on cheddar

Written 2026-09-10. The script itself is unmodified; everything below is about how it behaves
and how it is being run. Line numbers refer to `Model_to_STDP_earlystop.py` in this folder
(same file size as the copy on cheddar).

---

## 1. Current state (as of 2026-09-10 19:55)

| Item | Value |
|---|---|
| Server | `varun@cheddar.eecs.kth.se` — NVIDIA RTX 3080 Ti (12 GB) |
| Folder on server | `~/SIMBRAIN_BCPNN/examples/STDP` |
| Interpreter | system `/usr/bin/python3.10` (no venv / conda) |
| Running command | `python3 -u Model_to_STDP_earlystop.py --multiple_test_no 99 2>&1 \| tee -a run.log` |
| Runs inside | tmux session `stdp` (survives SSH disconnects, **not** reboots) |
| Started | 2026-09-10 19:54 |
| Repetitions finished | 1 (from the first run, before the restart) |
| Repetitions remaining | 99 (this run) |
| Speed | ~3 s per training batch, 1000 batches per epoch |
| Expected finish | roughly 2026-09-24 (99 x ~3.3 h), longer if some repetitions don't stop early |

---

## 2. Will it terminate? Yes.

Every loop has a fixed bound — there is no unbounded `while` loop:

- `for test_cnt in range(multiple_test_no)` — the whole train+test is repeated
  **100 times by default** (`:26`, `:157`).
- `for epoch in range(n_epochs)` — at most **10 epochs** per repetition (`:246`).
- Each epoch = 50,000 training images (5/6 of MNIST train) / batch 50 = **1,000 batches**.
  The `step == n_train * 5 / 6` break (`:254`) is never reached; the DataLoader ends the epoch.
- Early stopping can only shorten a repetition.
- `while` loops in `bindsnet/datasets/preprocess.py`, `simbrain/memristor_fit.py` and
  `simbrain/Fitting_Functions/conductance_fitting.py` are not on this run's path
  (image augmentation / device fitting; default device is `ideal`).

### Why it takes so long

1. **100 repetitions.** A line is written to `Accuracy_Results_earlystop.txt` only when a whole
   repetition finishes.
2. **Validation dominates the cost.** Validation starts once **110,000 training images** have
   been seen (`init_num`, hard-coded at `:162`). Each check runs all 10,000 validation images
   for 250 timesteps:
   - accuracy **< 90 %** -> check every 5,000 training images (up to 78 checks)
   - accuracy **> 90 %** -> check every 1,000 training images (up to 390 checks)
3. **CPU-bound.** On cheddar the GPU sat at a steady ~11 % while the main Python process used
   100 % of one core.

### Early-stopping rule (`:334-354`)

- If validation accuracy `>=` best so far -> counter reset to 0, weights saved to `best_model.pth`.
- Else if accuracy < 90 % -> counter += 100; else counter += 20.
- Stop when counter >= 500 (`patience`), i.e. 5 non-improving checks below 90 %,
  or 25 non-improving checks above 90 %.
- Caveat: because the reset uses `>=`, an exactly flat accuracy never triggers early stopping.
  The 10-epoch cap still ends the repetition.

---

## 3. Results so far

`Accuracy_Results_earlystop.txt`:

```
All activity accuracy:0.9302   best capacity:149000   test capacity:174000   total time:11871.090168952942
```

| Field | Meaning |
|---|---|
| **All activity accuracy: 0.9302** | **93.02 %** on the full 10,000-image MNIST test set, "all activity" classification (each neuron labelled with the digit it fires most for; prediction = digit whose neurons fire most on average). |
| **best capacity: 149000** | Training images seen when validation accuracy peaked (~3 epochs). These are the weights in `best_model.pth`. |
| **test capacity: 174000** | Misleading name — training images seen when early stopping ended training. 149k + 25 checks x 1k = 174k, consistent with the > 90 % rule above. |
| **total time: 11871 s** | **3 h 18 m** wall time for that repetition (train + validation + test). |

- The peak **validation** accuracy is not saved to the file; it was only printed to the terminal.
- **Caveat on the test accuracy:** the final test loads the weights from 149k images (`:388`)
  but uses the neuron-to-digit `assignments` from the end of training (174k). `assignments` is
  never saved alongside the weights, so the number is not exactly the accuracy of the best
  model (could be slightly higher or lower).
- **Plots:** `plots/weights`, `plots/performance`, `plots/assignments` exist but are **empty**.
  The script creates the folders and file names (`:230-237`) but never draws anything; the
  `--plot` flag is unused. This is expected.
- Only the CUDA seed is set (`:102`), so the train/validation split and shuffle order differ
  between repetitions. Repetitions are independent samples, so stopping and restarting
  doesn't bias results.

---

## 4. Backups made (on cheddar, 2026-09-10 19:49)

In `~/SIMBRAIN_BCPNN/examples/STDP`, verified byte-identical with `cmp`, timestamps preserved:

| Backup | Source | Notes |
|---|---|---|
| `best_model_run1.pth` | `best_model.pth` (18:32:30) | Repetition 1's best weights (20,113,909 bytes) |
| `Accuracy_Results_earlystop.backup.txt` | `Accuracy_Results_earlystop.txt` (19:13:14) | 1 line |

`best_model.pth` is overwritten by every repetition, so only the last repetition's best model
remains unless you copy it.

---

## 5. How to stop and restart safely

### What "safe" means for this script

- **Finished repetitions are never lost** — each is one appended line in the results file.
- **The repetition in progress is always lost.** The script cannot resume mid-repetition; a
  restart builds a new network. `best_model.pth` is only loaded for the final test, never to
  resume training.
- **Killing at any moment is safe.** Even if it happens during a save and `best_model.pth` is
  corrupted, the next repetition's first validation check always overwrites it before it is
  ever read.
- **Least wasteful moment to stop:** just after a new line appears in the results file.

### Step 1 — Stop

Attached to tmux (`tmux attach -t stdp`): press **Ctrl+C** (twice if needed).

Or from any SSH session, without attaching:

```sh
tmux send-keys -t stdp C-c
# fallback if the tmux session is gone:
pkill -INT -f Model_to_STDP_earlystop.py
```

Check when the last repetition finished, to choose a good moment:

```sh
wc -l ~/SIMBRAIN_BCPNN/examples/STDP/Accuracy_Results_earlystop.txt
date -r ~/SIMBRAIN_BCPNN/examples/STDP/Accuracy_Results_earlystop.txt
```

Verify it is fully stopped:

```sh
ps -ef | grep Model_to_STDP | grep -v grep   # should print nothing
nvidia-smi                                   # no python3 listed
```

Optional — keep this repetition's best model and a copy of the results:

```sh
cd ~/SIMBRAIN_BCPNN/examples/STDP
cp -p best_model.pth best_model_runN.pth     # pick a new name each time
cp -p Accuracy_Results_earlystop.txt Accuracy_Results_earlystop.backup.txt
```

### Step 2 — Restart inside tmux

The script doesn't know how many repetitions are already done, so work out how many are left:

```sh
tmux new -s stdp                              # if a session "stdp" still exists: tmux attach -t stdp
cd ~/SIMBRAIN_BCPNN/examples/STDP             # required: data and output paths are relative
N=$((100 - $(wc -l < Accuracy_Results_earlystop.txt)))
echo "running $N more repetitions"
python3 -u Model_to_STDP_earlystop.py --multiple_test_no $N 2>&1 | tee -a run.log
```

- `-u` = write output immediately, so `run.log` stays current.
- `tee -a` = append to `run.log` instead of replacing it.
- Detach: **Ctrl+B, then D**. The job keeps running after you log out.

### Step 3 — Check on it

```sh
tail -3 ~/SIMBRAIN_BCPNN/examples/STDP/run.log
tmux attach -t stdp
```

After a reboot of cheddar the tmux session is gone; just do Step 2 again.

---

## 6. Gotchas if you shorten runs with flags

- **`init_num = 110000` is hard-coded (`:162`).** If `--n_train` / `--n_epochs` are small enough
  that 110,000 training images are never reached, no validation runs and `best_model.pth` is
  never saved. The final test (`:388`) would then **silently load an old `best_model.pth`**
  from a previous run and report a meaningless accuracy.
- **Only some `--n_train` values shorten an epoch.** The break at `:254` only fires when
  `ceil(n_train / 50) * 5/6` is a whole number (e.g. `6000` works; `5000` gives 83.33, so the
  full epoch still runs).
- `--plot` does nothing in this script.
