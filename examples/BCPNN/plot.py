
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import TwoSlopeNorm
import numpy as np

def plot_receptive_fields(model, save_dir="figures"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    Hhid = model.param.Hhid
    Hin = model.param.Hin

    # Connectivity: hidden hypercolumns x input hypercolumns
    conn = np.asarray(model.ihprj.get_field_int("CHJHIX"))
    conn = conn.reshape(Hhid, Hin)

    ncols = 6
    nrows = int(np.ceil(Hhid / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(1.5 * ncols, 1.5 * nrows))
    axs = axs.ravel()

    for h in range(Hhid):
        axs[h].imshow(conn[h].reshape(28, 28), cmap="gray")
        axs[h].axis("off")
        axs[h].set_title(f"H{h}", fontsize=8)

    for ax in axs[Hhid:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / "receptive_fields.png", dpi=200)
    plt.close()


def plot_filters(model, save_dir="figures", hids=(0, 1, 2, 3), mids=(0, 1, 2, 3, 4)):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    p = model.param

    # Extended weights: Hhid x Mhid x Hin x Min
    W = np.asarray(model.ihprj.get_field_flt("WJIX"))
    W = W.reshape(p.Hhid, p.Mhid, p.Hin, p.Min)

    # Use ON-channel input weights
    F = W[:, :, :, 1]  # Hhid x Mhid x Hin

    # Use only nonzero values for color scaling
    vals = []
    for h in hids:
        for m in mids:
            f = F[h, m]
            vals.append(f[f != 0])

    vals = np.concatenate(vals)
    vmax = np.percentile(np.abs(vals), 99)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    cmap = plt.cm.seismic.copy()
    cmap.set_bad("white")

    fig, axs = plt.subplots(
        len(hids),
        len(mids),
        figsize=(1.5 * len(mids), 1.5 * len(hids)),
    )

    axs = np.asarray(axs)
    if axs.ndim == 1:
        axs = axs.reshape(len(hids), len(mids))

    for r, h in enumerate(hids):
        for c, m in enumerate(mids):
            ax = axs[r, c]

            filt = F[h, m].reshape(28, 28)

            # Mask zero/non-connected weights
            filt = np.ma.masked_where(filt == 0, filt)

            ax.imshow(filt, cmap=cmap, norm=norm)
            # ax.axis("off")

            if r == 0:
                ax.set_title(f"m{m}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"H{h}", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_dir / "weight_filters.png", dpi=200)
    plt.close()
