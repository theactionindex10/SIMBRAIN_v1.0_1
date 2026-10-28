import torch
import argparse
from dataclasses import dataclass
import numpy as np
import BcpnnSim as sim  
from plot import plot_receptive_fields, plot_filters
from preprocess import MNISTDataset
from simbrain.mapping import BCPNNMapping

# network configuration
parser = argparse.ArgumentParser(description="Run MNIST with a Python-built BCPNNSim network")
parser.add_argument("--multiple_test_no", type=int, default=100) # Number of tests to run
parser.add_argument("--n_epochs", type=int, default=10) # Number of training epochs
parser.add_argument("--n_test", type=int, default=1000) # Number of test samples
parser.add_argument("--n_train", type=int, default=1000) # Number of training samples
parser.add_argument("--epochs_usup", type=int, default=5)
parser.add_argument("--epochs_sup", type=int, default=5)
parser.add_argument("--time", type=int, default=3)
# memristor device configuration
parser.add_argument("--memristor_device", type=str, default='ideal') # trace: do not use memristors; ideal, ferro, MF, CMS, or mine
parser.add_argument("--c2c_variation", type=bool, default=False)
parser.add_argument("--d2d_variation", type=int, default=0) # 0: No d2d variation, 1: both, 2: Gon/Goff only, 3: nonlinearity only
parser.add_argument("--stuck_at_fault", type=bool, default=False)
parser.add_argument("--retention_loss", type=bool, default=False)
parser.add_argument("--aging_effect", type=int, default=0) # 0: No aging effect, 1: equation 1, 2: equation 2
# circuit configuration
parser.add_argument("--memristor_structure", type=str, default='trace') # trace or STDP_crossbar
parser.add_argument("--input_bit", type=int, default=1) # 1-64 for STDP, 2-64 for MLP & CNN
parser.add_argument("--ADC_precision", type=int, default=16) # 2-32
parser.add_argument("--ADC_setting", type=int, default=4) # 2: two memristor crossbars use one ADC; 4: one memristor crossbar use one ADC
parser.add_argument("--ADC_rounding_function", type=str, default='floor')  # floor or round
parser.add_argument("--wire_width", type=int, default=200) # In practice, wire_width shall be set around 1/2 of the memristor size; Ideal: 200nm, Ferro: 200nm, MF: 10um, CMS: 10um 
parser.add_argument("--CMOS_technode", type=int, default=45) # 7, 10, 14, 22, 32, 45, 65, 90, 130(nm)
parser.add_argument("--device_roadmap", type=str, default='HP') # HP: High Performance or LP: Low Power
parser.add_argument("--temperature", type=int, default=300) # Temperature will only affect the peripheral circuits, not the memristor itself
parser.add_argument("--hardware_estimation", type=int, default=False)
parser.add_argument("--memristor_inference", type=int, default=0)

parser.set_defaults(plot=False, gpu=True)

args = parser.parse_args()
n_epochs_usup = args.epochs_usup
n_epochs_sup = args.epochs_sup
n_test = args.n_test
n_train = args.n_train
plot = args.plot
gpu = args.gpu
sim_params = {'device_structure': args.memristor_structure, 'device_name': args.memristor_device,
              'c2c_variation': args.c2c_variation, 'd2d_variation': args.d2d_variation,
              'stuck_at_fault': args.stuck_at_fault, 'retention_loss': args.retention_loss,
              'aging_effect': args.aging_effect, 'wire_width': args.wire_width, 'input_bit': args.input_bit,
              'batch_interval': args.time, 'CMOS_technode': args.CMOS_technode, 'ADC_precision': args.ADC_precision,
              'ADC_setting': args.ADC_setting, 'ADC_rounding_function': args.ADC_rounding_function,
              'device_roadmap': args.device_roadmap, 'temperature': args.temperature,
              'hardware_estimation': args.hardware_estimation, 'memristor_inference': args.memristor_inference}
@dataclass
class Params:
    # Reproducibility
    seed: int = 88192386

    # Dataset sizes
    trnpat: int = 1000
    tenpat: int = 1000

    # Network architecture: 784x2 -> 30x100 -> 1x10
    Hin: int = 784
    Min: int = 2
    Hhid: int = 30
    Mhid: int = 100
    Hut: int = 1
    Mut: int = 10

    # Learning schedule
    nepoch_usup: int = 5
    nepoch_sup: int = 5
    nloop: int = 3

    # BCPNN learning parameters
    eps: float = 1e-8
    taup: float = 1.0

    # Population dynamics
    actfn: str = "SOFTMAX"
    maxfq: float = 1000.0
    nampl: float = 0.01
    nfreq: float = 100.0

    # Structural plasticity on input -> hidden projection
    updconnint: int = 1000
    nactHi: int = 81
    nsilHi: int = 703
    nswap: float = 100.0
    nrepl: float = -1.0
    swaprthr: float = 1.1
    knrepl: float = 2.0

    
def encode_mnist(img):
    """
    Encode one MNIST image as 784 hypercolumns x 2 minicolumns.

    For each pixel value p:
        p -> [1 - p, p]

    Output shape:
        784 * 2 = 1568
    """
    x = np.asarray(img, dtype=np.float32).reshape(-1) #Reshape image into 1-D array
    x = np.clip(x, 0.0, 1.0) # Clipping the image pixels to binary 1 or 0 values.

    pat = np.empty((784, 2), dtype=np.float32)  #Creates a 784 HC x 2 MC empty array of float type
    pat[:, 0] = 1.0 - x #selecting first column and assigning  probability value 1-p
    pat[:, 1] = x #selecting second column and assigning probability value p

    return pat.reshape(-1)


def iter_encoded_mnist(mnist, split, n):
    """Yield encoded MNIST patterns and labels."""
    for idx in range(n):
        img = mnist.get_image(idx, split=split)
        pat = encode_mnist(img)
        lbl = mnist.get_label(idx, split=split)
        yield pat, lbl


# --------------------------------------------------
# BCPNN model
# --------------------------------------------------

class BCPNNMNISTModel:
    def __init__(self, param):
        self.param = param
        self.updconn_counter = 0
        self.rec = {'zi' : [], 'axoact' : [], 'hihjhi' : [],  'zj' : [], 'pi' : [], 'pj' : [], 'pji' : []}
        self.pji_index = None
        sim.clear_all()
        sim.gsetseed(param.seed)

        self._build_network()

    def _build_network(self):
        self.net = sim.Netw("mnist_net")
        self._build_populations()
        self._build_projections()

    def _build_populations(self):
        p = self.param

        self.inpop = self.net.add_pop(p.Hin, p.Min, "inpop")
        self.inpop.setactfn(p.actfn)
        self.inpop.setmaxfq(p.maxfq)

        self.hidpop = self.net.add_pop(p.Hhid, p.Mhid, "hidpop")
        self.hidpop.setactfn(p.actfn)
        self.hidpop.setnfreq(p.nfreq)
        self.hidpop.setnampl(p.nampl)

        self.outpop = self.net.add_pop(p.Hut, p.Mut, "outpop")
        self.outpop.setactfn(p.actfn)

    def _build_projections(self):
        p = self.param
        
        if(args.memristor_inference == 1):
            self.m = BCPNNMapping(sim_params, shape = {'Pi': 47040, 'Pj': 3000, 'Pji': 4704000} )
            self.m.voltage_generation_bcpnn('notes/ihprj_golden_traces.npz')
            self.m.set_batch_size_bcpnn(1,True)

        self.ihprj = self.net.add_sparse_prj(
            self.inpop,
            self.hidpop,
            p.nactHi,
            p.nsilHi,
            "ihprj",
        )
        self._configure_ihprj()

        self.hoprj = self.net.add_prj(self.hidpop, self.outpop, "hoprj")
        self.hoprj.settaup(p.taup)
        self.hoprj.seteps(p.eps)

    def _configure_ihprj(self):
        p = self.param

        self.ihprj.settaup(p.taup)
        self.ihprj.seteps(p.eps)
        self.ihprj.setwgain(1)
        self.ihprj.setbgain(1)

        self.ihprj.setnswap(p.nswap)
        self.ihprj.setswaprthr(p.swaprthr)
        self.ihprj.setknrepl(p.knrepl)

        if p.nrepl >= 0:
            self.ihprj.setnrepl(p.nrepl)

    def reset_state(self):
        sim.pop_resetall()

    def set_input(self, pat):
        self.inpop.setinput(np.asarray(pat, dtype=np.float32).reshape(-1))

    def set_label(self, lbl):
        self.outpop.setinput(np.asarray(lbl, dtype=np.float32).reshape(-1))

    def run_steps(self, prns=None):
        p = self.param

        for _ in range(p.nloop):
            if prns is None:
                self.net.updstate()
            else:
                self.net.updstate_prns(prns)
            sim.advance()

    def train_unsupervised_online(self, pat):
        p = self.param

        self.hidpop.setbwgain(1)
        self.reset_state()
        self.set_input(pat)

        for loop in range(p.nloop):
            prns = [1.0 if loop == p.nloop - 1 else 0.0, 0.0]
            self.net.updstate_prns(prns)
            if prns[0] > 0 and args.memristor_inference == 1:
                Hj, denHi, Mi, Mj = p.Hhid, p.Hin, p.Min, p.Mhid          # 30, 784, 2, 100
                denNi = denHi * Mi                                         # 1568
                axoact = np.asarray(self.ihprj.get_field_flt("AXOACT"))
                hihjhi = np.asarray(self.ihprj.get_field_int("HIHJHI"))
                hr = hihjhi.reshape(Hj, denHi).astype(int)
                src = (hr[:, :, None] * Mi + np.arange(Mi)[None, None, :]).reshape(Hj, denNi)
                zi = (axoact[src] * (1 - p.eps) + p.eps).reshape(-1).astype(np.float32)   
                zj = np.asarray(self.ihprj.get_field_flt("ZJ"), dtype=np.float32)
                drive = {'Pi': zi,
                         'Pj': zj,
                         'Pji': (zi.reshape(Hj, 1, denNi) * zj.reshape(Hj, Mj, 1)).reshape(-1)}
                mem_P = {}
                for name in ('Pi', 'Pj', 'Pji'):
                    self.m.mapping_write_bcpnn(drive=torch.from_numpy(drive[name]).reshape(1, -1), trace=name)
                    mem_P[name] = self.m.mapping_read_bcpnn(name).reshape(-1).numpy().astype(np.float32)
                    mem_P[name] = np.maximum(mem_P[name], p.eps)       # engine floor: a zero-state cell reads < 0 through the ADC (floor rounding) and log() would give NaN
                self.ihprj.set_field_flt("PI", mem_P['Pi'])
                self.ihprj.set_field_flt("PJ", mem_P['Pj'])
                self.ihprj.set_field_flt("PJI", mem_P['Pji'])
                self.ihprj.updbw(force=True)                               # Bj, Wji from the memristor traces
            sim.advance()

        self._maybe_update_connections()

    def train_supervised_online(self, pat, lbl):
        p = self.param

        # Clamp output label during supervised learning.
        self.hidpop.setnampl(0)
        self.outpop.setbwgain(0)

        self.reset_state()
        self.set_input(pat)
        self.set_label(lbl)

        for loop in range(p.nloop):
            prns = [0.0, 1.0 if loop == p.nloop - 1 else 0.0]
            self.net.updstate_prns(prns)
            sim.advance()

        # Re-enable output bottom-up support and update output projection.
        self.outpop.setbwgain(1)
        self.hoprj.updbw()

    def infer_online(self, pat, lbl=None):
        self.reset_state()
        self.set_input(pat)
        self.outpop.clearinput()

        self.run_steps()

        pred_vec = np.asarray(self.outpop.get_field_flt("ACT"))
        pred_id = int(np.argmax(pred_vec))

        if lbl is None:
            return pred_id, pred_vec

        true_id = int(np.argmax(lbl))
        return int(pred_id == true_id), pred_id, pred_vec

    def _maybe_update_connections(self):
        p = self.param

        if p.updconnint <= 0:
            return

        if self.updconn_counter % p.updconnint == 0:
            self.ihprj.swapconns()
            self.ihprj.replconns()

            if p.nswap == 0:
                self.ihprj.updMIsc()

            self.ihprj.miscsum()

        self.updconn_counter += 1


# --------------------------------------------------
# Training / evaluation
# --------------------------------------------------

def train_model(model, mnist):
    p = model.param

    print("\nTraining")

    print("\nUnsupervised")
    for epoch in range(p.nepoch_usup):
        for pat, _ in iter_encoded_mnist(mnist, "train", p.trnpat):
            model.train_unsupervised_online(pat)

        print(
            f"  epoch {epoch + 1}/{p.nepoch_usup} "
            f"| simstep {sim.get_simstep():6d} "
            f"| swapped {model.ihprj.gnswapped:5d}"
        )

    print("\nSupervised")
    for epoch in range(p.nepoch_sup):
        for pat, lbl in iter_encoded_mnist(mnist, "train", p.trnpat):
            model.train_supervised_online(pat, lbl)

        print(f"  epoch {epoch + 1}/{p.nepoch_sup} | simstep {sim.get_simstep():6d}")


def evaluate(model, mnist, split, n):
    n_correct = 0

    for pat, lbl in iter_encoded_mnist(mnist, split, n):
        correct, _, _ = model.infer_online(pat, lbl)
        n_correct += correct

    return 100.0 * n_correct / n


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    param = Params(
        trnpat=args.n_train,
        tenpat=args.n_test,
        nepoch_usup=args.epochs_usup,
        nepoch_sup=args.epochs_sup,
    )

    print("\nBCPNNSim MNIST")
    print(f"architecture : {param.Hin}x{param.Min} -> {param.Hhid}x{param.Mhid} -> {param.Hut}x{param.Mut}")
    print(f"train        : {param.trnpat}")
    print(f"test         : {param.tenpat}")

    mnist = MNISTDataset()
    model = BCPNNMNISTModel(param)

    train_model(model, mnist)
    

    train_acc = evaluate(model, mnist, "train", param.trnpat)
    test_acc = evaluate(model, mnist, "test", param.tenpat)

    plot_receptive_fields(model, save_dir=r"figures")
    plot_filters(model, save_dir="figures")
    print(f"saved figures to: figures")
    
    print("\nResults")
    print(f"train accuracy : {train_acc:.2f}%")
    print(f"test accuracy  : {test_acc:.2f}%")
    print("\nFin.")


if __name__ == "__main__":
    main()    

