[TOC]

# What you need to know before execution 


This folder contains all the relevant scripts, log files of runs, figures, outputs, and scripts for visualization.

This script is an integration of the SIMBRAIN framework into the BCPNNSim codebase. 

This includes two separate repositories, namely a forked version of the BCPNNSim2.1 (a private repository maintained by the author of the BCPNN-SIMBRAIN integration) and SIMBRAIN_v1.0_1. 

The BCPNNSim2.1_memristor includes all the P-traces (Pi, Pj, Pji) being updated by way of voltage pulses applied to memristors to simulate trace behavior. This is done in C++, and can be viewed if the user runs the following commands:


```bash
cd BCPNNSim2.1_memristor/libsrc_cu/src/

vim Prj.cpp Prj_cu.cu
```
The example code in SIMBRAIN used to run the BCPNN is linked to the C++ code through python - C++ bindings.

Before the updated SIMBRAIN repository is cloned, the BCPNNSim2.1 code with the memristor inference needs to be cloned first. 

### Clone the BCPNNSim2.1 repository 

```bash
git clone https://github.com/theactionindex10/BCPNNSim2.1.git BCPNNSim2.1_memristor
``` 

Make sure the current branch is the required one. 

```bash
cd BCPNNSim2.1_memristor
git switch memristor_traces
git branch  
```

The branch must be memristor_traces. This is the branch where all the relevant script changes that allow for memristor inference through the BCPNNSim2.1 C++ engine 


```bash
  master
* memristor_traces
  varun-2026 
```

While still in BCPNNSim2.1_memristor/, run the following commands

```bash
cd python_api/
make -f Makefile.cuda   #or: make -f Makefile.cpu, if running on a CPU. This README presupposes a CUDA deployment
```


The README.md file present in this repo must be read before proceeding to the SIMBRAIN implementation. The user will need to have BCPNNSim2.1_memristor running on their system before implementing the SIMBRAIN example. 

### Clone the SIMBRAIN repository

NOTE: It is good practice to keep the two repositories: BCPNNSim2.1_memristor and SIMBRAIN in the same main directory, but separate from each other. They SHOULD NOT BE NESTED IN ONE ANOTHER!


```bash
git clone https://github.com/theactionindex10/SIMBRAIN_v1.0_1.git 

```

Make sure the branch is once again the right one.

```bash
cd SIMBRAIN_v1.0_1
#Obtain all the relevant branches
git switch python-pipeline 
git switch user_working
git switch master
git branch

```


```bash
* master
  python-pipeline
  user_working
```


### Navigate to the relevant directory

The folder contains a simbrain/ folder. This does not contain the relevant mapping.py module that will be used for the BCPNN implementation. The relevant module is found in examples/BCPNN/simbrain/.

Run the following commands while in SIMBRAIN_v1.0_1/. It is a sample command, that can be modified as per the user's preferences.

```bash
cd examples/BCPNN
python3 -u Model_to_BCPNN.py --memristor_inference 1 --ADC_precision 20 --seed 1 --dac_bits 10 2>&1 | tee log_files/example_run.log
```

NOTE: This will produce a run with memristor inference, DAC bits, ADC precision, and a user input seed. --memristor_inference 0 would produce a golden run i.e., a run of the python_api/pyMNIST/run.py in BCPNNSim2.1 without any memristor_inference.

The following parameters will allow the user to observe the variation in accuracy of the BCPNN for the given experiment detailed in BCPNNSim2.1. One of these is repurposed from SIMBRAIN, namely --ADC_precision and it proves to be crucial to the accuracy of the network.

NOTE: MNIST is downloaded into Data/mnist on the first run. The script needs torch, numpy and matplotlib; building BCPNNSim2.1_memristor needs pybind11.


|        Parameter        |                          Definition                          |  Default   |
| :---------------------: | :----------------------------------------------------------: | :--------: |
|    --memristor_inference|                Memristor inference toggle                    |      0     |
|    --ADC_precision      |                The ADC resolution: int (2-32)                |     16     |
|    --adc_range          |           Input range of the ADC for reading:float (0.0-1.0) |     0.2    |
|    --seed               |                Seed to the network                           |  88192386  |
|    --dac_bits           |                DAC bits for writing: int (0-12)              |0=continuous|
|    --pulse_mode         |                Pulse mode scheme for voltage generation      |     gap    |


Sample output 

```bash
BCPNNSim MNIST
architecture : 784x2 -> 30x100 -> 1x10
train        : 1000
test         : 1000
seed         : 1
memristor traces inside the engine: ihprj.memristor = True | pulse mode gap | ADC 20 bits, DAC 10 bits, full scale {'Pi': 1.0, 'Pj': 0.2, 'Pji': 0.2} | bands {'Pi': 3, 'Pj': 3, 'Pji': 3}

Training

Unsupervised
  epoch 1/5 | simstep   3000 | swapped  1058
  epoch 2/5 | simstep   6000 | swapped  2548
  epoch 3/5 | simstep   9000 | swapped  3959
  epoch 4/5 | simstep  12000 | swapped  5042
  epoch 5/5 | simstep  15000 | swapped  5644

Supervised
  epoch 1/5 | simstep  18000
  epoch 2/5 | simstep  21000
  epoch 3/5 | simstep  24000
  epoch 4/5 | simstep  27000
  epoch 5/5 | simstep  30000
saved figures to: figures

Results
train accuracy : 99.30%
test accuracy  : 88.80%

Fin.

``` 

The code is still being refined and debugged. As of now, it runs for an ideal memristor. Runs have not been conducted with the memristor non-idealities added in. For any queries or critiques of the code contact : varunk@kth.se

