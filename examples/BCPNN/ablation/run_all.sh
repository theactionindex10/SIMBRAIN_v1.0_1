#!/bin/bash
cd ~/SIMBRAIN_BCPNN/examples/BCPNN
run() { name=$1; shift; echo "=== $name start $(date +%H:%M:%S)" >> ablation/progress.txt; OMP_NUM_THREADS=6 python3 -u Model_to_BCPNN_ablation.py "$@" > ablation/$name.log 2>&1; echo "=== $name done  $(date +%H:%M:%S) exit $?" >> ablation/progress.txt; }
run A_Pi_only        --memristor_inference 1 --mem_traces Pi
run B_Pj_only        --memristor_inference 1 --mem_traces Pj
run C_Pji_only       --memristor_inference 1 --mem_traces Pji
run D_all_noswap     --memristor_inference 1 --updconnint 0
run E_golden_noswap  --memristor_inference 0 --updconnint 0
echo "ALL DONE $(date +%H:%M:%S)" >> ablation/progress.txt
