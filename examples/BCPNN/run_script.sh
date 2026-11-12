for s in 1 2 3 4; do
  /usr/bin/time -v python3 -u Model_to_BCPNN_engine.py --memristor_inference 0 --seed $s 2>&1 | tee golden_seed$s.log
  /usr/bin/time -v python3 -u Model_to_BCPNN_engine.py --memristor_inference 1 --ADC_precision 16 --adc_range 0.275 --seed $s 2>&1 | tee mem16_seed$s.log
 /usr/bin/time -v python3 -u Model_to_BCPNN_engine.py --memristor_inference 1 --ADC_precision 20 --adc_range 0.275 --seed $s 2>&1 | tee mem20_seed$s.log
 /usr/bin/time -v python3 -u Model_to_BCPNN_engine.py --memristor_inference 1 --ADC_precision 32 --adc_range 0.275 --seed $s 2>&1 | tee mem20_seed$s.log

done
