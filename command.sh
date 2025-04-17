cd ./src/GAMMA
time python d_m.py   --fitness1 latency   --fitness2 power   --l1_size 512   --l2_size 108000   --NocBW 81920000   --epochs 100   --model vgg16   --singlelayer 1   --log_level 0  --num_pe 168
cd ../../