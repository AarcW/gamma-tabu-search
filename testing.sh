cd ./src/GAMMA
python d_m.py --fitness1 latency --fitness2 power --num_pe 16 --l1_size 128 --l2_size 1024 --model vgg16 --singlelayer 1 --epochs 10 --log_level 0
cd ../../
