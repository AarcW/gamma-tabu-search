cd ./src/GAMMA
python d_m.py --fitness1 latency --fitness2 power --num_pe 168 --l1_size 512 --l2_size 108000 --model vgg16 --singlelayer 1 --epochs 1
cd ../../
