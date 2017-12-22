# train lstm
python train.py --network lstm --num-label 2 --num-lstm-layer 1 --train-path data/train.dat --test-path data/test.dat --vocab-path data/vocab.txt --model-prefix model/m1 --gpus 0,1,2,3 --num-epoch 2

# train bi-lstm
python train.py --network bilstm --num-label 2 --num-lstm-layer 2 --train-path data/train.dat --test-path data/test.dat --vocab-path data/vocab.txt --model-prefix model/m2 --gpus 0,1,2,3 --num-epoch 2
