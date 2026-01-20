
model="FAESRec"

#hyperparameters
for n_layer in 1 2 3 4
do

    CUDA_VISIBLE_DEVICES=2 python run_seq.py --dataset Amazon_Baby \   --model ${model} \ --n_layers ${n_layer}

done