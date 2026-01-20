dataset="Amazon_Baby"
models=("FAESRec") # model

for model in ${models[@]}
do
    echo "model：$model"
    python run_seq.py --dataset ${dataset} \   --model ${model} \ --gpu_id ${gpu_id} \
    
done
