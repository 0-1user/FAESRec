## baseline "SIGMA" "EchoMamba4Rec" "Mamba4Rec" "LinRec" "CFIT4SRec" "FEARec" "FMLPRec" "SASRec" "GRU4Rec" "BERT4Rec"
##datasets: Amazon_Sports Amazon_Baby Amazon_Video_Games Amazon_Office_Products
gpu_id=0   
dataset="Amazon_Baby"
models=("FAESRec") # model

for model in ${models[@]}
do
    echo "model：$model"
    python run_seq.py --dataset ${dataset} \   --model ${model} \ --gpu_id ${gpu_id} \
    
done
