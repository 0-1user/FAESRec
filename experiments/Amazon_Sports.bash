## baseline "SIGMA" "EchoMamba4Rec" "Mamba4Rec" "LinRec" "CFIT4SRec" "FEARec" "FMLPRec" "SASRec" "GRU4Rec" "BERT4Rec"
##datasets:Amazon_Beauty Amazon_Sports Amazon_Toys Amazon_Clothing ml-1m Amazon_Video_Games
gpu_id=0    #gpu id 更改
dataset="Amazon_Sports"
models=("SIGMA" "EchoMamba4Rec" "Mamba4Rec" "LinRec" "CFIT4SRec" "FEARec" "FMLPRec" "SASRec" "GRU4Rec" "BERT4Rec")

for model in ${models[@]}
do
    echo "当前模型：$model"
    python run_seq.py --dataset ${dataset} \   --model ${model} \ --gpu_id ${gpu_id} \
    
done