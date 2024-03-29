DATASETS=("cora" "citeseer")
data_length=${#DATASETS[@]}

# Experiment: effects of noise to aggregation strategies
NOISES=(0 2 4)
noise_length=${#NOISES[@]}

for ((i=0;i<$data_length;i++))
do
    for ((j=0;j<$noise_length;j++))
    do
        python test.py --dataset "${DATASETS[$i]}" --noise_num ${NOISES[$j]} \
            --task_name "aggregation" -n "gcn" "sgc" "gin"
    done
done

