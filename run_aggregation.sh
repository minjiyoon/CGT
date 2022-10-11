set -o errexit

DATASETS=("amazon_computer")
WIDTHS=(20)
data_length=${#DATASETS[@]}

# Experiment 1: effects of noise to aggregation strategies
NOISES=(0)
BATCH_SIZES=(16 8 4)
noise_length=${#NOISES[@]}
for ((i=0;i<$data_length;i++))
do
    for ((j=0;j<$noise_length;j++))
    do
        python test.py --dataset "${DATASETS[$i]}" --batch_size ${BATCH_SIZES[$j]} \
            --subgraph_sample_num ${WIDTHS[$i]} --sample_num ${WIDTHS[$i]}  \
            --exp_name "Aggregation" \
            --task_name "aggregation" -n "gcn" "sgc" "gin" "gat" \
            --noise_num ${NOISES[$j]} \
            --org_code --self_connection
    done
done

