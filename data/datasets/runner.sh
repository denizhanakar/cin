# datasets=('texas' 'wisconsin' 'film' 'squirrel' 'chameleon' 'cornell' 'citeseer' 'pubmed' 'cora')
# models=('BundleHansenGebhart' 'ChebNet' 'ChebNetBundle' 'SIGN' 'SIGNBundle')
# models=('BundleHansenGebhart' 'ChebNetBundle' 'SIGNBundle')
# models=('SIGNBundle' 'SIGN')


# models=('gcwn-full' 'gcwn-equiv-full' 'cwn-full')
models=('gcwn' 'gcwn-equiv' 'cwn')
# models=('cwn')

for model in "${models[@]}"; do
    printf "CURRENTLY RUNNING $model COMPLETE\n"
    ./exp/scripts/qm9/complete/$model.sh
    # printf "CURRENTLY RUNNING $model DEFAULT\n"
    # ./exp/scripts/qm9/default/$model.sh
done

# modelalt=('gcwn' 'cwn')

# for model in "${modelalt[@]}"; do
#     printf "CURRENTLY RUNNING $model COMPLETE\n"
#     ./exp/scripts/qm9/complete/$model.sh
#     # printf "CURRENTLY RUNNING $model DEFAULT\n"
#     # ./exp/scripts/qm9/default/$model.sh
# done


# for model in "${models[@]}"; do
#     for dataset in "${datasets[@]}"; do
#         python exp/run.py --model "$model" --dataset "$dataset"
#     done
# done
