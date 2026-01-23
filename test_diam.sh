export CUBLAS_WORKSPACE_CONFIG=:16:8 # to eliminate LSTM non-deterministic

if [[ "$1" == "--link_only" ]]
then
    prefix=diam_checkpoints
else
    prefix=diam_checkpoints
fi

GPU_ID=0

for seed in 1
do
  for lr in 2e-5
  do
    for epochs in 6
    do
        python main.py --encoder_model "$prefix"/"$seed"_"$lr"_"$epochs"/ \
          --seed $seed \
          --gpu $GPU_ID \
          --data_dir data/diam \
          --max_num_test_contexts 20 \
          --eval \
          --fp16 \
          --output_dir data/diam/test_predictions.json
          $1
    done
  done
done
