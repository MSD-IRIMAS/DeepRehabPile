#!/bin/bash

task="classification"
info_archive="DeepRehabPileForExperiments.json"

# Check if the JSON file exists
if [ ! -f "$info_archive" ]; then
  echo "File $info_archive not found"
  exit 1
fi

list_of_estimators=("FCN" "H_Inception" "LITEMV" "ConvTran" "ConvLSTM" "GRU" "VanTran" "DisjointCNN" "STGCN")

prefix_output_dir=" output_dir="
prefix_root_path=" root_path="

prefix_dataset_name=" dataset_name="
prefix_fold_number=" fold_number="

prefix_task=" task="
prefix_estimator=" estimator="
prefix_train_estimator=" train_estimator="

prefix_epochs=" epochs="
prefix_batch_size=" batch_size="
prefix_runs=" runs="

output_dir="results"
root_path="/home/myuser/code/datasets/"
train_estimator="true"
epochs=1500
batch_size=64
runs=5

# Read and loop over the specified task in the JSON file
jq -c ".$task[]" "$info_archive" | while read -r entry; do
  # Extract dataset name and number of folds
  dataset_name=$(echo "$entry" | jq -r '.[0]')
  number_of_folds=$(echo "$entry" | jq -r '.[1]')

  for ((fold_number=0; fold_number<number_of_folds; fold_number++)); do

    for estimator in "${list_of_estimators[@]}"; do

        running_experiment_name=$task"_"$dataset_name"_"$fold_number"_"$estimator
        args=$running_experiment_name
        args=$args$prefix_output_dir$output_dir
        args=$args$prefix_root_path$root_path
        args=$args$prefix_task$task
        args=$args$prefix_dataset_name$dataset_name
        args=$args$prefix_fold_number$fold_number
        args=$args$prefix_estimator$estimator
        args=$args$prefix_train_estimator$train_estimator
        args=$args$prefix_epochs$epochs$prefix_batch_size$batch_size$prefix_runs$runs

        chmod +x run_job.sh
        bash run_job.sh $args

    done
  done
done
