#!/bin/bash

root_path="https://maxime-devanne.com/datasets/RehabPile/"

list_of_dataset_names_origin=('EHE_reg' 'IRDS_clf_bn' 'KERAAL_clf_bn' 'KERAAL_clf_mc' 'KIMORE_clf_bn' 'KIMORE_reg' 'KINECAL_clf_bn' 'SPHERE_clf_bn' 'UCDHE_clf_bn' 'UCDHE_clf_mc' 'UIPRMD_clf_bn' 'UIPRMD_reg')

mkdir -p "classification/"
mkdir -p "regression/"

target_task=""

for dataset_origin in "${list_of_dataset_names_origin[@]}"; do

    if [[ $dataset_origin == *"clf"* ]]; then
        target_task="classification"
    fi

    if [[ $dataset_origin == *"reg"* ]]; then
        target_task="regression"
    fi

    subfolders=$(wget -q -O - $root_path$dataset_origin | grep -oP '(?<=href=")[^"]*(?=/")')

    for folder in $subfolders; do

        if [[ "$folder" == "/datasets/RehabPile" ]]; then
            continue
        fi

        new_dataset_name=$dataset_origin"_"$folder
        new_path_download=$target_task"/"$new_dataset_name

        included_dir="/datasets/RehabPile/"$dataset_origin"/"$folder

        wget -r -np -nH --cut-dirs=4 -R "index.html*" $folder -P "$new_path_download" -I $included_dir $root_path$dataset_origin"/"$folder
        rm $new_path_download"/"$folder
        echo $new_dataset_name

    done

done
