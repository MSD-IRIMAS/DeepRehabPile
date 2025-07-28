> ⚠️ **Alert:** If you are using this code with **Keras v3**, make sure you are using **Keras ≥ 3.6.0**.
> Earlier versions of Keras v3 do not honor `trainable=False`, which will result in **training hand-crafted filters** in **LITEMV** and **H-Inception** unexpectedly.

# Deep Learning for Skeleton Based Human Motion Rehabilitation Assessment: A Benchmark

Authors: [Ali Ismail-Fawaz](https://hadifawaz1999.github.io/)<sup>1,†</sup>, [Maxime Devanne](https://maxime-devanne.com/)<sup>1,†</sup>, [Stefano Berreti](https://www.micc.unifi.it/berretti/)<sup>2</sup>, [Jonathan Weber](https://www.jonathan-weber.eu/)<sup>1</sup> and [Germain Forestier](https://germain-forestier.info/)<sup>1,3</sup>

<sup>†</sup> These authors contributed equally to this work<br>
<sup>1</sup> IRIMAS, Universite de Haute-Alsace, France<br>
<sup>2</sup> MICC, University of Florence, Italy<br>
<sup>3</sup> DSAI, Monash University, Australia

This repository is the source code of the article titled "[Deep Learning for Skeleton Based Human Motion Rehabilitation Assessment: A Benchmark]()".
In this article, we present a benchmark comparison between nine different deep learning architectures on for Skeleton Based Human Rehabilitation Assessment.
This archive contains 39 classification datasets and 21 extrinsic regression datasets. More details about the dataset information is available on the [article webpage]().

## Abstract

Automated assessment of human motion plays a vital role in rehabilitation, enabling objective evaluation of patient performance and progress.
Unlike general human activity recognition, rehabilitation motion assessment focuses on analyzing the quality of movement within the same action class, requiring the detection of subtle deviations from ideal motion.
Recent advances in deep learning and video-based skeleton extraction have opened new possibilities for accessible, scalable motion assessment using affordable devices such as smartphones or webcams.
However, the field lacks standardized benchmarks, consistent evaluation protocols, and reproducible methodologies, limiting progress and comparability across studies.
In this work, we address these gaps by (i) aggregating existing rehabilitation datasets into a unified archive, (ii) proposing a general benchmarking framework for evaluating deep learning methods in this domain, and (iii) conducting extensive benchmarking of multiple architectures across classification and regression tasks.
All datasets and implementations are released to the community to support transparency and reproducibility.
This paper aims to establish a solid foundation for future research in automated rehabilitation assessment and foster the development of reliable, accessible, and personalized rehabilitation solutions.

## Data

In order to download the 60 datasets of our archive simply use the two following commands when in the root directory of this repository:
```bash
cd datasets
chmod +x get_datasets.sh
./get_datasets.sh
```
This will create two sub-folders under the `datasets` folder: `datasets/classification/` and `datasets/regression/` where the datasets are stored inside.
For each dataset sub-folder, there exists a single `json` file containing the informaiton of the datasets alongside `k` folders for each fold (train-test split) on this specific dataset, where `k` is the number of folds which varries from dataset to another.

## Docker

This repository supports the usage of docker. In order to create the docker image using the [dockerfile](dockerfile), simply run the following command (assuming you have docker installed and nvidia cuda container as well):
```bash
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t deep-rehab-pile-image .
```
After the image has been successfully built, you can create the docker container using the following command:
```bash
docker run --gpus all -it --name deep-rehab-pile-container -v "$(pwd):/home/myuser/code" --user $(id -u):$(id -g) deep-rehab-pile-image bash
```

The code will be stored under the directory `/home/myuser/code/` inside the docker container. This will allow you to use GPU acceleration.

## Requirements

If you do not want to use docker, simply install the project using the following command:
```bash
python3 -m venv ./deep-rehab-pile-venv
source ./deep-rehab-pile-venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

Make sure you have [`jq`](https://jqlang.org/) installed on your system. This project supports `python>=3.11` only.

```
numpy==1.26.4
scikit-learn==1.4.2
aeon==1.0.1
keras==3.6.0
tensorflow==2.16.1
hydra-core==1.3.2
omegaconf==2.3.0
pandas==2.0.3
matplotlib==3.9.0
```

## Running the code on a single experiment

For each experiment, our code runs multiple initialization (default 5) of a model on a single fold of a single dataset, reports the evaluation metrics on each initialization, as well as the ensemble performance of all initializations. The results reported in our article are of the ensemble performance.

If you wish to run a single experiment on a single dataset, on a single fold of this dataset, using a single model then first you have to execute your docker container to open a terminal inside if you're not inside the container:
```bash
docker exec -it deep-rehab-pile-container bash
```
Then you can run the following command for example top run LITEMV on the IRDS_clf_bn_EFL classification dataset on fold number 0:
```bash
python3 main.py task=classification dataset_name=IRDS_clf_bn_EFL fold_number=0 estimator=LITEMV
```
The code uses [hydra](https://hydra.cc/docs/intro/) for the parameter configuration, simply see the [hydra configuration file](config/config_hydra.yaml) for a detailed view on the parameters of our experiments.

## Running the whole benchmark

If you wish to run all the experiments to reproduce the results of our article simply run the following for classification experiments:
```bash
chmod +x run_classification_experiments.sh
nohup ./run_classification_experiments.sh &
```
and the following for regression:
```bash
chmod +x run_regression_experiments.sh
nohup ./run_regression_experiments.sh &
```

## Results

<img id="img-overview"
      src="https://github.com/MSD-IRIMAS/DeepRehabPile/blob/main/static/num-params-plot.png"
      class="interpolation-image"
      style="width: 100%; height: 100%; border: none;"> </img>

## Cite this work

If you use this work please cite the following:
```bibtex
@article{ismail-fawaz2025DeepRehabPile,
  author = {Ismail-Fawaz, Ali and Devanne, Maxime and Berretti, Sefano and Weber, Jonathan and Forestier, Germain},
  title = {Deep Learning for Skeleton Based Human Motion Rehabilitation Assessment: A Benchmark},
  journal={arxiv preprint},
  year = {2025}
}
```

## Acknowledgments

This work was supported by the ANR DELEGATION project (grant ANR-21-CE23-0014) of the French Agence Nationale de la Recherche. The authors would like to acknowledge the High Performance Computing Center of the University of Strasbourg for supporting this work by providing scientific support and access to computing resources. Part of the computing resources were funded by the Equipex Equip@Meso project (Programme Investissements d'Avenir) and the CPER Alsacalcul/Big Data. The authors would also like to thank the creators and providers of the original datasets in our archive.
