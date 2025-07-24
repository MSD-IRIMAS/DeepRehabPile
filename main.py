"""Main file to run."""

import hydra
from omegaconf import DictConfig, OmegaConf

from _main_classification import main_classification
from _main_regression import main_regression


@hydra.main(config_name="config_hydra.yaml", config_path="config")
def main(args: DictConfig):
    """Run experiments.

    Main function to run experiments.

    Parameters
    ----------
    args: DictConfig
        The input configuration.

    Returns
    -------
    None
    """
    with open("config.yaml", "w") as f:
        OmegaConf.save(args, f)

    if args.task == "regression":
        main_regression(args=args)
    elif args.task == "classification":
        main_classification(args=args)
    else:
        raise ValueError("No task exist: " + args.task)


if __name__ == "__main__":
    main()
