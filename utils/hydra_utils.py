from typing import Optional
import os
from omegaconf import OmegaConf


def _dict_to_str(d: dict) -> str:
    """
    Convert a dictionary to a string without quotes.
    """
    output = "{"
    for key, value in d.items():
        if value is None:
            value = "null"
        output += (
            f"{key}: {_dict_to_str(value) if isinstance(value, dict) else value}, "
        )
    output = output[:-2] + "}"
    return output


def _yaml_to_cli(
    yaml_path: str,
    prefix: Optional[str] = None,
) -> list[str]:
    """
    Convert a yaml file to a list of command line arguments.
    """
    cfg = OmegaConf.load(yaml_path)
    cli = []
    for key, value in OmegaConf.to_container(cfg).items():
        if value is None:
            value = "null"
        cli.append(
            f"++{prefix + '.' if prefix else ''}{key}={_dict_to_str(value) if isinstance(value, dict) else value}"
        )
    return cli


def unwrap_shortcuts(
    argv: list[str],
    config_path: str,
    config_name: str,
) -> list[str]:
    """
    Unwrap shortcuts by replacing them with commands from corresponding yaml files.
    All shortcuts should be in the form of `@shortcut_name`.
    Example:
    - @latent -> unwrap configurations/shortcut/latent/base.yaml and configurations/shortcut/latent/dataset_name.yaml
    - @mit_vision/h100 -> unwrap configurations/shortcut/mit_vision/h100.yaml
    """
    # find the default dataset
    defaults = OmegaConf.load(f"{config_path}/{config_name}.yaml").defaults
    dataset = next(default.dataset for default in defaults if "dataset" in default)
    # check if dataset is overridden
    for arg in argv:
        if arg.startswith("dataset="):
            dataset = arg.split("=")[1]

    if dataset is None:
        raise ValueError("Dataset name is not provided.")

    new_argv = []
    for arg in argv:
        if arg.startswith("@"):
            shortcut = arg[1:]
            base_path = f"{config_path}/shortcut/{shortcut}/base.yaml"

            if os.path.exists(base_path):
                new_argv += _yaml_to_cli(base_path)
                dataset_path = f"{config_path}/shortcut/{shortcut}/{dataset}.yaml"
                if os.path.exists(dataset_path):
                    new_argv += _yaml_to_cli(dataset_path)
            else:
                default_path = f"{config_path}/shortcut/{shortcut}.yaml"
                if os.path.exists(default_path):
                    new_argv += _yaml_to_cli(default_path)
                else:
                    raise ValueError(f"Shortcut @{shortcut} not found.")
        elif arg.startswith("algorithm/backbone="):
            # this is a workaround to enable overriding the backbone in the command line
            # otherwise, the backbone could be re-overridden by
            # the backbone cfgs in dataset-experiment dependent cfgs
            new_argv += override_backbone(arg[19:])
        else:
            new_argv.append(arg)

    return new_argv


def override_backbone(name: str) -> list[str]:
    """
    Override the backbone with the specified name.
    """
    return ["algorithm.backbone=null"] + _yaml_to_cli(
        f"configurations/algorithm/backbone/{name}.yaml", prefix="algorithm.backbone"
    )
