from registration.utils import store_transform

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# vvv Manually change the parameters below vvv
parameters = {
    "angle":         0,
    "shear_x":       0,
    "shear_y":       0,
    "scale_x":       1,
    "scale_y":       1,
    "translation_x": 0,
    "translation_y": 0,
    "k1":            0,
    "k2":            0,
    "k3":            0,
    "p1":            0,
    "p2":            0,
}

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    store_transform(
        name="manual_transform",
        source="manual",
        metric=cfg.optimization.metric,
        best_metric=None,
        parameters=parameters,
        config=cfg,
    )

if __name__ == "__main__":
    main()
