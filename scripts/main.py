import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../configs", config_name="config")

def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()