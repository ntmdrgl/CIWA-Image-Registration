import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Optimization Algorithm Models
# - Grey Wolf Optimizer (GWO)
# - Whale Optimization Algorithm (WOA)
# - Backpropagation

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()