from typing import Dict
from autoencoders.models import *


class ModelFactory():
    """Factory class to initialize AutoEncoder models

    Raises:
        ValueError: raise ValueError when the given config doesn't include proper 'model_type'

    Returns:
        _type_: _description_
    """
    @staticmethod
    def create_model(config: Dict):
        """factory method, which is responsible for initializing a model.

        Args:
            config (Dict): wandb config

        Raises:
            ValueError: raise ValueError when the given config doesn't include proper 'model_type'

        Returns:
            nn.Module: requested AutoEncoder model. Every model would be initialized in CPU only
        """
        model_type = config["model_type"]
        if model_type == "classic-AE":
            model = AutoEncoder.from_config(config)
        elif model_type == "conv-AE":
            model = ConvAE.from_config(config)
        else:
            raise ValueError(
                "Unknown model type; check 'model_type' in config file")

        return model
