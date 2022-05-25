"""Configure of experiment."""
from typing import Dict, Text

import attr


@attr.s(auto_attribs=True)
class ExperimentalParams:
  """
  Arrtributes:
    learning_rate: int, learning rate. Defaul is 0.001.
    epochs: int, epochs for model training. Default is 100.
    optimizer: str, name of optimizer for model. Default is `adam`.
    loss: str, name of loss function. Default is `mse`.
    metrics: str, name of metrics to evaluate model. Default is `mae`.
    batch_size: int, batch size. Default is 32.
  """
  learning_rate: float = 0.0001
  epochs: int = 1
  optimizer: Text = 'adam'
  loss: Text = 'mse'
  metrics: Text = 'mae'
  batch_size: int = 32

  @classmethod
  def from_dict(cls, experimental_parameters: Dict):
    """extracting values from model.
    Args:
        query_set (Dict): query set from model.
    Returns:
        db_config: initalized experimental_parameters.
    """
    return cls(
        learning_rate=experimental_parameters['learning_rate'],
        epochs=experimental_parameters['epochs'],
        optimizer=experimental_parameters['optimizer'],
        loss=experimental_parameters['loss'],
        metrics=experimental_parameters['metrics'],
        batch_size=experimental_parameters['batch_size'],
    )
