"""
Loads an object saved with torch.save from a file.

torch.load uses Python's unpickling facilities but treats storages, which underlie tensors, specially. They are first deserialized on the CPU and are then moved to the device they were saved from. If this fails (e.g. because the run time system doesn't have certain devices), an exception is raised. However, storages can be dynamically remapped to an alternative set of devices using the map_location argument.

"""

import torch

from util.logging.log import Logger
from util.device.device_config import TorchDevice

from typing import Union, List, Callable, Dict

logger = Logger(__name__)()


class LoadLocation:
    def __init__(self):
        self.current_load_destination = TorchDevice.get()
        logger.info(f"Currently loading models to: {self.current_load_destination}")

    def get(self):
        return self.current_load_destination

    def get_without_index(self):
        return str(self.get()).split(":")[0]

    def set(self, device: Union[torch.device, Callable, str, Dict[str, torch.device]]):
        """
        Args:
            device: Union[torch.device, Callable, str, Dict[str, torch.device]]: a function, :class:`torch.device`, string or a dict specifying how to remap storage
            locations
        """
        if device is None:
            self.current_load_destination = TorchDevice.get()
        else:
            self.current_load_destination = device

        logger.info(f"Set load location to: {self.current_load_destination}")
