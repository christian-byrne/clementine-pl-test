import torch

from util.logging.log import Logger

from typing import Union, List

logger = Logger(__name__)()


class TorchDevice:
    DEVICE: Union[torch.device, None] = None

    @classmethod
    def get(cls):
        if cls.DEVICE:
            return cls.DEVICE

        if torch.cuda.is_available():
            device_index = 0

            if torch.cuda.device_count() > 1:
                for i in range(torch.cuda.device_count()):
                    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
                device_index = int(input("Select device: "))
            cls.DEVICE = torch.device(f"cuda:{device_index}")
        else:
            cls.DEVICE = torch.device("cpu")

        logger.info(f"Using device: {cls.DEVICE}")
        return cls.DEVICE

    @classmethod
    def get_without_index(cls):
        return str(cls.get()).split(":")[0]
