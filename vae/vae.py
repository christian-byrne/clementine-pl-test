from torch import nn

from pathlib import Path
from util.logging.log import Logger
from util.device.device_config import TorchDevice
from load.load_location import LoadLocation

from diffusers.models.unets.unet_2d_blocks import (
    AutoencoderTinyBlock,
    get_down_block,
    get_up_block,
)
from typing import Tuple

logger = Logger(__name__)()


class VAE_Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
    ):
        super().__init__()

        self.convulation_input = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            # device=TorchDevice.get(),
            # dtype=None,
        )
