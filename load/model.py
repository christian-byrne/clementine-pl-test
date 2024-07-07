import torch
import safetensors
from diffusers import StableDiffusionPipeline

from pathlib import Path
from util.logging.log import Logger
from load.load_location import LoadLocation

from typing import Union

logger = Logger(__name__)()


class ModelLoader:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = model_path

        self._is_safetensor = None
        self.diffusion_pipeline = None

    def generate_image(self, prompt: str):
        with torch.autocast("cuda"):
            pipeline = self.get_diffusion_pipeline()
            images = pipeline(prompt, height=512, width=512, num_inference_steps=16)

        logger.info(f"Generated image with prompt: {prompt}")
        logger.info(f"Generated images: {images}")
        return images

    def open_image_outputs(self, image_outputs):
        for image in image_outputs["images"]:
            image.show()

    def is_safetensors(self, model_path: Path):
        if self._is_safetensor is not None:
            return self._is_safetensor

        # magic_number = b"\x53\x44"  # Define as bytes with correct order
        # with open(model_path, "rb") as f:
        #     first_two_bytes = f.read(2)
        #     res = first_two_bytes == magic_number  # Cache result
        #     self.__is_safetensor = res
        if model_path.suffix == ".safetensors":
            res = True
            self._is_safetensor = res

        logger.info(f"{model_path} is a SafeTensors model: {res}")
        return res

    def release_model(self):
        self.model = None

    def load_model(self) -> None:
        if self.is_safetensors(self.model_path):
            return safetensors.safe_open(
                self.model_path, "pt", LoadLocation().get_without_index()
            )
        self.model = torch.load(self.model_path, map_location=LoadLocation().get)

    def get_model(self) -> Union[torch.nn.Module, None]:
        return self.model

    def get_diffusion_pipeline(self) -> StableDiffusionPipeline:
        if self.diffusion_pipeline is not None:
            return self.diffusion_pipeline

        self.diffusion_pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path="SG161222/Realistic_Vision_V2.0",
            local_files_only=False,
            safety_checker=None,
            requires_safety_checker=False,
        )
        return self.diffusion_pipeline
