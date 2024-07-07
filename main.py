import json
from pathlib import Path

from load.model.load_model import ModelLoader
from util.device.device_config import TorchDevice


device = TorchDevice.get()

with json.load(open("models.json")) as config:
    model_path = Path(config[0])

model = ModelLoader(model_path)
x = model.generate_image("A green pasture with a blue sky.")
model.open_image_outputs(x)
