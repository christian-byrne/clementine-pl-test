import json
from pathlib import Path

from load.model import ModelLoader
from util.device.device_config import TorchDevice

import cProfile
from snakeviz.cli import main as snakeviz_main


PROFILE_FILENAME = "stats.prof"
RUN_WITH_PROFILE = False


def profile(func):
    def wrapper(*args, **kwargs):
        if not RUN_WITH_PROFILE:
            return func(*args, **kwargs)

        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **kwargs)
        pr.disable()

        stats_filename = PROFILE_FILENAME
        pr.dump_stats(stats_filename)

        snakeviz_main([stats_filename])

        return retval

    return wrapper


@profile
def main():
    device = TorchDevice.get()
    model_path = Path(json.load(open("models.json"))["test"])

    model = ModelLoader(model_path)
    x = model.generate_image("A green pasture with a blue sky.")
    model.open_image_outputs(x)


main()
