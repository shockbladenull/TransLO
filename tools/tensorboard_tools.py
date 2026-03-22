import os
from typing import Dict, Iterable, Tuple

import numpy as np
from PIL import Image


def should_log_histograms(epoch: int, every_n_epochs: int = 5) -> bool:
    return every_n_epochs > 0 and epoch > 0 and epoch % every_n_epochs == 0


def log_scalar_group(writer, tag_prefix: str, scalars: Dict[str, float], step: int) -> None:
    if writer is None:
        return

    for name, value in scalars.items():
        writer.add_scalar("{}/{}".format(tag_prefix, name), value, step)


def log_model_histograms(
    writer,
    named_parameters: Iterable[Tuple[str, object]],
    step: int,
    parameter_prefix: str = "params",
    gradient_prefix: str = "grads",
) -> None:
    if writer is None:
        return

    for name, parameter in named_parameters:
        tag_name = name.replace(".", "/")
        writer.add_histogram("{}/{}".format(parameter_prefix, tag_name), parameter.detach().cpu(), step)
        if parameter.grad is not None:
            writer.add_histogram("{}/{}".format(gradient_prefix, tag_name), parameter.grad.detach().cpu(), step)


def log_image_file(writer, tag: str, image_path: str, step: int) -> None:
    if writer is None or not os.path.isfile(image_path):
        return

    with Image.open(image_path) as image:
        image_array = np.asarray(image.convert("RGB"))

    writer.add_image(tag, image_array, step, dataformats="HWC")


def log_oxford_route_images(writer, sequence_name: str, output_dir: str, step: int) -> None:
    if writer is None:
        return

    log_image_file(
        writer,
        "oxford_detailed/{}/full_route_path".format(sequence_name),
        os.path.join(output_dir, "full_route_path.png"),
        step,
    )
    log_image_file(
        writer,
        "oxford_detailed/{}/full_route_path_3D".format(sequence_name),
        os.path.join(output_dir, "full_route_path_3D.png"),
        step,
    )


def train_global_step(epoch: int, step: int, steps_per_epoch: int) -> int:
    return max(epoch - 1, 0) * max(steps_per_epoch, 0) + step
