from typing import Dict, Iterable, Tuple


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


def train_global_step(epoch: int, step: int, steps_per_epoch: int) -> int:
    return max(epoch - 1, 0) * max(steps_per_epoch, 0) + step
