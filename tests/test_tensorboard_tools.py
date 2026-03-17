import torch

from tools.tensorboard_tools import (
    log_model_histograms,
    log_scalar_group,
    should_log_histograms,
    train_global_step,
)


class RecordingWriter:
    def __init__(self):
        self.scalars = []
        self.histograms = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def add_histogram(self, tag, value, step):
        self.histograms.append((tag, tuple(value.shape), step))


def test_log_scalar_group_writes_prefixed_tags():
    writer = RecordingWriter()

    log_scalar_group(writer, "train", {"step_loss": 1.25, "lr": 1e-3}, 7)

    assert writer.scalars == [
        ("train/step_loss", 1.25, 7),
        ("train/lr", 1e-3, 7),
    ]


def test_should_log_histograms_every_five_epochs():
    assert not should_log_histograms(0)
    assert not should_log_histograms(1)
    assert not should_log_histograms(4)
    assert should_log_histograms(5)
    assert should_log_histograms(10)


def test_log_model_histograms_records_parameters_and_existing_gradients():
    writer = RecordingWriter()
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    parameter.grad = torch.full_like(parameter, 0.5)
    no_grad_parameter = torch.nn.Parameter(torch.zeros(1))

    log_model_histograms(
        writer,
        [
            ("encoder.weight", parameter),
            ("encoder.bias", no_grad_parameter),
        ],
        step=5,
    )

    assert ("params/encoder/weight", (2, 3), 5) in writer.histograms
    assert ("grads/encoder/weight", (2, 3), 5) in writer.histograms
    assert ("params/encoder/bias", (1,), 5) in writer.histograms
    assert ("grads/encoder/bias", (1,), 5) not in writer.histograms


def test_train_global_step_offsets_epochs_by_loader_length():
    assert train_global_step(1, 1, 100) == 1
    assert train_global_step(1, 10, 100) == 10
    assert train_global_step(2, 1, 100) == 101
    assert train_global_step(3, 5, 100) == 205
