import pytest

from dccnn_arpes.cli import data, denoise, train
from dccnn_arpes.cli import eval as eval_cli


def test_cli_modules_expose_main():
    assert callable(data.main)
    assert callable(train.main)
    assert callable(eval_cli.main)
    assert callable(denoise.main)


def test_train_cli_requires_explicit_config():
    """Starting with implicit defaults must not create an untraceable experiment."""
    with pytest.raises(SystemExit):
        train.main([])
