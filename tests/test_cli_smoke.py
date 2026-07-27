from dccnn_arpes.cli import data, denoise, train
from dccnn_arpes.cli import eval as eval_cli


def test_cli_modules_expose_main():
    assert callable(data.main)
    assert callable(train.main)
    assert callable(eval_cli.main)
    assert callable(denoise.main)
