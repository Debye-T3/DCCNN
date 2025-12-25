import torch
import torch.nn as nn

try:
    from pytorch_msssim import MS_SSIM  # type: ignore
except ModuleNotFoundError:
    class MS_SSIM(nn.Module):  # fallback approximation
        def __init__(self, *_, **__):
            super().__init__()

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            mae = torch.mean(torch.abs(pred - target))
            return torch.clamp(1.0 - mae, 0.0, 1.0)

def loss_function(output, target, alpha):
    mae = nn.L1Loss()(output, target)
    msssim = sum(
        1 - MS_SSIM(data_range=1.0, channel=1)(output[i:i+1], target[i:i+1])
        for i in range(output.shape[0])
    ) / output.shape[0]

    total_loss = (1 - alpha) * mae + alpha * msssim

    return total_loss, mae, msssim
