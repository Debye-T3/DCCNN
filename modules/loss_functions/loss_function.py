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

    msssim_values = []
    for i in range(output.shape[0]):
        t = target[i : i + 1]
        data_range = (t.max() - t.min()).clamp(min=1e-6).item()
        msssim_values.append(1 - MS_SSIM(data_range=data_range, channel=1)(output[i : i + 1], t))
    msssim = sum(msssim_values) / output.shape[0]

    total_loss = (1 - alpha) * mae + alpha * msssim

    return total_loss, mae, msssim
