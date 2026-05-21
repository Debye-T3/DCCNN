"""Preview image generator with k-space conversion support."""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


K_CONSTANT = 0.5123  # sqrt(2m_e) / hbar in eV^{-1/2} * A^{-1}


def compute_contrast(data: np.ndarray, pmin: float, pmax: float) -> Tuple[float, float]:
    """Percentile-based contrast limits."""
    if data.size == 0:
        return 1e-6, 1.0
    positive = data[data > 0]
    if positive.size == 0:
        positive = np.abs(data.ravel())
    vmin = float(np.percentile(positive, pmin)) if positive.size else 0.0
    vmax = float(np.percentile(positive, pmax)) if positive.size else 1.0
    if vmax <= vmin:
        vmax = float(positive.max()) if positive.size else 1.0
        vmin = max(vmax * 1e-3, 1e-6)
    return vmin, vmax


def to_kspace(
    energy_axis: np.ndarray,
    angle_axis: np.ndarray,
    hv: float,
    work_function: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert angle axis to k-parallel momentum axis.

    k_parallel [A^{-1}] = K_CONSTANT * sqrt(E_kin) * sin(theta)
    where E_kin = hv - work_function - E_binding
    """
    if hv is None or hv <= 0:
        raise ValueError("Photon energy (hv) is required for k-space conversion.")

    if energy_axis.size > 1 and energy_axis[-1] < energy_axis[0]:
        e_kin_1d = energy_axis.astype(np.float64)
    else:
        e_kin_1d = hv - work_function - energy_axis.astype(np.float64)

    e_kin_ref = float(np.median(np.clip(e_kin_1d, 0.01, None)))
    if e_kin_ref < 0.01:
        e_kin_ref = 0.01
    theta_rad = np.radians(angle_axis.astype(np.float64))
    k_parallel = K_CONSTANT * np.sqrt(e_kin_ref) * np.sin(theta_rad)

    return k_parallel.astype(np.float32), energy_axis


def generate_preview(
    spectrum: np.ndarray,
    energy_axis: np.ndarray,
    angle_axis: np.ndarray,
    destination: Path,
    *,
    cmap: str = "inferno",
    pmin: float = 1.0,
    pmax: float = 99.5,
    use_log: bool = True,
    use_kspace: bool = False,
    hv: Optional[float] = None,
    work_function: float = 4.2,
) -> None:
    """Generate a preview PNG of the ARPES spectrum."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(spectrum, a_min=0.0, a_max=None)

    x_axis = angle_axis.copy()
    x_label = "Angle [deg]"

    if use_kspace:
        if hv is None or hv <= 0:
            raise ValueError("Photon energy (hv) is required for k-space preview.")
        k_axis, e_axis = to_kspace(energy_axis, angle_axis, hv, work_function)
        x_axis = k_axis
        x_label = r"$k_{\parallel}$ [$\AA^{-1}$]"
    else:
        e_axis = energy_axis

    extent = [
        float(x_axis[0]), float(x_axis[-1]),
        float(e_axis[0]), float(e_axis[-1]),
    ]

    norm = None
    if use_log:
        vmin, vmax = compute_contrast(data, pmin, pmax)
        norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7, 5))
    kwargs = {"origin": "lower", "aspect": "auto", "cmap": cmap, "extent": extent}
    if norm is not None:
        kwargs["norm"] = norm
    im = ax.imshow(data, **kwargs)
    fig.colorbar(im, ax=ax)
    title = "ARPES Spectrum"
    if use_log:
        title += " (log scale)"
    if use_kspace:
        title += " — k-space"
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Energy [eV]")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
