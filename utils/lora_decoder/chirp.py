import numpy as np

from .config import LoRaConfig


def upchirp(cfg: LoRaConfig, n_samples: int | None = None) -> np.ndarray:
    n = cfg.symbol_samples if n_samples is None else int(n_samples)
    ts = n / cfg.fs
    t = np.arange(n, dtype=np.float64) / cfg.fs
    return np.exp(1j * 2.0 * np.pi * (-cfg.bw / 2.0 * t + (cfg.bw / (2.0 * ts)) * t**2))


def downchirp(cfg: LoRaConfig, n_samples: int | None = None) -> np.ndarray:
    return np.conj(upchirp(cfg, n_samples))


def dechirp(symbol: np.ndarray, cfg: LoRaConfig) -> np.ndarray:
    return symbol * downchirp(cfg, len(symbol))
