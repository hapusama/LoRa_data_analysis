from __future__ import annotations

import numpy as np

from .chirp import downchirp
from .config import LoRaConfig


class SymbolDemodulator:
    def __init__(self, cfg: LoRaConfig) -> None:
        self.cfg = cfg
        self._down = downchirp(cfg)

    def demod_symbol(self, symbol: np.ndarray, reduced_rate: bool) -> int:
        mixed = symbol * self._down
        spectrum = np.fft.fft(mixed)
        bin_idx = int(np.argmax(np.abs(spectrum))) % self.cfg.num_bins

        if reduced_rate:
            bin_idx = int(round(bin_idx / 4.0)) % self.cfg.num_bins_header

        word = bin_idx ^ (bin_idx >> 1)
        return int(word)

    def extract_words(self, data: np.ndarray, start: int, n_symbols: int, reduced_rate: bool) -> list[int]:
        ns = self.cfg.symbol_samples
        words: list[int] = []

        for i in range(n_symbols):
            s0 = start + i * ns
            s1 = s0 + ns
            if s1 > len(data):
                break
            words.append(self.demod_symbol(data[s0:s1], reduced_rate=reduced_rate))

        return words
