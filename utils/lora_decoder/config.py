from dataclasses import dataclass


@dataclass(frozen=True)
class LoRaConfig:
    sf: int
    bw: float
    fs: float
    preamble_symbols: int = 8
    implicit_header: bool = False
    coding_rate: int = 1
    has_crc: bool = True
    reduced_rate: bool = False

    def __post_init__(self) -> None:
        if not (6 <= self.sf <= 12):
            raise ValueError("sf must be in [6, 12]")
        if self.coding_rate not in (1, 2, 3, 4):
            raise ValueError("coding_rate must be 1..4 (4/5..4/8)")
        if self.bw <= 0 or self.fs <= 0:
            raise ValueError("bw and fs must be positive")

    @property
    def num_bins(self) -> int:
        return 1 << self.sf

    @property
    def num_bins_header(self) -> int:
        return 1 << (self.sf - 2)

    @property
    def symbol_samples(self) -> int:
        return int(round(self.fs * (2**self.sf) / self.bw))

    @property
    def pause_after_sfd_samples(self) -> int:
        return self.symbol_samples + self.symbol_samples // 4
