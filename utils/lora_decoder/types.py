from dataclasses import dataclass
from typing import Optional


@dataclass
class SyncResult:
    packet_start: int
    sfd_pos: int
    header_start: int
    score: float


@dataclass
class LoRaHeader:
    length: int
    coding_rate: int
    has_crc: bool
    raw: bytes


@dataclass
class LoRaFrame:
    start_index: int
    header: LoRaHeader
    payload: bytes
    mac_crc: Optional[int]
    snr_db: float
