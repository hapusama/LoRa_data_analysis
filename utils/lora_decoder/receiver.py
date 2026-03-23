from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .coding import BitPipelineDecoder
from .config import LoRaConfig
from .demod import SymbolDemodulator
from .sync import PreambleSynchronizer
from .types import LoRaFrame, LoRaHeader


class LoRaReceiver:
    """高层 LoRa 接收器。

    封装了同步、符号解调和位级译码流水线，
    提供 `decode` / `decode_file` 两个主入口。
    """

    def __init__(
        self,
        config: LoRaConfig,
        detect_threshold: float = 0.75,
        upchirp_corr_threshold: float = 0.20,
    ) -> None:
        self.config = config
        self.sync = PreambleSynchronizer(
            config,
            detect_threshold=detect_threshold,
            upchirp_corr_threshold=upchirp_corr_threshold,
        )
        self.demod = SymbolDemodulator(config)
        self.bitpipe = BitPipelineDecoder()

    @staticmethod
    def _parse_header(decoded_hdr: bytes) -> Optional[LoRaHeader]:
        if len(decoded_hdr) < 3:
            return None

        length = decoded_hdr[0]
        b1 = decoded_hdr[1]
        coding_rate = (b1 >> 5) & 0x07
        has_crc = bool((b1 >> 4) & 0x01)
        coding_rate = min(max(coding_rate, 1), 4)

        return LoRaHeader(
            length=length,
            coding_rate=coding_rate,
            has_crc=has_crc,
            raw=decoded_hdr[:3],
        )

    def _payload_symbol_count(self, payload_len: int, coding_rate: int) -> int:
        """根据 payload 长度和 CR 估算需要读取的 payload 符号数。"""
        redundancy = 2 if self.config.reduced_rate else 0
        symbols_per_block = coding_rate + 4
        bits_needed = float(payload_len) * 8.0
        symbols_needed = bits_needed * (symbols_per_block / 4.0) / float(self.config.sf - redundancy)
        blocks_needed = int(math.ceil(symbols_needed / symbols_per_block))
        return blocks_needed * symbols_per_block

    def decode(self, data: np.ndarray) -> list[LoRaFrame]:
        """对一段 IQ 数据执行完整 LoRa 解码并返回帧列表。"""
        ns = self.config.symbol_samples
        frames: list[LoRaFrame] = []

        syncs = self.sync.detect(data)
        for s in syncs:
            header_words = self.demod.extract_words(
                data,
                start=s.header_start,
                n_symbols=8,
                reduced_rate=True,
            )
            if len(header_words) < 8:
                continue

            decoded_header = self.bitpipe.decode_block(
                header_words,
                ppm=self.config.sf - 2,
                is_header=True,
                coding_rate=self.config.coding_rate,
            )
            header = self._parse_header(decoded_header)
            if header is None:
                continue

            total_len = header.length + (2 if header.has_crc else 0)
            n_payload_symbols = self._payload_symbol_count(total_len, header.coding_rate)

            payload_start = s.header_start + 8 * ns
            payload_words = self.demod.extract_words(
                data,
                start=payload_start,
                n_symbols=n_payload_symbols,
                reduced_rate=self.config.reduced_rate,
            )

            block_size = header.coding_rate + 4
            ppm = self.config.sf - (2 if self.config.reduced_rate else 0)
            decoded_payload = bytearray()
            for i in range(0, len(payload_words), block_size):
                block = payload_words[i : i + block_size]
                if len(block) < block_size:
                    break
                decoded_payload.extend(
                    self.bitpipe.decode_block(
                        block,
                        ppm=ppm,
                        is_header=False,
                        coding_rate=header.coding_rate,
                    )
                )

            decoded_payload = decoded_payload[:total_len]
            mac_crc = None
            payload = bytes(decoded_payload)
            if header.has_crc and len(payload) >= 2:
                mac_crc = (payload[-2] << 8) | payload[-1]
                payload = payload[:-2]

            frames.append(
                LoRaFrame(
                    start_index=s.packet_start,
                    header=header,
                    payload=payload,
                    mac_crc=mac_crc,
                    snr_db=10.0 * math.log10(max(s.score, 1e-12)),
                )
            )

        return frames

    def decode_file(self, file_path: str) -> list[LoRaFrame]:
        """从 .bin 读取 complex64 IQ 并解码。"""
        data = np.fromfile(file_path, dtype=np.complex64)
        return self.decode(data)
