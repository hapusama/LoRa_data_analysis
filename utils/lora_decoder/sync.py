from __future__ import annotations

import numpy as np
from scipy.signal import correlate

from .chirp import upchirp, downchirp
from .config import LoRaConfig
from .types import SyncResult


class PreambleSynchronizer:
    """LoRa 前导码同步器。

    处理流程与 gr-lora 的思路一致：
    1) 先用相邻 chirp 的归一化相关做粗检测；
    2) 再用单 upchirp 互相关定位到更精确的前导码起点；
    3) 最后在预期位置附近用 downchirp 搜索 SFD。
    """

    def __init__(
        self,
        cfg: LoRaConfig,
        detect_threshold: float = 0.90,
        upchirp_corr_threshold: float = 0.30,
        search_margin_ratio: float = 0.25,
    ) -> None:
        self.cfg = cfg
        self.detect_threshold = detect_threshold
        self.upchirp_corr_threshold = upchirp_corr_threshold
        self.search_margin_ratio = search_margin_ratio
        self._up = upchirp(cfg)
        self._down = downchirp(cfg)

    @staticmethod
    def _norm_corr(x: np.ndarray, y: np.ndarray) -> float:
        dot = np.vdot(x, y)
        e = np.sqrt(np.vdot(x, x).real * np.vdot(y, y).real) + 1e-12
        return float(np.abs(dot) / e)

    def _coarse_preamble_positions(self, data: np.ndarray) -> list[int]:
        """粗检测：寻找“相邻符号高度相似”的候选位置。"""
        ns = self.cfg.symbol_samples
        if len(data) < 2 * ns:
            return []

        hop = max(ns // 4, 1)
        candidates: list[int] = []
        last = -ns

        for i in range(0, len(data) - 2 * ns, hop):
            c = self._norm_corr(data[i : i + ns], data[i + ns : i + 2 * ns])
            if c >= self.detect_threshold and (i - last) > ns:
                candidates.append(i)
                last = i

        return candidates

    def _refine_to_upchirp(self, data: np.ndarray, coarse: int) -> tuple[int, float] | None:
        """细化：在 coarse 附近用 upchirp 模板互相关取峰值。"""
        ns = self.cfg.symbol_samples
        start = max(0, coarse - ns)
        end = min(len(data), coarse + 2 * ns)
        seg = data[start:end]
        if len(seg) < ns:
            return None

        corr = correlate(seg, self._up, mode="valid")
        power = np.abs(corr) ** 2
        if np.max(power) <= 0:
            return None

        peak = int(np.argmax(power))
        local = seg[peak : peak + ns]
        if len(local) < ns:
            return None

        # 用真正的归一化相关系数打分，避免“局部最大值归一化”导致阈值失真。
        score = self._norm_corr(local, self._up)
        if score < self.upchirp_corr_threshold:
            return None

        return start + peak, score

    def _coarse_positions_by_upchirp_corr(self, data: np.ndarray, threshold: float) -> list[int]:
        """回退粗检测：直接用单 upchirp 全局互相关找峰。"""
        ns = self.cfg.symbol_samples
        if len(data) < ns:
            return []

        corr = correlate(data, self._up, mode="valid")
        power = np.abs(corr) ** 2
        m = float(np.max(power))
        if m <= 0:
            return []

        power = power / m
        peaks = np.where(power > threshold)[0]

        dedup: list[int] = []
        last = -ns
        for p in peaks:
            if p - last >= ns:
                dedup.append(int(p))
                last = int(p)
        return dedup

    def _find_sfd(self, data: np.ndarray, preamble_start: int) -> int | None:
        """在预期 SFD 区间用 downchirp 搜索峰值。"""
        ns = self.cfg.symbol_samples
        margin = int(ns * self.search_margin_ratio)
        expected = preamble_start + self.cfg.preamble_symbols * ns
        win_start = max(0, expected - margin)
        win_end = min(len(data), expected + ns + margin)

        seg = data[win_start:win_end]
        if len(seg) < ns:
            return None

        corr = correlate(seg, self._down, mode="valid")
        peak = int(np.argmax(np.abs(corr) ** 2))
        return win_start + peak

    def detect(self, data: np.ndarray) -> list[SyncResult]:
        """返回所有检测到的包同步结果。"""
        ns = self.cfg.symbol_samples
        preamble_len = self.cfg.preamble_symbols * ns

        starts: list[SyncResult] = []
        last_packet = -preamble_len

        for coarse in self._coarse_preamble_positions(data):
            refined = self._refine_to_upchirp(data, coarse)
            if refined is None:
                continue
            preamble_start, score = refined

            sfd = self._find_sfd(data, preamble_start)
            if sfd is None:
                continue

            packet_start = sfd - preamble_len
            if packet_start < 0:
                continue
            if packet_start - last_packet <= preamble_len:
                continue

            header_start = sfd + self.cfg.pause_after_sfd_samples
            starts.append(
                SyncResult(
                    packet_start=packet_start,
                    sfd_pos=sfd,
                    header_start=header_start,
                    score=score,
                )
            )
            last_packet = packet_start

        # 回退：若主路径没有命中，用单 upchirp 相关结果再尝试一轮。
        if not starts:
            fallback_coarse = self._coarse_positions_by_upchirp_corr(
                data,
                threshold=max(self.upchirp_corr_threshold, 0.12),
            )

            for coarse in fallback_coarse:
                refined = self._refine_to_upchirp(data, coarse)
                if refined is None:
                    continue
                preamble_start, score = refined

                sfd = self._find_sfd(data, preamble_start)
                if sfd is None:
                    continue

                packet_start = sfd - preamble_len
                if packet_start < 0:
                    continue
                if packet_start - last_packet <= preamble_len:
                    continue

                header_start = sfd + self.cfg.pause_after_sfd_samples
                starts.append(
                    SyncResult(
                        packet_start=packet_start,
                        sfd_pos=sfd,
                        header_start=header_start,
                        score=score,
                    )
                )
                last_packet = packet_start

        return starts
