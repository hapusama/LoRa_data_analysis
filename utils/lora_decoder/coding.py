from __future__ import annotations

from typing import Iterable


def rotl(bits: int, count: int = 1, size: int = 8) -> int:
    if size <= 0:
        return 0
    mask = (1 << size) - 1
    count %= size
    bits &= mask
    return ((bits << count) & mask) | (bits >> (size - count))


def select_bits(data: int, indices: Iterable[int]) -> int:
    out = 0
    for i, idx in enumerate(indices):
        if data & (1 << idx):
            out |= (1 << i)
    return out


def _pn9_sequence(length: int, seed: int = 0x01FF) -> list[int]:
    state = seed & 0x1FF
    out: list[int] = []
    for _ in range(length):
        b = 0
        for i in range(8):
            new_bit = ((state >> 5) ^ state) & 1
            state = ((state >> 1) | (new_bit << 8)) & 0x1FF
            b |= (state & 1) << i
        out.append(b)
    return out


class BitPipelineDecoder:
    shuffle_pattern = (5, 0, 1, 2, 4, 3, 6, 7)
    data_indices = (1, 2, 3, 5)
    hamming_lut = (0x0, 0x0, 0x4, 0x0, 0x6, 0x0, 0x0, 0x2,
                   0x7, 0x0, 0x0, 0x3, 0x0, 0x5, 0x1, 0x0)

    def deinterleave(self, words: list[int], ppm: int) -> list[int]:
        bits_per_word = len(words)
        if bits_per_word == 0:
            return []
        if bits_per_word > 8:
            raise ValueError("deinterleave currently supports <=8 bits_per_word")

        out = [0 for _ in range(ppm)]
        offset_start = ppm - 1

        for i, w in enumerate(words):
            r = rotl(w, i, ppm)
            j = 1 << offset_start
            x = offset_start
            while j:
                out[x] |= (1 if (r & j) else 0) << i
                j >>= 1
                x -= 1

        return out

    def deshuffle(self, words: list[int], is_header: bool) -> list[int]:
        to_decode = 5 if is_header else len(words)
        out: list[int] = []
        for i in range(min(to_decode, len(words))):
            r = 0
            for j, pos in enumerate(self.shuffle_pattern):
                r |= (1 if (words[i] & (1 << pos)) else 0) << j
            out.append(r)
        if is_header:
            out.append(0)
        return out

    def dewhiten(self, words: list[int], is_header: bool, coding_rate: int) -> list[int]:
        if is_header:
            prng = [0] * len(words)
        else:
            prng = _pn9_sequence(len(words), seed=0x01FF if coding_rate <= 2 else 0x01A5)
        return [(w ^ p) & 0xFF for w, p in zip(words, prng)]

    def _hamming_decode_soft_byte(self, v: int) -> int:
        p1 = (v >> 0) & 1
        p2 = (v >> 4) & 1
        p3 = (v >> 6) & 1
        p4 = (v >> 7) & 1
        p1c = ((v >> 2) & 1) ^ ((v >> 3) & 1) ^ ((v >> 5) & 1)
        p2c = ((v >> 1) & 1) ^ ((v >> 2) & 1) ^ ((v >> 3) & 1)
        p3c = ((v >> 1) & 1) ^ ((v >> 2) & 1) ^ ((v >> 5) & 1)
        p4c = ((v >> 1) & 1) ^ ((v >> 3) & 1) ^ ((v >> 5) & 1)
        syndrome = ((1 if p1 != p1c else 0) |
                    ((1 if p2 != p2c else 0) << 1) |
                    ((1 if p3 != p3c else 0) << 2) |
                    ((1 if p4 != p4c else 0) << 3))
        if syndrome:
            v ^= 1 << self.hamming_lut[syndrome]
        d0 = (v >> 1) & 1
        d1 = (v >> 2) & 1
        d2 = (v >> 3) & 1
        d3 = (v >> 5) & 1
        return d0 | (d1 << 1) | (d2 << 2) | (d3 << 3)

    def _extract_data_only(self, words: list[int], is_header: bool) -> list[int]:
        out: list[int] = []
        for i in range(0, len(words), 2):
            d1 = select_bits(words[i], self.data_indices) & 0xF
            d2 = select_bits(words[i + 1], self.data_indices) & 0xF if i + 1 < len(words) else 0
            out.append(((d1 << 4) | d2) if is_header else ((d2 << 4) | d1))
        return out

    def hamming_decode(self, words: list[int], is_header: bool, coding_rate: int) -> list[int]:
        if coding_rate in (3, 4):
            out: list[int] = []
            for i in range(0, len(words), 2):
                d1 = self._hamming_decode_soft_byte(words[i])
                d2 = self._hamming_decode_soft_byte(words[i + 1]) if i + 1 < len(words) else 0
                out.append(((d1 << 4) | d2) if is_header else ((d2 << 4) | d1))
            return out
        return self._extract_data_only(words, is_header)

    def decode_block(self, words: list[int], ppm: int, is_header: bool, coding_rate: int) -> bytes:
        deinterleaved = self.deinterleave(words, ppm)
        deshuffled = self.deshuffle(deinterleaved, is_header)
        dewhitened = self.dewhiten(deshuffled, is_header, coding_rate)
        decoded = self.hamming_decode(dewhitened, is_header, coding_rate)
        return bytes(decoded)
