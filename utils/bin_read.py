from statistics import correlation
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, correlate, spectrogram
from matplotlib.font_manager import FontProperties
import os
import glob

def detect_packets(
    data,
    symbol_samples,
    BW,
    fs,
    preamble_symbols=5,
    threshold=0.5,
    search_margin_ratio=0.5,
):
    """
    使用标准LoRa前导码检测，并利用 SFD 下啁啾进行精确对齐。

    data: IQ复数组
    symbol_samples: 每个符号的采样点数
    BW: 带宽 (Hz)
    fs: 采样率 (Hz)
    preamble_symbols: 前导码 upchirp 数
    threshold: 粗检测的相关阈值
    search_margin_ratio: 在预期 SFD 处前后搜索的比例 (相对 symbol_samples)
    返回值: 经过 SFD 精确对齐的包起始索引列表
    """

    # 计算前导码长度
    preamble_length = symbol_samples*preamble_symbols

    # 生成 upchirp / downchirp 模板
    Ts = symbol_samples / fs
    t = np.arange(symbol_samples) / fs
    upchirp = np.exp(1j * 2 * np.pi * (-BW / 2 * t + (BW / (2 * Ts)) * t**2))
    downchirp = np.conj(upchirp)

    # 粗检测：使用长前导码模板（多个 upchirp）互相关
    preamble_template = np.tile(upchirp, 1)
    corr = correlate(data, preamble_template, mode="valid")
    power = np.abs(corr) ** 2
    power = power / np.max(power)
    coarse_peaks = np.where(power > threshold)[0]

    # 精对齐：在预期的 SFD 位置用 downchirp 再相关
    margin = int(symbol_samples * search_margin_ratio)
    starts = []
    last_start = -preamble_length

    for coarse in coarse_peaks:
        if coarse - last_start <= preamble_length:
            continue

        # 预期 SFD 起点（在前导码之后）
        expected_sfd = coarse + preamble_length
        win_start = max(expected_sfd - margin, 0)
        win_end = min(expected_sfd + symbol_samples + margin, len(data))

        segment = data[win_start:win_end]
        if len(segment) < symbol_samples // 2:
            continue

        sfd_corr = correlate(segment, downchirp, mode="valid")
        sfd_power = np.abs(sfd_corr) ** 2
        local_peak = np.argmax(sfd_power)

        # 精确 SFD 位置 = 窗口起点 + 局部峰位置
        sfd_pos = win_start + local_peak
        refined_start = sfd_pos - preamble_length

        if refined_start >= 0:
            starts.append(refined_start)
            last_start = refined_start

    return starts

def segment_packets(data, packet_samples, starts):
    """
    根据起始索引分割包。
    """
    packets = []
    for start in starts:
        end = min(start + packet_samples, len(data))
        packets.append(data[start:end])
    return packets

def dechirp_symbol(symbol, BW, fs, sf=12):
    """
    去啁啾。若提供 sf，则优先用 Ts = 2**sf / BW（并校验样本数），否则用 n/fs。
    """
    n = len(symbol)
    if sf is not None:
        Ts_expected = (2**sf) / BW
        n_expected = int(round(fs * Ts_expected))
        # 若样本数与期望不同，退回到直接由 n/fs 计算 Ts
        if n_expected == n:
            Ts = Ts_expected
        else:
            Ts = n / fs
    else:
        Ts = n / fs

    t = np.arange(n) / fs
    # 参考 upchirp（与 detect 中一致的定义）
    upchirp = np.exp(1j * 2 * np.pi * (-BW/2 * t + (BW / (2 * Ts)) * t**2))
    return symbol * np.conj(upchirp)

def process_bin_file(bin_file, save_fig_dir, symbol_samples, packet_samples, preamble_symbols, BW, fs, threshold=0.4, sf=7 ):
    """
    处理单个bin文件：检测包后仅生成前导码时频图。
    """
    # 获取文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(bin_file))[0]

    # 读取完整数据
    data = np.fromfile(bin_file, dtype=np.complex64)
    print(f"Processing {base_name}: {len(data)} samples")

    # 检测包起始
    starts = detect_packets(data, symbol_samples, BW, fs, preamble_symbols, threshold=threshold)
    if not starts:
        print(f"No packets detected in {base_name}")
        return

    # 选择前导码中的全部 upchirp 符号进行分析（只取1个 chirp）
    sym_start = starts[0]
    sym_end = sym_start + symbol_samples*preamble_symbols

    symbol = data[sym_start:sym_end]
    # dechirp
    symbol = dechirp_symbol(symbol, BW, fs, sf=sf)
    
    chirp_count = len(symbol) / symbol_samples
    print(f"Using preamble upchirp symbol at packet offset {sym_start} (length {len(symbol)} samples, {chirp_count:.1f} chirps)")

    # --- Spectrogram for the selected symbol ---
    nperseg = min(128, len(symbol)) if len(symbol) >= 32 else len(symbol)
    # nperseg = 128
    noverlap = int(nperseg * 0.5)
    f_spec, t_spec, Sxx = spectrogram(symbol, fs=fs, nperseg=nperseg, noverlap=noverlap, return_onesided=False)
    # shift freq axis and convert to dB
    Sxx_shift = np.fft.fftshift(Sxx, axes=0)
    f_spec_shift = np.fft.fftshift(f_spec)
    Sxx_db = 10 * np.log10(np.abs(Sxx_shift) + 1e-20)

    # 绘图：仅保留时频图
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    # spectrogram extent - 处理单时间窗情况
    if t_spec.size <= 1:
        extent = (0.0, len(symbol) / fs * 1000, f_spec_shift[0] / 1000, f_spec_shift[-1] / 1000)
    else:
        extent = (t_spec[0] * 1000, t_spec[-1] * 1000, f_spec_shift[0] / 1000, f_spec_shift[-1] / 1000)

    im = ax.imshow(Sxx_db, aspect='auto', origin='lower', extent=extent, cmap='viridis')
    ax.set_title('Preamble Symbol Spectrogram')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (kHz)')
    fig.colorbar(im, ax=ax, format='%+2.0f dB')
    ax.set_ylim(-BW/2000, BW/2000)

    # 标记每个chirp的边界
    symbol_duration_ms = (symbol_samples / fs) * 1000
    for i in range(1, int(preamble_symbols)):
        ax.axvline(i * symbol_duration_ms, color='w', linestyle='--', linewidth=0.7, alpha=0.35)

    plt.tight_layout()
    save_path = os.path.join(save_fig_dir, f'{base_name}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved preamble symbol spectrogram to {save_path}")

    # --- Phase analysis: 连续相位变化图 ---
    time_axis = np.arange(len(symbol)) / fs * 1000  # 时间轴 (ms)
    phase = np.unwrap(np.angle(symbol))  # 展开相位，避免 -π 到 π 的跳跃

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.plot(time_axis, phase, linewidth=1.5, color='steelblue', alpha=0.8)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Phase (radians)', fontsize=12)
    ax.set_title('Preamble Phase Continuity (16 Chirps)', fontsize=14)
    ax.grid(True, alpha=0.3)

    # 标记每个chirp的边界
    for i in range(1, int(preamble_symbols)):
        ax.axvline(i * symbol_duration_ms, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.text(i * symbol_duration_ms, ax.get_ylim()[1] * 0.95, f'C{i}', 
                fontsize=9, ha='right', color='red', alpha=0.7)

    plt.tight_layout()
    phase_path = os.path.join(save_fig_dir, f'{base_name}_phase.png')
    plt.savefig(phase_path, dpi=150)
    plt.close()
    print(f"Saved preamble phase diagram to {phase_path}")



if __name__ == "__main__":
    # LoRa参数 (请根据你的设置填写)
    CF = 487700000  # 中心频率 (Hz), 例如433MHz
    BW = 125000  # 带宽 (Hz), 例如125kHz
    SF = 11      # 扩频因子, 例如7
    fs = 500000 # 采样率 (Hz), 例如1MHz
    preamble_symbols = 16  # 前导码符号数
    detect_threshold = 0.3  # 前导码粗检测阈值
    payload_bytes = 64    # 有效载荷字节数
    # 编码率: 1->4/5, 2->4/6, 3->4/7, 4->4/8
    coding_rate = 1  # 使用4/5时设置为1

    # 计算符号采样点数
    symbol_samples = int(fs * (2**SF) / BW)
    # 计算包采样点数 (粗略)
    packet_samples = preamble_symbols * symbol_samples + int((payload_bytes * 8) / SF * symbol_samples)

    print(f"Symbol samples: {symbol_samples}, Packet samples: {packet_samples}")

    # 文件夹路径
    folder_path = r'd:\Desktop\data_analysis\rawData\dong'

    # 创建save_fig文件夹
    save_fig_dir = 'save_fig'
    os.makedirs(save_fig_dir, exist_ok=True)

    # 获取所有bin文件
    bin_files = glob.glob(os.path.join(folder_path, '*.bin'))

    # 对每个bin文件调用处理函数
    for bin_file in bin_files:
        process_bin_file(bin_file, save_fig_dir, symbol_samples, packet_samples, preamble_symbols, BW, fs, threshold=detect_threshold,sf=SF)
