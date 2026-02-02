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
    search_margin_ratio=0.25,
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
    preamble_length = preamble_symbols * symbol_samples

    # 生成 upchirp / downchirp 模板
    Ts = symbol_samples / fs
    t = np.arange(symbol_samples) / fs
    upchirp = np.exp(1j * 2 * np.pi * (-BW / 2 * t + (BW / (2 * Ts)) * t**2))
    downchirp = np.conj(upchirp)

    # 粗检测：用前导码长模板互相关
    preamble_template = np.tile(upchirp, preamble_symbols)
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

def plot_bin_file(file_path, save_path, start_sample=0, num_samples=1000):
    """
    读取bin文件并绘制时域幅度、频域幅度和功率谱密度图，然后保存到指定路径。
    可以指定起始样本和样本数量来进行数据切片。
    """
    # 读取IQ数据，假设是complex64格式
    full_data = np.fromfile(file_path, dtype=np.complex64)

    print(f"Loaded {len(full_data)} samples from {os.path.basename(file_path)}")

    # 数据切片
    end_sample = start_sample + num_samples
    if end_sample > len(full_data):
        end_sample = len(full_data)
    data = full_data[start_sample:end_sample]   # 只读取指定样本数
    # data = np.fromfile(file_path, dtype=np.complex64) # 全部采样
    print("前5个样本的实部(I):", data[:5].real)
    print("前5个样本的虚部(Q):", data[:5].imag)
    # 计算I和Q的相似度
    correlation = np.corrcoef(data.real, data.imag)[0,1]
    print(f"I路和Q路的相关系数: {correlation}")
    print(f"Using samples from {start_sample} to {end_sample-1} ({len(data)} samples)")

    # 计算时域幅度
    amplitude_time = np.abs(full_data)

    # 计算频域
    fft_data = np.fft.fft(data)
    freq = np.fft.fftfreq(len(data))

    # 计算PSD
    f, Pxx = welch(data, fs=1.0, nperseg=1024, return_onesided=False)

    # 创建一个大图，包含3个子窗口
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 子图1：时域幅度
    axes[0].plot(amplitude_time)
    axes[0].set_title('Time Domain Amplitude')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True)

    # 子图2：频域幅度
    axes[1].plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(fft_data)))
    axes[1].set_title('Frequency Domain Amplitude (FFT)')
    axes[1].set_xlabel('Frequency')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True)

    # 子图3：功率谱密度
    axes[2].semilogy(f, Pxx)
    axes[2].set_title('Power Spectral Density')
    axes[2].set_xlabel('Frequency')
    axes[2].set_ylabel('PSD')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Plot saved to {save_path}")


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

def process_bin_file(bin_file, save_fig_dir, symbol_samples, packet_samples, preamble_symbols, BW, fs, sf=7 ):
    """
    处理单个bin文件：检测包并生成图表。
    """
    # 获取文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(bin_file))[0]

    # 读取完整数据
    data = np.fromfile(bin_file, dtype=np.complex64)
    print(f"Processing {base_name}: {len(data)} samples")

    # 检测包起始
    starts = detect_packets(data, symbol_samples, BW, fs, preamble_symbols, threshold=0.8)
    if not starts:
        print(f"No packets detected in {base_name}")
        return

    # 取第一个检测到的包的起始位置
    pkt_start = starts[0]
    pkt_end = min(pkt_start + packet_samples, len(data))
    packet = data[pkt_start:pkt_end]
    print(f"Extracted packet len (length {len(packet)})")
    
    # 选择前导码中的一个 upchirp 符号进行分析，默认选第一个 preamble 符号
    sym_start = starts[0]
    sym_end = sym_start + symbol_samples

    symbol = data[sym_start:sym_end]
    # dechirp
    # symbol = dechirp_symbol(symbol, BW, fs, sf=sf)
    print(f"Using preamble upchirp symbol at packet offset {sym_start} (length {len(symbol)})")


    # --- FFT for the selected symbol ---
    def next_pow2(x):
        return 1 << int(np.ceil(np.log2(x)))

    # packet = dechirp_symbol(packet, BW, fs)   # 可选：对整个包去啁啾以观察频谱变化
    Nfft = max(1024, next_pow2(len(symbol) * 2))
    fft_sym = np.fft.fft(symbol, n=Nfft)
    freq = np.fft.fftfreq(Nfft, d=1/fs)
    fft_sym_shift = np.fft.fftshift(fft_sym)
    freq_shift = np.fft.fftshift(freq)

    # 仅保留带宽范围内的频点
    mask = (freq_shift >= -BW/2) & (freq_shift <= BW/2)
    freq_plot = freq_shift[mask]
    amp_plot = np.abs(fft_sym_shift[mask])

    # --- Spectrogram for the selected symbol ---
    nperseg = min(128, len(symbol)) if len(symbol) >= 32 else len(symbol)
    # nperseg = 128
    noverlap = int(nperseg * 0.5)
    f_spec, t_spec, Sxx = spectrogram(symbol, fs=fs, nperseg=nperseg, noverlap=noverlap, return_onesided=False)
    # shift freq axis and convert to dB
    Sxx_shift = np.fft.fftshift(Sxx, axes=0)
    f_spec_shift = np.fft.fftshift(f_spec)
    Sxx_db = 10 * np.log10(np.abs(Sxx_shift) + 1e-20)

    # --- PSD for the selected symbol ---
    psd_nperseg = min(len(symbol), 256)
    psd_noverlap = int(psd_nperseg * 0.5)
    f_psd, Pxx = welch(symbol, fs=fs, nperseg=psd_nperseg, noverlap=psd_noverlap, return_onesided=False)
    f_psd_shift = np.fft.fftshift(f_psd)
    Pxx_shift = np.fft.fftshift(Pxx)

    # 绘图：三图（FFT / spectrogram / PSD）
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(freq_plot / 1000, amp_plot)
    axes[0].set_title('Preamble Upchirp Packet FFT')
    axes[0].set_xlabel('Frequency (kHz)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True)

    # spectrogram extent - 处理单时间窗情况
    if t_spec.size <= 1:
        extent = (0.0, len(symbol) / fs * 1000, f_spec_shift[0] / 1000, f_spec_shift[-1] / 1000)
    else:
        extent = (t_spec[0] * 1000, t_spec[-1] * 1000, f_spec_shift[0] / 1000, f_spec_shift[-1] / 1000)

    im = axes[1].imshow(Sxx_db, aspect='auto', origin='lower', extent=extent, cmap='viridis')
    axes[1].set_title('Preamble Symbol Spectrogram')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Frequency (kHz)')
    fig.colorbar(im, ax=axes[1], format='%+2.0f dB')
    axes[1].set_ylim(-BW/2000, BW/2000)

    axes[2].semilogy(f_psd_shift / 1000, np.abs(Pxx_shift))
    axes[2].set_title('Preamble Symbol PSD (Welch)')
    axes[2].set_xlabel('Frequency (kHz)')
    axes[2].set_ylabel('PSD')
    axes[2].grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_fig_dir, f'{base_name}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved preamble symbol FFT and spectrogram to {save_path}")


def plot_combined_spectra(bin_files, save_fig_path, symbol_samples, preamble_symbols, BW, fs, threshold=0.8):
    """
    对多个 bin 文件进行前导码检测，取第一个检测到的 upchirp 符号，统一画 FFT 与 PSD 到一张图。

    bin_files: 绝对或相对路径列表
    save_fig_path: 输出图保存路径（含文件名）
    """

    def next_pow2(x):
        return 1 << int(np.ceil(np.log2(x)))

    fft_curves = []
    psd_curves = []

    for bin_file in bin_files:
        data = np.fromfile(bin_file, dtype=np.complex64)
        starts = detect_packets(data, symbol_samples, BW, fs, preamble_symbols, threshold=threshold)
        if not starts:
            print(f"Skip {os.path.basename(bin_file)}: no packet detected")
            continue

        sym_start = starts[0]
        sym_end = sym_start + symbol_samples
        if sym_end > len(data):
            print(f"Skip {os.path.basename(bin_file)}: symbol truncated")
            continue

        symbol = data[sym_start:sym_end]
        symbol = dechirp_symbol(symbol, BW, fs)
        # FFT
        Nfft = max(1024, next_pow2(len(symbol) * 2))
        fft_sym = np.fft.fft(symbol, n=Nfft)
        freq = np.fft.fftfreq(Nfft, d=1/fs)
        fft_sym_shift = np.fft.fftshift(fft_sym)
        freq_shift = np.fft.fftshift(freq)
        mask = (freq_shift >= -BW/2) & (freq_shift <= BW/2)
        fft_curves.append((freq_shift[mask] / 1000, np.abs(fft_sym_shift[mask]), os.path.basename(bin_file)))

        # PSD (Welch)
        psd_nperseg = min(len(symbol), 256)
        psd_noverlap = int(psd_nperseg * 0.5)
        f_psd, Pxx = welch(symbol, fs=fs, nperseg=psd_nperseg, noverlap=psd_noverlap, return_onesided=False)
        f_psd_shift = np.fft.fftshift(f_psd)
        Pxx_shift = np.fft.fftshift(Pxx)
        psd_curves.append((f_psd_shift / 1000, np.abs(Pxx_shift), os.path.basename(bin_file)))

        
    if not fft_curves:
        print("No valid packets to plot")
        return

    # 保存 FFT 图
    simsun_fp = FontProperties(family='SimSun')
    fig_fft, ax_fft = plt.subplots(1, 1, figsize=(12, 5))
    for freq_khz, amp, label in fft_curves:
        ax_fft.plot(freq_khz, amp, label=label, linewidth=3)
    ax_fft.set_title('')
    ax_fft.set_xlabel('Frequency (kHz)', fontsize=22, fontfamily='Times New Roman')
    ax_fft.set_ylabel('Amplitude', fontsize=22, fontfamily='Times New Roman')
    ax_fft.tick_params(axis='both', labelsize=22)
    for lbl in ax_fft.get_xticklabels() + ax_fft.get_yticklabels():
        lbl.set_fontfamily('Times New Roman')
    # 仅保留 FFT 横坐标 0~20 kHz 区域，便于观察旁瓣
    ax_fft.set_xlim(2, 3)
    ax_fft.grid(True)
    ax_fft.legend(prop=FontProperties(family='Times New Roman', size=22))
    fig_fft.text(0.5, 0.02, '前导码上啁啾频率幅度谱', ha='center', fontproperties=simsun_fp, fontsize=22)
    fft_path = save_fig_path if save_fig_path.lower().endswith('.png') else f"{save_fig_path}_fft.png"
    if fft_path == save_fig_path and save_fig_path.lower().endswith('.png'):
        fft_path = save_fig_path.replace('.png', '_fft.png')
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(fft_path, dpi=150)
    plt.close()

    # 保存 PSD 图
    fig_psd, ax_psd = plt.subplots(1, 1, figsize=(12, 5))
    for freq_khz, psd, label in psd_curves:
        ax_psd.semilogy(freq_khz, psd, label=label, linewidth=3)
    ax_psd.set_title('')
    ax_psd.set_xlabel('Frequency (kHz)', fontsize=22, fontfamily='Times New Roman')
    ax_psd.set_ylabel('PSD', fontsize=22, fontfamily='Times New Roman')
    ax_psd.tick_params(axis='both', labelsize=22)
    for lbl in ax_psd.get_xticklabels() + ax_psd.get_yticklabels():
        lbl.set_fontfamily('Times New Roman')
    ax_psd.grid(True)
    ax_psd.legend(prop=FontProperties(family='Times New Roman', size=22))
    fig_psd.text(0.5, 0.02, '前导码上啁啾功率谱密度', ha='center', fontproperties=simsun_fp, fontsize=22)
    psd_path = save_fig_path if save_fig_path.lower().endswith('.png') else f"{save_fig_path}_psd.png"
    if psd_path == save_fig_path and save_fig_path.lower().endswith('.png'):
        psd_path = save_fig_path.replace('.png', '_psd.png')
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(psd_path, dpi=150)
    plt.close()

    print(f"Saved FFT spectra to {fft_path}")
    print(f"Saved PSD spectra to {psd_path}")

if __name__ == "__main__":
    # LoRa参数 (请根据你的设置填写)
    CF = 487700000  # 中心频率 (Hz), 例如433MHz
    BW = 125000  # 带宽 (Hz), 例如125kHz
    SF = 9       # 扩频因子, 例如7
    fs = 250000 # 采样率 (Hz), 例如1MHz
    preamble_symbols = 5  # 前导码符号数
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
        process_bin_file(bin_file, save_fig_dir, symbol_samples, packet_samples, preamble_symbols, BW, fs)
        plot_bin_file(bin_file, os.path.join(save_fig_dir, os.path.splitext(os.path.basename(bin_file))[0] + '.png'), start_sample=0, num_samples=5000)

    # 组合绘制指定文件的 FFT + PSD
    # target_names = ['5m.bin', '10m.bin', '15m.bin']
    # target_files = [os.path.join(folder_path, name) for name in target_names if os.path.exists(os.path.join(folder_path, name))]
    # if target_files:
    #     combined_path = os.path.join(save_fig_dir, 'combined_fft_psd_dechirped.png')
    #     plot_combined_spectra(target_files, combined_path, symbol_samples, preamble_symbols, BW, fs)
    # else:
    #     print("No target files found for combined plot")
