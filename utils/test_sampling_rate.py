"""
测试采样率 - 检查哪个采样率能正确显示16个前导码chirp
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import os


def count_visible_chirps(data, packet_start, fs, bw, sf, preamble_symbols=16):
    """
    通过时频图分析可见的chirp数量
    """
    symbol_samples = int(fs * (2**sf) / bw)
    
    # 提取前导码区域（16个符号 + 几个额外）
    extract_samples = int(symbol_samples * (preamble_symbols + 4))
    segment = data[packet_start:packet_start + extract_samples]
    
    if len(segment) < symbol_samples * 2:
        return 0, None
    
    # 生成时频图
    nperseg = min(256, symbol_samples // 4)
    f, t, Sxx = spectrogram(segment, fs=fs, nperseg=nperseg, 
                            noverlap=nperseg//2, return_onesided=False)
    
    # 计算每个时间点的能量（用于检测chirp边界）
    Sxx_mag = np.abs(np.fft.fftshift(Sxx, axes=0))
    energy_per_time = np.max(Sxx_mag, axis=0)
    
    # 通过能量峰值检测chirp数量
    # 找局部最大值
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(energy_per_time, distance=len(energy_per_time)//(preamble_symbols*2))
    
    estimated_chirps = len(peaks)
    
    return estimated_chirps, (f, t, Sxx, energy_per_time, peaks)


def test_sampling_rate(data, packet_start, bw, sf, test_fs_values):
    """
    测试不同的采样率
    """
    print(f"\n测试 SF{sf}, 寻找正确的采样率...")
    print(f"前导码应包含 16 个 upchirps")
    print("-" * 60)
    
    results = []
    
    for fs in test_fs_values:
        est_chirps, details = count_visible_chirps(data, packet_start, fs, bw, sf)
        
        # 计算预期的符号持续时间
        symbol_duration = (2**sf) / bw * 1000  # ms
        samples_per_symbol = int(fs * (2**sf) / bw)
        
        print(f"fs={fs/1e6:.2f}MHz: ", end="")
        
        if est_chirps == 0:
            print(f"无法检测 (symbol_samples={samples_per_symbol})")
        else:
            match = "[OK]" if 14 <= est_chirps <= 18 else "[NG]"
            print(f"检测到 ~{est_chirps} 个chirp (symbol_samples={samples_per_symbol}, duration={symbol_duration:.2f}ms) {match}")
            
            results.append({
                'fs': fs,
                'est_chirps': est_chirps,
                'samples_per_symbol': samples_per_symbol,
                'symbol_duration_ms': symbol_duration,
                'details': details
            })
    
    # 找出最佳匹配
    best = None
    best_diff = float('inf')
    for r in results:
        diff = abs(r['est_chirps'] - 16)
        if diff < best_diff:
            best_diff = diff
            best = r
    
    if best:
        print(f"\n>>> 最佳匹配: fs={best['fs']/1e6:.2f}MHz (检测到 {best['est_chirps']} 个chirp)")
    
    return best, results


def visualize_preamble(data, packet_start, fs, bw, sf, save_path):
    """
    可视化前导码和SFD
    """
    symbol_samples = int(fs * (2**sf) / bw)
    
    # 提取16个前导码 + SFD + 几个header符号
    n_symbols = 16 + 4
    segment = data[packet_start:packet_start + int(symbol_samples * n_symbols)]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 时频图
    nperseg = min(256, symbol_samples // 4)
    f, t, Sxx = spectrogram(segment, fs=fs, nperseg=nperseg, 
                            noverlap=nperseg//2, return_onesided=False)
    
    Sxx_shift = np.fft.fftshift(Sxx, axes=0)
    f_shift = np.fft.fftshift(f)
    
    extent = [t[0]*1000, t[-1]*1000, f_shift[0]/1e3, f_shift[-1]/1e3]
    im = ax.imshow(10*np.log10(np.abs(Sxx_shift)+1e-20), aspect='auto', 
                   origin='lower', extent=extent, cmap='viridis', vmin=-100, vmax=-40)
    plt.colorbar(im, ax=ax, label='dB')
    
    # 标记chirp边界
    symbol_duration_ms = symbol_samples / fs * 1000
    for i in range(1, 20):
        x = i * symbol_duration_ms
        if x < extent[1]:
            if i == 16:
                ax.axvline(x, color='red', linestyle='--', linewidth=2, label='SFD Start')
            elif i == 18:
                ax.axvline(x, color='orange', linestyle='--', linewidth=2, label='SFD End')
            else:
                ax.axvline(x, color='white', linestyle=':', linewidth=0.5, alpha=0.5)
    
    # 添加标注
    preamble_end = 16 * symbol_duration_ms
    ax.text(preamble_end/2, bw/2e3 + 20, f'Preamble\n(16 upchirps)', 
            ha='center', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lime', alpha=0.7))
    ax.text(preamble_end + symbol_duration_ms, bw/2e3 + 20, 'SFD', 
            ha='center', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_title(f'Preamble & SFD (SF{sf}, BW{bw/1e3:.0f}kHz, fs{fs/1e6:.1f}MHz)')
    ax.set_ylim(-bw/2e3 - 30, bw/2e3 + 50)
    ax.legend()
    ax.grid(True, alpha=0.2, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  保存: {save_path}")
    plt.close()


def main():
    file_path = "rawData/dong/1_0_9_11_14_16.bin"
    bw = 125000
    sf = 10  # 假设SF10是对的
    
    print(f"文件: {file_path}")
    print(f"假设 SF{sf}, BW{bw/1e3:.0f}kHz")
    
    data = np.fromfile(file_path, dtype=np.complex64)
    
    # 先用1MHz找到前导码位置
    from scipy.signal import correlate
    symbol_samples_1M = int(1e6 * (2**sf) / bw)
    t = np.arange(symbol_samples_1M) / 1e6
    Ts = symbol_samples_1M / 1e6
    upchirp_1M = np.exp(1j * 2 * np.pi * (-bw/2 * t + (bw/(2*Ts)) * t**2))
    
    corr = correlate(data[:5000000], upchirp_1M, mode='valid')
    packet_start = np.argmax(np.abs(corr)**2)
    print(f"\n找到前导码位置: {packet_start}")
    
    # 测试不同的采样率
    test_fs = [500000, 1000000, 1500000, 2000000, 2500000]
    best, all_results = test_sampling_rate(data, packet_start, bw, sf, test_fs)
    
    # 可视化最佳匹配的几种配置
    print(f"\n生成可视化图...")
    os.makedirs('save_fig', exist_ok=True)
    
    for r in all_results[:3]:  # 前3个结果
        fs = r['fs']
        save_path = f'save_fig/preamble_test_sf{sf}_fs{int(fs/1e6)}M.png'
        visualize_preamble(data, packet_start, fs, bw, sf, save_path)


if __name__ == "__main__":
    main()
