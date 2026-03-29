"""
LoRa 信号诊断脚本 - 分析原始数据特征
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, correlate
import os

def analyze_lora_signal(file_path, sf=9, bw=125000, fs=1000000, preamble_symbols=16, output_dir='save_fig'):
    """分析 LoRa 信号并生成诊断图"""
    
    # 读取数据
    data = np.fromfile(file_path, dtype=np.complex64)
    print(f"文件名: {os.path.basename(file_path)}")
    print(f"总采样数: {len(data)}")
    print(f"采样率假设: {fs/1e6:.2f} MHz")
    print(f"带宽: {bw/1e3:.0f} kHz")
    print(f"扩频因子: SF{sf}")
    print(f"前导码符号数: {preamble_symbols}")
    print()
    
    # 计算符号采样数
    symbol_samples = int(fs * (2**sf) / bw)
    print(f"符号采样数: {symbol_samples}")
    
    # 生成 upchirp 和 downchirp
    t = np.arange(symbol_samples) / fs
    Ts = symbol_samples / fs
    upchirp = np.exp(1j * 2 * np.pi * (-bw/2 * t + (bw / (2*Ts)) * t**2))
    downchirp = np.conj(upchirp)
    
    # 1. 绘制整体频谱图
    print("生成整体频谱图...")
    nperseg = min(1024, len(data)//100)
    f, t_spec, Sxx = spectrogram(data, fs=fs, nperseg=nperseg, noverlap=nperseg//2, return_onesided=False)
    Sxx_shift = np.fft.fftshift(Sxx, axes=0)
    f_shift = np.fft.fftshift(f)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    
    # IQ 数据时域
    axes[0].plot(np.real(data[:min(len(data), 500000)]), 'b-', alpha=0.7, label='I')
    axes[0].plot(np.imag(data[:min(len(data), 500000)]), 'r-', alpha=0.7, label='Q')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('IQ Data (first 500k samples)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 频谱图
    extent = [t_spec[0], t_spec[-1], f_shift[0]/1e3, f_shift[-1]/1e3]
    im = axes[1].imshow(10*np.log10(np.abs(Sxx_shift)+1e-20), aspect='auto', origin='lower', 
                        extent=extent, cmap='viridis')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (kHz)')
    axes[1].set_title('Spectrogram')
    axes[1].set_ylim(-bw/2e3-50, bw/2e3+50)
    plt.colorbar(im, ax=axes[1], label='dB')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_spectrogram.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存到: {output_dir}/overall_spectrogram.png")
    
    # 2. 使用 upchirp 相关检测前导码
    print("\n检测前导码位置...")
    
    # 只取前 2M 采样进行快速检测
    search_data = data[:min(len(data), 2000000)]
    
    # 相关检测
    corr = correlate(search_data, upchirp, mode='valid')
    corr_power = np.abs(corr)**2
    
    # 归一化
    corr_power_norm = corr_power / np.max(corr_power)
    
    # 找峰值
    threshold = 0.1
    peaks = np.where(corr_power_norm > threshold)[0]
    
    print(f"  相关阈值: {threshold}")
    print(f"  检测到的峰值数: {len(peaks)}")
    
    # 去重峰值（间隔至少一个符号长度）
    min_dist = symbol_samples
    unique_peaks = []
    last_peak = -min_dist
    for p in peaks:
        if p - last_peak >= min_dist:
            unique_peaks.append(p)
            last_peak = p
    
    print(f"  去重后的峰值数: {len(unique_peaks)}")
    
    if unique_peaks:
        print(f"  前5个峰值位置: {unique_peaks[:5]}")
        
        # 3. 分析第一个峰值附近
        first_peak = unique_peaks[0]
        print(f"\n分析第一个峰值 (位置: {first_peak})...")
        
        # 提取前导码区域
        preamble_start = max(0, first_peak - symbol_samples//2)
        preamble_end = min(len(data), preamble_start + preamble_symbols * symbol_samples * 2)
        preamble_data = data[preamble_start:preamble_end]
        
        # 绘制前导码区域频谱图
        nperseg_p = min(256, len(preamble_data)//10)
        f_p, t_p, Sxx_p = spectrogram(preamble_data, fs=fs, nperseg=nperseg_p, 
                                       noverlap=nperseg_p//2, return_onesided=False)
        Sxx_p_shift = np.fft.fftshift(Sxx_p, axes=0)
        f_p_shift = np.fft.fftshift(f_p)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 前导码频谱图
        extent_p = [t_p[0]*1000, t_p[-1]*1000, f_p_shift[0]/1e3, f_p_shift[-1]/1e3]
        im = axes[0].imshow(10*np.log10(np.abs(Sxx_p_shift)+1e-20), aspect='auto', origin='lower',
                            extent=extent_p, cmap='viridis')
        axes[0].set_xlabel('Time (ms)')
        axes[0].set_ylabel('Frequency (kHz)')
        axes[0].set_title(f'Preamble Region Spectrogram (around sample {first_peak})')
        axes[0].set_ylim(-bw/2e3-20, bw/2e3+20)
        plt.colorbar(im, ax=axes[0], label='dB')
        
        # 标记预期的 chirp 边界
        for i in range(preamble_symbols + 5):
            x = (i * symbol_samples / fs) * 1000
            if x < extent_p[1]:
                axes[0].axvline(x, color='w', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # 相关峰图
        axes[1].plot(corr_power_norm[:min(len(corr_power_norm), 500000)], 'b-', linewidth=0.5)
        axes[1].axhline(threshold, color='r', linestyle='--', label=f'threshold={threshold}')
        axes[1].axvline(first_peak, color='g', linestyle='-', label=f'first peak={first_peak}')
        axes[1].set_xlabel('Sample')
        axes[1].set_ylabel('Normalized Correlation Power')
        axes[1].set_title('Upchirp Correlation (first 500k)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'preamble_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  保存到: {output_dir}/preamble_analysis.png")
        
        # 4. 分析单个 chirp
        print("\n分析单个 upchirp...")
        single_chirp = data[first_peak:first_peak + symbol_samples]
        
        # 去啁啾
        dechirped = single_chirp * downchirp
        fft_dechirped = np.fft.fft(dechirped)
        fft_mag = np.abs(fft_dechirped)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 原始 chirp 时域
        axes[0, 0].plot(np.real(single_chirp), 'b-', alpha=0.7, label='I')
        axes[0, 0].plot(np.imag(single_chirp), 'r-', alpha=0.7, label='Q')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title('Single Upchirp (Time Domain)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 原始 chirp 频谱
        freqs = np.fft.fftfreq(len(single_chirp), 1/fs) / 1e3
        axes[0, 1].plot(freqs, 20*np.log10(np.abs(np.fft.fft(single_chirp))+1e-20))
        axes[0, 1].set_xlabel('Frequency (kHz)')
        axes[0, 1].set_ylabel('Power (dB)')
        axes[0, 1].set_title('Single Upchirp Spectrum')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim(-bw/1e3-50, bw/1e3+50)
        
        # 去啁啾后时域
        axes[1, 0].plot(np.real(dechirped), 'b-', alpha=0.7, label='I')
        axes[1, 0].plot(np.imag(dechirped), 'r-', alpha=0.7, label='Q')
        axes[1, 0].set_xlabel('Sample')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].set_title('Dechirped (Time Domain)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # FFT 峰值
        axes[1, 1].plot(fft_mag)
        axes[1, 1].set_xlabel('Bin')
        axes[1, 1].set_ylabel('Magnitude')
        axes[1, 1].set_title(f'FFT of Dechirped (Peak at bin {np.argmax(fft_mag)})')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'single_chirp_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  保存到: {output_dir}/single_chirp_analysis.png")
        
        # 5. 检查 SFD 位置
        print("\n检查 SFD 位置...")
        expected_sfd = first_peak + preamble_symbols * symbol_samples
        search_start = max(0, expected_sfd - symbol_samples//2)
        search_end = min(len(data), expected_sfd + symbol_samples//2)
        
        sfd_search_data = data[search_start:search_end]
        sfd_corr = correlate(sfd_search_data, downchirp, mode='valid')
        sfd_peak = np.argmax(np.abs(sfd_corr)**2)
        actual_sfd = search_start + sfd_peak
        
        print(f"  预期 SFD 位置: {expected_sfd}")
        print(f"  实际 SFD 位置: {actual_sfd}")
        print(f"  偏差: {actual_sfd - expected_sfd} samples ({(actual_sfd - expected_sfd)/symbol_samples*100:.1f}% of symbol)")
        
        # 6. 测试不同采样率假设
        print("\n\n===== 不同采样率假设测试 =====")
        for test_fs in [500000, 1000000, 2000000]:
            test_symbol_samples = int(test_fs * (2**sf) / bw)
            test_t = np.arange(test_symbol_samples) / test_fs
            test_Ts = test_symbol_samples / test_fs
            test_upchirp = np.exp(1j * 2 * np.pi * (-bw/2 * test_t + (bw / (2*test_Ts)) * test_t**2))
            
            # 相关检测
            test_corr = correlate(data[:1000000], test_upchirp, mode='valid')
            test_corr_power = np.abs(test_corr)**2
            test_max_power = np.max(test_corr_power)
            
            print(f"  fs={test_fs/1e6:.1f}MHz, symbol_samples={test_symbol_samples}: max correlation power = {test_max_power:.2e}")
    
    print("\n诊断完成!")

if __name__ == "__main__":
    import sys
    
    file_path = "rawData/dong/1_0_9_11_14_16.bin"
    
    # 从文件名解析参数: 1_0_9_11_14_16
    # 猜测: SF=9, BW=125000, preamble=16, CR=4/6 (编码率2)
    
    os.makedirs('save_fig', exist_ok=True)
    
    analyze_lora_signal(
        file_path=file_path,
        sf=9,
        bw=125000,
        fs=1000000,  # 尝试 1MHz
        preamble_symbols=16,
        output_dir='save_fig'
    )
