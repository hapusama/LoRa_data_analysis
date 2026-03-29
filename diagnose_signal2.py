"""
LoRa 信号诊断脚本 - 检查中心频率偏移和参数匹配
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, correlate
import os

def find_frequency_offset(data, fs, bw, sf):
    """通过 upchirp 检测中心频率偏移"""
    symbol_samples = int(fs * (2**sf) / bw)
    
    # 生成参考 upchirp
    t = np.arange(symbol_samples) / fs
    Ts = symbol_samples / fs
    upchirp = np.exp(1j * 2 * np.pi * (-bw/2 * t + (bw / (2*Ts)) * t**2))
    
    # 分段相关检测
    segment_len = symbol_samples * 2
    offsets = []
    
    for start in range(0, min(len(data) - segment_len, 10000000), segment_len):
        segment = data[start:start + segment_len]
        
        # FFT 相关
        corr = correlate(segment, upchirp, mode='valid')
        if len(corr) > 0:
            peak_idx = np.argmax(np.abs(corr)**2)
            if peak_idx < len(corr):
                # 提取检测到的 chirp
                detected_chirp = segment[peak_idx:peak_idx + symbol_samples]
                
                # 去啁啾
                dechirped = detected_chirp * np.conj(upchirp)
                fft_result = np.fft.fft(dechirped)
                fft_mag = np.abs(fft_result)
                
                # 找到峰值 bin
                peak_bin = np.argmax(fft_mag)
                
                # 计算频率偏移
                freq_resolution = fs / symbol_samples
                freq_offset = peak_bin * freq_resolution
                if freq_offset > fs/2:
                    freq_offset -= fs
                    
                offsets.append(freq_offset)
    
    return np.median(offsets) if offsets else 0

def test_sf_configurations(data, fs, bw):
    """测试不同 SF 配置的相关性"""
    results = []
    
    for sf in range(7, 13):
        symbol_samples = int(fs * (2**sf) / bw)
        if symbol_samples > len(data) // 10:
            continue
            
        t = np.arange(symbol_samples) / fs
        Ts = symbol_samples / fs
        upchirp = np.exp(1j * 2 * np.pi * (-bw/2 * t + (bw / (2*Ts)) * t**2))
        
        # 相关检测
        test_data = data[:min(len(data), 1000000)]
        corr = correlate(test_data, upchirp, mode='valid')
        max_power = np.max(np.abs(corr)**2)
        
        results.append({
            'sf': sf,
            'symbol_samples': symbol_samples,
            'max_power': max_power,
            'chirp_duration_ms': symbol_samples / fs * 1000
        })
    
    return results

def analyze_chirp_bandwidth(data, peak_pos, fs, bw, sf):
    """分析检测到的 chirp 的实际带宽"""
    symbol_samples = int(fs * (2**sf) / bw)
    
    # 提取 chirp
    chirp = data[peak_pos:peak_pos + symbol_samples]
    
    # 计算频谱
    fft_result = np.fft.fft(chirp)
    freqs = np.fft.fftfreq(len(chirp), 1/fs)
    
    # 找到 -3dB 带宽
    fft_mag = np.abs(fft_result)
    fft_mag_db = 20 * np.log10(fft_mag + 1e-20)
    
    peak_mag = np.max(fft_mag_db)
    threshold = peak_mag - 3
    
    # 找到高于阈值的频率范围
    above_threshold = freqs[fft_mag_db > threshold]
    if len(above_threshold) > 0:
        actual_bw = np.max(above_threshold) - np.min(above_threshold)
    else:
        actual_bw = 0
    
    return freqs, fft_mag_db, actual_bw

def main():
    file_path = "rawData/dong/1_0_9_11_14_16.bin"
    
    # 从文件名解析
    # 1_0_9_11_14_16: 可能是 SF=9, BW=125k (编码11?), CR=4/6 (编码14?), preamble=16
    
    print("=" * 60)
    print("LoRa 信号深度诊断")
    print("=" * 60)
    
    data = np.fromfile(file_path, dtype=np.complex64)
    print(f"\n文件: {file_path}")
    print(f"总采样数: {len(data)}")
    
    # 测试不同的 SF 和采样率组合
    configs = [
        {'sf': 9, 'bw': 125000, 'fs': 1000000, 'name': 'SF9 BW125k fs=1M'},
        {'sf': 9, 'bw': 125000, 'fs': 500000, 'name': 'SF9 BW125k fs=500k'},
        {'sf': 10, 'bw': 125000, 'fs': 1000000, 'name': 'SF10 BW125k fs=1M'},
        {'sf': 11, 'bw': 125000, 'fs': 1000000, 'name': 'SF11 BW125k fs=1M'},
        {'sf': 9, 'bw': 250000, 'fs': 1000000, 'name': 'SF9 BW250k fs=1M'},
    ]
    
    print("\n" + "-" * 60)
    print("测试不同配置的相关性")
    print("-" * 60)
    
    best_config = None
    best_power = 0
    
    for cfg in configs:
        sf, bw, fs = cfg['sf'], cfg['bw'], cfg['fs']
        symbol_samples = int(fs * (2**sf) / bw)
        
        if symbol_samples > len(data):
            continue
        
        # 生成 upchirp
        t = np.arange(symbol_samples) / fs
        Ts = symbol_samples / fs
        upchirp = np.exp(1j * 2 * np.pi * (-bw/2 * t + (bw / (2*Ts)) * t**2))
        
        # 相关检测
        test_data = data[:min(len(data), 2000000)]
        corr = correlate(test_data, upchirp, mode='valid')
        max_power = np.max(np.abs(corr)**2)
        
        # 找到峰值位置
        peak_idx = np.argmax(np.abs(corr)**2)
        
        print(f"\n{cfg['name']}:")
        print(f"  Symbol samples: {symbol_samples}")
        print(f"  Max correlation power: {max_power:.4e}")
        print(f"  Peak position: {peak_idx}")
        
        if max_power > best_power:
            best_power = max_power
            best_config = {**cfg, 'symbol_samples': symbol_samples, 'peak_idx': peak_idx}
    
    if best_config:
        print("\n" + "=" * 60)
        print(f"最佳配置: {best_config['name']}")
        print("=" * 60)
        
        sf = best_config['sf']
        bw = best_config['bw']
        fs = best_config['fs']
        symbol_samples = best_config['symbol_samples']
        peak_idx = best_config['peak_idx']
        
        # 分析频率偏移
        print("\n分析频率偏移...")
        freq_offset = find_frequency_offset(data[:10000000], fs, bw, sf)
        print(f"  估计的频率偏移: {freq_offset/1e3:.2f} kHz")
        
        # 如果有频率偏移，校正后重新分析
        if abs(freq_offset) > 1000:
            print(f"\n  检测到显著频率偏移，进行校正...")
            t = np.arange(len(data)) / fs
            data_corrected = data * np.exp(-1j * 2 * np.pi * freq_offset * t)
        else:
            data_corrected = data
            
        # 分析 chirp 带宽
        print("\n分析实际 chirp 带宽...")
        freqs, fft_mag_db, actual_bw = analyze_chirp_bandwidth(
            data_corrected, peak_idx, fs, bw, sf
        )
        print(f"  配置的 BW: {bw/1e3:.0f} kHz")
        print(f"  实际的 BW (approx): {actual_bw/1e3:.1f} kHz")
        
        # 绘制分析图
        os.makedirs('save_fig', exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 去啁啾后的 FFT
        chirp = data_corrected[peak_idx:peak_idx + symbol_samples]
        t_chirp = np.arange(symbol_samples) / fs
        Ts = symbol_samples / fs
        upchirp = np.exp(1j * 2 * np.pi * (-bw/2 * t_chirp + (bw / (2*Ts)) * t_chirp**2))
        dechirped = chirp * np.conj(upchirp)
        fft_result = np.fft.fft(dechirped)
        
        axes[0, 0].plot(np.abs(fft_result))
        axes[0, 0].set_xlabel('FFT Bin')
        axes[0, 0].set_ylabel('Magnitude')
        axes[0, 0].set_title('Dechirped FFT (corrected)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 频谱
        axes[0, 1].plot(freqs/1e3, fft_mag_db)
        axes[0, 1].set_xlabel('Frequency (kHz)')
        axes[0, 1].set_ylabel('Power (dB)')
        axes[0, 1].set_title('Detected Chirp Spectrum')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim(-bw/1e3 - 50, bw/1e3 + 50)
        
        # 3. 前导码区域频谱图
        preamble_start = max(0, peak_idx - symbol_samples)
        preamble_end = min(len(data_corrected), peak_idx + 20 * symbol_samples)
        preamble_data = data_corrected[preamble_start:preamble_end]
        
        nperseg = min(256, len(preamble_data)//20)
        if nperseg >= 64:
            f, t_spec, Sxx = spectrogram(preamble_data, fs=fs, nperseg=nperseg, 
                                          noverlap=nperseg//2, return_onesided=False)
            Sxx_shift = np.fft.fftshift(Sxx, axes=0)
            f_shift = np.fft.fftshift(f)
            
            extent = [t_spec[0]*1000, t_spec[-1]*1000, f_shift[0]/1e3, f_shift[-1]/1e3]
            axes[1, 0].imshow(10*np.log10(np.abs(Sxx_shift)+1e-20), aspect='auto', 
                              origin='lower', extent=extent, cmap='viridis')
            axes[1, 0].set_xlabel('Time (ms)')
            axes[1, 0].set_ylabel('Frequency (kHz)')
            axes[1, 0].set_title('Preamble Spectrogram (corrected)')
            axes[1, 0].set_ylim(-bw/2e3 - 30, bw/2e3 + 30)
        
        # 4. 前导码检测的峰值位置
        test_data = data_corrected[:min(len(data_corrected), 5000000)]
        corr = correlate(test_data, upchirp, mode='valid')
        corr_power = np.abs(corr)**2
        
        # 找所有峰值
        threshold = np.max(corr_power) * 0.1
        peaks = []
        for i in range(1, len(corr_power)-1):
            if corr_power[i] > threshold and corr_power[i] > corr_power[i-1] and corr_power[i] > corr_power[i+1]:
                peaks.append(i)
        
        # 去重
        min_dist = symbol_samples
        unique_peaks = []
        last = -min_dist
        for p in peaks:
            if p - last >= min_dist:
                unique_peaks.append(p)
                last = p
        
        axes[1, 1].plot(corr_power[:500000], 'b-', linewidth=0.5)
        for p in unique_peaks[:10]:
            if p < 500000:
                axes[1, 1].axvline(p, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Sample')
        axes[1, 1].set_ylabel('Correlation Power')
        axes[1, 1].set_title('Preamble Detection Peaks')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('save_fig/detailed_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n分析图保存到: save_fig/detailed_analysis.png")
        
        # 输出推荐参数
        print("\n" + "=" * 60)
        print("推荐的解码参数:")
        print("=" * 60)
        print(f"  --sf {sf}")
        print(f"  --bw {bw}")
        print(f"  --fs {fs}")
        print(f"  --preamble 16")
        print(f"  --cr 2  # 4/6 编码率")
        if abs(freq_offset) > 1000:
            print(f"\n  注意: 检测到 {freq_offset/1e3:.2f} kHz 频率偏移")
            print(f"  建议在解调前进行频率校正")

if __name__ == "__main__":
    main()
