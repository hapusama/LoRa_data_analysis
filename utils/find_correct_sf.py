"""
查找正确的SF配置 - 通过测试不同SF的dechirp效果
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import os
import glob


def test_sf_dechirp(data, packet_start, fs, bw, sf):
    """
    测试指定SF的dechirp效果
    返回: FFT峰值、峰值锐度、峰值集中度
    """
    # 计算符号长度
    symbol_samples = int(fs * (2**sf) / bw)
    
    if packet_start + symbol_samples > len(data):
        return None, None, None
    
    # 提取前导码第一个chirp
    preamble_chirp = data[packet_start:packet_start + symbol_samples]
    
    # 生成对应SF的downchirp
    t = np.arange(symbol_samples) / fs
    Ts = symbol_samples / fs
    downchirp = np.exp(-1j * 2 * np.pi * (-bw/2 * t + (bw/(2*Ts)) * t**2))
    
    # 去啁啾
    dechirped = preamble_chirp * downchirp
    fft_result = np.fft.fft(dechirped)
    fft_mag = np.abs(fft_result)
    
    # 找到峰值
    peak_bin = np.argmax(fft_mag)
    peak_value = fft_mag[peak_bin]
    
    # 计算峰值锐度（峰值与周围平均的比值）
    # 取峰值周围100个bin计算
    window = 100
    start = max(0, peak_bin - window)
    end = min(len(fft_mag), peak_bin + window)
    
    # 排除峰值本身
    surrounding = np.concatenate([fft_mag[start:peak_bin], fft_mag[peak_bin+1:end]])
    avg_surrounding = np.mean(surrounding) if len(surrounding) > 0 else 1e-10
    
    sharpness = peak_value / avg_surrounding
    
    # 计算峰值集中度（峰值能量占总能量的比例）
    total_energy = np.sum(fft_mag**2)
    peak_energy = np.sum(fft_mag[max(0, peak_bin-5):min(len(fft_mag), peak_bin+6)]**2)
    concentration = peak_energy / total_energy if total_energy > 0 else 0
    
    return {
        'sf': sf,
        'symbol_samples': symbol_samples,
        'peak_bin': peak_bin,
        'peak_value': peak_value,
        'sharpness': sharpness,
        'concentration': concentration,
        'fft_mag': fft_mag
    }


def find_preamble_candidates(data, fs, bw, sf, threshold=0.1):
    """使用前导码相关检测找到候选位置"""
    symbol_samples = int(fs * (2**sf) / bw)
    
    # 生成upchirp
    t = np.arange(symbol_samples) / fs
    Ts = symbol_samples / fs
    upchirp = np.exp(1j * 2 * np.pi * (-bw/2 * t + (bw/(2*Ts)) * t**2))
    
    # 相关检测
    corr = correlate(data[:min(len(data), 5000000)], upchirp, mode='valid')
    corr_power = np.abs(corr)**2
    
    # 归一化
    corr_power = corr_power / np.max(corr_power)
    
    # 找峰值
    peaks = np.where(corr_power > threshold)[0]
    
    # 去重
    min_dist = symbol_samples
    unique_peaks = []
    last = -min_dist
    for p in peaks:
        if p - last >= min_dist:
            unique_peaks.append(p)
            last = p
    
    return unique_peaks[:5]  # 返回前5个候选


def analyze_sf_for_file(file_path, fs=1000000, bw=125000):
    """
    分析文件，测试不同SF的dechirp效果
    """
    print(f"\n{'='*70}")
    print(f"文件: {os.path.basename(file_path)}")
    print(f"采样率: {fs/1e6}MHz, 带宽: {bw/1e3}kHz")
    print(f"{'='*70}\n")
    
    # 读取数据
    data = np.fromfile(file_path, dtype=np.complex64)
    print(f"数据长度: {len(data)} 样本")
    
    # 测试的SF范围
    test_sfs = [7, 8, 9, 10, 11, 12]
    
    results = {}
    
    for sf in test_sfs:
        print(f"\n测试 SF{sf}...")
        
        # 找到候选前导码位置
        candidates = find_preamble_candidates(data, fs, bw, sf, threshold=0.05)
        
        if not candidates:
            print(f"  未找到候选前导码")
            continue
        
        print(f"  找到 {len(candidates)} 个候选位置: {candidates[:3]}")
        
        # 测试每个候选位置
        best_result = None
        best_score = 0
        
        for pos in candidates[:3]:  # 只测试前3个
            result = test_sf_dechirp(data, pos, fs, bw, sf)
            if result is None:
                continue
            
            # 综合评分：锐度 * 集中度
            score = result['sharpness'] * result['concentration']
            
            print(f"    位置 {pos}: peak_bin={result['peak_bin']}, "
                  f"sharpness={result['sharpness']:.1f}, "
                  f"concentration={result['concentration']:.3f}, "
                  f"score={score:.1f}")
            
            if score > best_score:
                best_score = score
                best_result = result
                best_result['position'] = pos
        
        if best_result:
            results[sf] = best_result
            print(f"  --> SF{sf} 最佳: score={best_score:.1f}")
    
    # 找出最佳SF
    if not results:
        print("\n错误: 所有SF测试都失败")
        return None
    
    # 按综合评分排序
    best_sf = max(results.keys(), key=lambda k: results[k]['sharpness'] * results[k]['concentration'])
    
    print(f"\n{'='*70}")
    print(f"结论:")
    print(f"{'='*70}")
    
    # 打印所有SF的评分对比
    print(f"\nSF 对比表:")
    print(f"{'SF':<5} {'Position':<12} {'Peak Bin':<10} {'Sharpness':<12} {'Concentration':<15} {'Score':<10}")
    print("-" * 70)
    
    for sf in sorted(results.keys()):
        r = results[sf]
        score = r['sharpness'] * r['concentration']
        print(f"{sf:<5} {r['position']:<12} {r['peak_bin']:<10} {r['sharpness']:<12.1f} {r['concentration']:<15.3f} {score:<10.1f}")
    
    print(f"\n>>> 最佳SF: {best_sf} (score={results[best_sf]['sharpness'] * results[best_sf]['concentration']:.1f})")
    
    # 绘制对比图
    plot_sf_comparison(results, file_path)
    
    return best_sf, results


def plot_sf_comparison(results, file_path):
    """绘制不同SF的FFT对比图"""
    n_sfs = len(results)
    if n_sfs == 0:
        return
    
    # 创建子图
    cols = 3
    rows = (n_sfs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, (sf, r) in enumerate(sorted(results.items())):
        ax = axes[idx]
        
        fft_mag = r['fft_mag']
        peak_bin = r['peak_bin']
        
        # 只显示峰值附近的区域
        window = 500
        start = max(0, peak_bin - window)
        end = min(len(fft_mag), peak_bin + window)
        
        bins = np.arange(start, end)
        ax.plot(bins, fft_mag[start:end], 'b-', linewidth=0.8)
        ax.axvline(peak_bin, color='r', linestyle='--', 
                  label=f'Peak @ bin {peak_bin}')
        
        score = r['sharpness'] * r['concentration']
        ax.set_title(f'SF{sf}: sharpness={r["sharpness"]:.1f}, conc={r["concentration"]:.3f}\nscore={score:.1f}')
        ax.set_xlabel('FFT Bin')
        ax.set_ylabel('Magnitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    save_path = f'save_fig/sf_comparison_{os.path.splitext(os.path.basename(file_path))[0]}.png'
    os.makedirs('save_fig', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n对比图保存到: {save_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="查找正确的SF配置")
    parser.add_argument("--input", default=r"rawData\dong\1_0_9_11_14_16.bin", help=".bin文件")
    parser.add_argument("--bw", type=float, default=125000, help="带宽(Hz)")
    parser.add_argument("--fs", type=float, default=1000000, help="采样率(Hz)")
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.bin")))
    else:
        files = [args.input]
    
    for f in files:
        best_sf, all_results = analyze_sf_for_file(f, args.fs, args.bw)
        
        if best_sf:
            print(f"\n建议参数:")
            print(f"  --sf {best_sf}")
            print(f"  --bw {int(args.bw)}")
            print(f"  --fs {int(args.fs)}")


if __name__ == "__main__":
    main()
