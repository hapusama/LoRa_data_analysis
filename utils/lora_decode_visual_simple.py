"""
LoRa 解码可视化工具 - 简化版，只显示前导码和SFD的时频图
"""
import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

from lora_decoder import LoRaConfig, LoRaReceiver


def plot_preamble_sfd_only(data, packet_start, sfd_pos, cfg, save_path=None, title=""):
    """
    只绘制前导码和SFD的时频图，便于观察chirp对齐
    """
    ns = cfg.symbol_samples
    
    # 只提取前导码 + SFD + 一点点header的区域
    # 前导码16个符号 + SFD(2.25符号) + 几个header符号用于对比
    preamble_symbols = cfg.preamble_symbols
    n_symbols_show = preamble_symbols + 4  # 16 preamble + 2 SFD + 几个header
    
    show_start = packet_start
    show_end = min(len(data), packet_start + n_symbols_show * ns + ns//4)
    segment = data[show_start:show_end]
    
    # 创建图形 - 2行1列
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # ========== 子图1: 时频图 ==========
    ax1 = axes[0]
    
    # 计算合适的nperseg - 要能清晰显示chirp
    nperseg = min(256, ns // 4)
    noverlap = nperseg // 2
    
    f, t, Sxx = spectrogram(segment, fs=cfg.fs, nperseg=nperseg, 
                            noverlap=noverlap, return_onesided=False)
    Sxx_shift = np.fft.fftshift(Sxx, axes=0)
    f_shift = np.fft.fftshift(f)
    
    # 转换为ms时间
    t_ms = t * 1000
    extent = [t_ms[0], t_ms[-1], f_shift[0]/1e3, f_shift[-1]/1e3]
    
    im = ax1.imshow(10*np.log10(np.abs(Sxx_shift)+1e-20), aspect='auto', 
                    origin='lower', extent=extent, cmap='viridis', vmin=-100, vmax=-40)
    plt.colorbar(im, ax=ax1, label='dB', pad=0.02)
    
    # 标记前导码和SFD的边界
    symbol_duration_ms = ns / cfg.fs * 1000
    
    # 标记每个chirp的边界
    for i in range(1, preamble_symbols + 3):
        x = i * symbol_duration_ms
        if x < extent[1]:
            if i == preamble_symbols:
                # SFD开始 - 红色虚线
                ax1.axvline(x, color='red', linestyle='--', linewidth=2, 
                           label='SFD Start' if i == preamble_symbols else '')
            elif i == preamble_symbols + 2:
                # SFD结束 - 橙色虚线
                ax1.axvline(x, color='orange', linestyle='--', linewidth=2,
                           label='SFD End' if i == preamble_symbols + 2 else '')
            else:
                # 普通前导码边界 - 白色细线
                ax1.axvline(x, color='white', linestyle=':', linewidth=0.5, alpha=0.5)
    
    # 添加文本标注
    preamble_end_ms = preamble_symbols * symbol_duration_ms
    sfd_end_ms = (preamble_symbols + 2.25) * symbol_duration_ms
    
    ax1.text(preamble_end_ms / 2, cfg.bw/2e3 + 15, 
             f'Preamble\n({preamble_symbols} upchirps)', 
             ha='center', va='bottom', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lime', alpha=0.7))
    
    ax1.text((preamble_end_ms + sfd_end_ms) / 2, cfg.bw/2e3 + 15,
             'SFD\n(2.25 downchirps)', 
             ha='center', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Frequency (kHz)', fontsize=12)
    ax1.set_title(f'{title} - Preamble & SFD Spectrogram (SF{cfg.sf}, BW{cfg.bw/1e3:.0f}kHz)', fontsize=14)
    ax1.set_ylim(-cfg.bw/2e3 - 30, cfg.bw/2e3 + 40)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2, linestyle=':')
    
    # ========== 子图2: 单个chirp分析 ==========
    ax2 = axes[1]
    
    # 提取第一个前导码chirp和SFD的第一个chirp进行对比
    preamble_chirp = data[packet_start:packet_start + ns]
    sfd_chirp = data[sfd_pos:sfd_pos + ns]
    
    # 生成downchirp用于去啁啾
    t_chirp = np.arange(ns) / cfg.fs
    Ts = ns / cfg.fs
    downchirp = np.exp(-1j * 2 * np.pi * (-cfg.bw/2 * t_chirp + (cfg.bw/(2*Ts)) * t_chirp**2))
    
    # 去啁啾
    preamble_dechirped = preamble_chirp * downchirp
    sfd_dechirped = sfd_chirp * np.conj(downchirp)  # SFD是downchirp，所以用conj(up)=down
    
    # FFT
    fft_preamble = np.abs(np.fft.fft(preamble_dechirped))
    fft_sfd = np.abs(np.fft.fft(sfd_dechirped))
    
    bins = np.arange(len(fft_preamble))
    
    # 绘制FFT对比
    ax2.plot(bins, fft_preamble, 'b-', linewidth=1, label='Preamble (upchirp)', alpha=0.8)
    ax2.plot(bins, fft_sfd, 'r-', linewidth=1, label='SFD (downchirp)', alpha=0.8)
    
    # 标记峰值
    peak_preamble = np.argmax(fft_preamble)
    peak_sfd = np.argmax(fft_sfd)
    
    ax2.axvline(peak_preamble, color='blue', linestyle='--', alpha=0.7,
               label=f'Preamble peak @ bin {peak_preamble}')
    ax2.axvline(peak_sfd, color='red', linestyle='--', alpha=0.7,
               label=f'SFD peak @ bin {peak_sfd}')
    
    ax2.set_xlabel('FFT Bin', fontsize=12)
    ax2.set_ylabel('Magnitude', fontsize=12)
    ax2.set_title('Dechirped FFT Comparison (peaks should be close to bin 0 if aligned)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 限制x轴显示范围，只看峰值附近
    margin = 200
    ax2.set_xlim(0, min(len(bins), max(peak_preamble, peak_sfd) + margin))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存简化分析图: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # 打印对齐信息
    print(f"    对齐分析:")
    print(f"      - Preamble FFT 峰值: bin {peak_preamble} (应在 0 附近)")
    print(f"      - SFD FFT 峰值: bin {peak_sfd} (应在 0 附近)")
    if abs(peak_preamble) < 100 and abs(peak_sfd) < 100:
        print(f"      - [OK] 对齐良好!")
    else:
        print(f"      - [WARN] 可能存在频率偏移或采样率偏差")


def decode_with_simple_visualization(file_path, cfg, detect_threshold=0.1, 
                                     upchirp_threshold=0.1, save_dir='save_fig', 
                                     max_packets=5):
    """
    带简化可视化的解码函数
    """
    print(f"\n{'='*60}")
    print(f"处理文件: {os.path.basename(file_path)}")
    print(f"参数: SF{cfg.sf}, BW{cfg.bw/1e3:.0f}kHz, fs{cfg.fs/1e3:.0f}kHz, CR=4/{4+cfg.coding_rate}")
    print(f"{'='*60}\n")
    
    # 读取数据
    data = np.fromfile(file_path, dtype=np.complex64)
    print(f"数据长度: {len(data)} 样本 ({len(data)/cfg.fs:.3f}s @ {cfg.fs/1e3:.0f}kHz)")
    
    # 创建接收器
    rx = LoRaReceiver(cfg, detect_threshold=detect_threshold, 
                      upchirp_corr_threshold=upchirp_threshold)
    
    # 执行同步检测
    print("\n[1/2] 执行前导码同步检测...")
    syncs = rx.sync.detect(data)
    print(f"      检测到 {len(syncs)} 个潜在数据包")
    
    if not syncs:
        print("\n警告: 未检测到数据包，尝试降低阈值...")
        return []
    
    # 创建输出目录
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # 处理每个检测到的包
    frames = []
    
    print("\n[2/2] 生成简化可视化图...")
    
    for i, s in enumerate(syncs[:max_packets]):
        print(f"\n  数据包 #{i+1}:")
        print(f"    起始位置: {s.packet_start}")
        print(f"    SFD 位置: {s.sfd_pos}")
        print(f"    同步分数: {s.score:.4f}")
        
        # 生成简化可视化
        save_path = os.path.join(save_dir, f'{base_name}_pkt{i+1}_preamble.png')
        plot_preamble_sfd_only(data, s.packet_start, s.sfd_pos, cfg, save_path,
                              title=f"Packet {i+1}")
        
        # 尝试解码header
        ns = cfg.symbol_samples
        header_words = rx.demod.extract_words(data, start=s.header_start, 
                                               n_symbols=8, reduced_rate=True)
        if len(header_words) >= 8:
            decoded_header = rx.bitpipe.decode_block(header_words, ppm=cfg.sf-2, 
                                                      is_header=True, coding_rate=cfg.coding_rate)
            header = rx._parse_header(decoded_header)
            if header:
                print(f"    Header: len={header.length}, CR=4/{4+header.coding_rate}, CRC={'有' if header.has_crc else '无'}")
    
    print(f"\n完成! 共生成 {min(len(syncs), max_packets)} 个包的可视化图")
    print(f"图表保存在: {save_dir}/")
    
    return syncs


def main():
    parser = argparse.ArgumentParser(description="LoRa 简化可视化解码工具")
    parser.add_argument("--input", default=r"rawData\dong", help=".bin 文件或文件夹")
    parser.add_argument("--sf", type=int, default=10, help="扩频因子 (6-12)")
    parser.add_argument("--bw", type=float, default=125000, help="带宽 (Hz)")
    parser.add_argument("--fs", type=float, default=1000000, help="采样率 (Hz)")
    parser.add_argument("--preamble", type=int, default=16, help="前导码符号数")
    parser.add_argument("--cr", type=int, default=2, help="编码率 1..4 对应 4/5..4/8")
    parser.add_argument("--detect-threshold", type=float, default=0.05, help="检测阈值")
    parser.add_argument("--upchirp-threshold", type=float, default=0.05, help="upchirp 相关阈值")
    parser.add_argument("--output-dir", default="save_fig", help="输出图表目录")
    parser.add_argument("--max-packets", type=int, default=3, help="最多处理几个包")
    args = parser.parse_args()
    
    cfg = LoRaConfig(
        sf=args.sf,
        bw=args.bw,
        fs=args.fs,
        preamble_symbols=args.preamble,
        implicit_header=False,
        coding_rate=args.cr,
        has_crc=True,
    )
    
    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.bin")))
    else:
        files = [args.input]
    
    for path in files:
        decode_with_simple_visualization(
            path, cfg,
            detect_threshold=args.detect_threshold,
            upchirp_threshold=args.upchirp_threshold,
            save_dir=args.output_dir,
            max_packets=args.max_packets
        )


if __name__ == "__main__":
    main()
