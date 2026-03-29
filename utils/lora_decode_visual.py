"""
LoRa 解码可视化工具 - 带时频图显示的解码器
结合了解码功能和 bin_read.py 的可视化能力
"""
import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, correlate

from lora_decoder import LoRaConfig, LoRaReceiver


def plot_packet_analysis(data, packet_start, sfd_pos, header_start, payload_start, 
                         frame_end, cfg, save_path=None, title=""):
    """
    绘制单个包的完整分析图，包括：
    1. 前导码区域时频图
    2. SFD 位置标记
    3. Header 区域
    4. Payload 区域
    5. 去啁啾后的 FFT 峰值
    """
    ns = cfg.symbol_samples
    
    # 计算显示范围（前导码前一些到 payload 结束）
    show_start = max(0, packet_start - ns // 2)
    show_end = min(len(data), frame_end + ns // 2)
    segment = data[show_start:show_end]
    
    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
    
    # ========== 子图1: 整体时频图 ==========
    ax1 = fig.add_subplot(gs[0, :])
    
    nperseg = min(256, len(segment) // 20)
    if nperseg >= 64:
        f, t, Sxx = spectrogram(segment, fs=cfg.fs, nperseg=nperseg, 
                                noverlap=nperseg//2, return_onesided=False)
        Sxx_shift = np.fft.fftshift(Sxx, axes=0)
        f_shift = np.fft.fftshift(f)
        
        extent = [t[0]*1000, t[-1]*1000, f_shift[0]/1e3, f_shift[-1]/1e3]
        im = ax1.imshow(10*np.log10(np.abs(Sxx_shift)+1e-20), aspect='auto', 
                        origin='lower', extent=extent, cmap='viridis', vmin=-100, vmax=-40)
        plt.colorbar(im, ax=ax1, label='dB')
        
        # 标记关键位置
        ax1.axvline((packet_start - show_start)/cfg.fs*1000, color='lime', 
                   linestyle='--', linewidth=2, label='Preamble Start')
        ax1.axvline((sfd_pos - show_start)/cfg.fs*1000, color='red', 
                   linestyle='--', linewidth=2, label='SFD')
        ax1.axvline((header_start - show_start)/cfg.fs*1000, color='yellow', 
                   linestyle='--', linewidth=2, label='Header Start')
        ax1.axvline((payload_start - show_start)/cfg.fs*1000, color='cyan', 
                   linestyle='--', linewidth=2, label='Payload Start')
        
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Frequency (kHz)')
        ax1.set_title(f'{title} - Spectrogram (SF{cfg.sf}, BW{cfg.bw/1e3:.0f}kHz)')
        ax1.set_ylim(-cfg.bw/2e3 - 20, cfg.bw/2e3 + 20)
        ax1.legend(loc='upper right')
    
    # ========== 子图2: 前导码去啁啾分析 ==========
    ax2 = fig.add_subplot(gs[1, 0])
    
    # 提取前导码第一个 chirp
    preamble_chirp = data[packet_start:packet_start + ns]
    t_chirp = np.arange(ns) / cfg.fs
    Ts = ns / cfg.fs
    
    # 生成 downchirp
    downchirp = np.exp(-1j * 2 * np.pi * (-cfg.bw/2 * t_chirp + (cfg.bw/(2*Ts)) * t_chirp**2))
    
    # 去啁啾
    dechirped = preamble_chirp * downchirp
    fft_result = np.fft.fft(dechirped)
    fft_mag = np.abs(fft_result)
    
    # 绘制 FFT
    bins = np.arange(len(fft_mag))
    ax2.plot(bins, fft_mag, 'b-', linewidth=0.8)
    peak_bin = np.argmax(fft_mag)
    ax2.axvline(peak_bin, color='r', linestyle='--', 
               label=f'Peak at bin {peak_bin}')
    ax2.set_xlabel('FFT Bin')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Preamble Dechirped FFT (should peak at bin 0 if aligned)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ========== 子图3: SFD 去啁啾分析 ==========
    ax3 = fig.add_subplot(gs[1, 1])
    
    # 提取 SFD 区域
    sfd_chirp = data[sfd_pos:sfd_pos + ns]
    dechirped_sfd = sfd_chirp * downchirp
    fft_sfd = np.abs(np.fft.fft(dechirped_sfd))
    
    ax3.plot(bins, fft_sfd, 'g-', linewidth=0.8)
    peak_bin_sfd = np.argmax(fft_sfd)
    ax3.axvline(peak_bin_sfd, color='r', linestyle='--', 
               label=f'Peak at bin {peak_bin_sfd}')
    ax3.set_xlabel('FFT Bin')
    ax3.set_ylabel('Magnitude')
    ax3.set_title('SFD Dechirped FFT (should peak at bin 0 for downchirp)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ========== 子图4: 前导码相位连续性 ==========
    ax4 = fig.add_subplot(gs[2, 0])
    
    # 提取多个前导码 chirp 并显示相位
    n_chirps_show = min(8, (sfd_pos - packet_start) // ns)
    colors = plt.cm.viridis(np.linspace(0, 1, n_chirps_show))
    
    for i in range(n_chirps_show):
        chirp_start = packet_start + i * ns
        chirp = data[chirp_start:chirp_start + ns]
        phase = np.unwrap(np.angle(chirp))
        # 去趋势
        phase = phase - np.linspace(phase[0], phase[-1], len(phase))
        t_ms = np.arange(len(phase)) / cfg.fs * 1000
        ax4.plot(t_ms, phase, color=colors[i], alpha=0.7, label=f'Chirp {i+1}' if i < 3 else '')
    
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Phase (rad)')
    ax4.set_title('Preamble Phase Continuity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ========== 子图5: Header/Payload 符号分析 ==========
    ax5 = fig.add_subplot(gs[2, 1])
    
    # 显示前几个 header/payload 符号的 FFT 峰值
    if payload_start > header_start:
        n_symbols_show = min(16, (payload_start - header_start) // ns)
        symbol_peaks = []
        
        for i in range(n_symbols_show):
            sym_start = header_start + i * ns
            if sym_start + ns > len(data):
                break
            symbol = data[sym_start:sym_start + ns]
            
            # 根据是否是 header 使用不同的 downchirp
            if cfg.implicit_header or i >= 8:
                # Payload 使用完整 SF
                downchirp_sym = downchirp
            else:
                # Header 使用 reduced rate (SF-2)
                ns_header = int(ns / 4)  # 简化处理
                downchirp_sym = downchirp
            
            dechirped_sym = symbol * downchirp_sym
            fft_sym = np.abs(np.fft.fft(dechirped_sym))
            peak = np.argmax(fft_sym)
            symbol_peaks.append(peak)
        
        ax5.stem(range(len(symbol_peaks)), symbol_peaks, basefmt=' ')
        ax5.axhline(cfg.num_bins//2, color='r', linestyle='--', alpha=0.5, label='Mid bin')
        ax5.set_xlabel('Symbol Index')
        ax5.set_ylabel('Peak Bin')
        ax5.set_title('Header Symbol Peaks (first 8 should be reduced rate)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存分析图: {save_path}")
    else:
        plt.show()
    
    plt.close()


def decode_with_visualization(file_path, cfg, detect_threshold=0.1, upchirp_threshold=0.1, 
                              save_dir='save_fig', max_packets=5):
    """
    带可视化的解码函数
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
    print("\n[1/3] 执行前导码同步检测...")
    syncs = rx.sync.detect(data)
    print(f"      检测到 {len(syncs)} 个潜在数据包")
    
    if not syncs:
        print("\n警告: 未检测到数据包，尝试降低阈值...")
        rx = LoRaReceiver(cfg, detect_threshold=0.01, upchirp_corr_threshold=0.01)
        syncs = rx.sync.detect(data)
        print(f"      重新检测到 {len(syncs)} 个潜在数据包")
    
    # 创建输出目录
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # 处理每个检测到的包
    frames = []
    ns = cfg.symbol_samples
    
    print("\n[2/3] 解码数据包并生成可视化...")
    
    for i, s in enumerate(syncs[:max_packets]):
        print(f"\n  数据包 #{i+1}:")
        print(f"    起始位置: {s.packet_start}")
        print(f"    SFD 位置: {s.sfd_pos}")
        print(f"    Header 位置: {s.header_start}")
        print(f"    同步分数: {s.score:.4f}")
        
        # 解码 header
        header_words = rx.demod.extract_words(data, start=s.header_start, 
                                               n_symbols=8, reduced_rate=True)
        if len(header_words) < 8:
            print(f"    错误: Header 符号不足 ({len(header_words)} < 8)")
            continue
        
        decoded_header = rx.bitpipe.decode_block(header_words, ppm=cfg.sf-2, 
                                                  is_header=True, coding_rate=cfg.coding_rate)
        header = rx._parse_header(decoded_header)
        
        if header is None:
            print(f"    错误: 无法解析 header")
            continue
        
        print(f"    Header 解码成功:")
        print(f"      - 载荷长度: {header.length} bytes")
        print(f"      - 编码率: 4/{4+header.coding_rate}")
        print(f"      - CRC: {'有' if header.has_crc else '无'}")
        
        # 计算 payload 位置和长度
        total_len = header.length + (2 if header.has_crc else 0)
        n_payload_symbols = rx._payload_symbol_count(total_len, header.coding_rate)
        payload_start = s.header_start + 8 * ns
        frame_end = payload_start + n_payload_symbols * ns
        
        # 生成可视化
        save_path = os.path.join(save_dir, f'{base_name}_pkt{i+1}_analysis.png')
        plot_packet_analysis(data, s.packet_start, s.sfd_pos, s.header_start,
                            payload_start, frame_end, cfg, save_path,
                            title=f"Packet {i+1}")
        
        # 解码 payload
        payload_words = rx.demod.extract_words(data, start=payload_start,
                                               n_symbols=n_payload_symbols,
                                               reduced_rate=cfg.reduced_rate)
        
        block_size = header.coding_rate + 4
        ppm = cfg.sf - (2 if cfg.reduced_rate else 0)
        decoded_payload = bytearray()
        
        for j in range(0, len(payload_words), block_size):
            block = payload_words[j:j+block_size]
            if len(block) < block_size:
                break
            decoded_payload.extend(rx.bitpipe.decode_block(block, ppm=ppm,
                                                            is_header=False,
                                                            coding_rate=header.coding_rate))
        
        decoded_payload = decoded_payload[:total_len]
        mac_crc = None
        payload = bytes(decoded_payload)
        
        if header.has_crc and len(payload) >= 2:
            mac_crc = (payload[-2] << 8) | payload[-1]
            payload = payload[:-2]
        
        from lora_decoder.types import LoRaFrame
        frame = LoRaFrame(start_index=s.packet_start, header=header, payload=payload,
                         mac_crc=mac_crc, snr_db=10.0*np.log10(max(s.score, 1e-12)))
        frames.append(frame)
        
        print(f"    Payload 解码完成: {len(payload)} bytes")
        if mac_crc is not None:
            print(f"      CRC: 0x{mac_crc:04x}")
        print(f"      Hex: {payload[:32].hex()}{'...' if len(payload) > 32 else ''}")
    
    # 打印总结
    print(f"\n[3/3] 解码完成!")
    print(f"      成功解码: {len(frames)} 个数据包")
    print(f"      可视化图表保存在: {save_dir}/")
    
    return frames


def main():
    parser = argparse.ArgumentParser(description="LoRa 解码可视化工具")
    parser.add_argument("--input", default=r"rawData\dong", help=".bin 文件或文件夹")
    parser.add_argument("--sf", type=int, default=11, help="扩频因子 (6-12)")
    parser.add_argument("--bw", type=float, default=125000, help="带宽 (Hz)")
    parser.add_argument("--fs", type=float, default=1000000, help="采样率 (Hz)")
    parser.add_argument("--preamble", type=int, default=16, help="前导码符号数")
    parser.add_argument("--cr", type=int, default=1, help="编码率 1..4 对应 4/5..4/8")
    parser.add_argument("--implicit", action="store_true", help="隐式 header 模式")
    parser.add_argument("--no-crc", action="store_true", help="无 CRC")
    parser.add_argument("--detect-threshold", type=float, default=0.1, help="检测阈值")
    parser.add_argument("--upchirp-threshold", type=float, default=0.1, help="upchirp 相关阈值")
    parser.add_argument("--output-dir", default="save_fig", help="输出图表目录")
    parser.add_argument("--max-packets", type=int, default=5, help="最多处理几个包")
    args = parser.parse_args()
    
    cfg = LoRaConfig(
        sf=args.sf,
        bw=args.bw,
        fs=args.fs,
        preamble_symbols=args.preamble,
        implicit_header=args.implicit,
        coding_rate=args.cr,
        has_crc=not args.no_crc,
    )
    
    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.bin")))
    else:
        files = [args.input]
    
    for path in files:
        frames = decode_with_visualization(
            path, cfg,
            detect_threshold=args.detect_threshold,
            upchirp_threshold=args.upchirp_threshold,
            save_dir=args.output_dir,
            max_packets=args.max_packets
        )
        
        print(f"\n{'='*60}")
        print(f"文件: {os.path.basename(path)}")
        print(f"解码包数: {len(frames)}")
        for i, frame in enumerate(frames, 1):
            crc_txt = f"0x{frame.mac_crc:04x}" if frame.mac_crc is not None else "None"
            print(f"  #{i}: start={frame.start_index}, len={len(frame.payload)}, "
                  f"cr=4/{4+frame.header.coding_rate}, crc={crc_txt}")


if __name__ == "__main__":
    main()
