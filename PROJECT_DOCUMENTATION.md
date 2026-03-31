# LoRa 信号分析与解码项目文档

> 项目路径: `d:\Desktop\data_analysis`
> 生成日期: 2026-03-29

---

## 项目概述

这是一个用于 **LoRa（Long Range）无线信号** 分析、诊断和解码的 Python 工具集。项目支持从原始 IQ 数据文件中检测、解调和解码 LoRa 数据包，并提供丰富的可视化功能。

### 核心能力
- 信号诊断: 分析原始 LoRa 信号的特征（频谱、前导码检测、参数匹配）
- 包检测: 自动检测数据包起始位置，支持前导码同步和 SFD 定位
- 解码: 完整的 LoRa 解码流程（解调、解交织、解扰码、汉明解码）
- 可视化: 生成时频图、相位分析图、FFT 对比图等

---

## 文件结构总览

```
data_analysis/
├── diagnose_signal.py          # 信号诊断脚本 v1
├── diagnose_signal2.py         # 信号诊断脚本 v2（增强版）
├── utils/
│   ├── bin_read.py             # 基础包检测和可视化工具
│   ├── find_correct_sf.py      # SF（扩频因子）自动检测工具
│   ├── lora_decode_bin.py      # LoRa 解码器命令行入口
│   ├── lora_decode_visual.py   # 带完整可视化的解码器
│   ├── lora_decode_visual_simple.py  # 简化版可视化解码器
│   ├── test_sampling_rate.py   # 采样率测试工具
│   └── lora_decoder/           # 核心解码库（Python包）
│       ├── __init__.py
│       ├── config.py           # LoRa 配置类
│       ├── types.py            # 数据类型定义
│       ├── chirp.py            # Chirp 信号生成
│       ├── sync.py             # 前导码同步检测
│       ├── demod.py            # 符号解调
│       ├── coding.py           # 编码/解码（交织、扰码、汉明码）
│       └── receiver.py         # 高层接收器
├── rawData/dong/               # 原始数据文件目录
│   └── 1_0_9_11_14_16.bin      # 示例 IQ 数据文件
├── save_fig/                   # 生成的图表保存目录
├── .vscode/launch.json         # VS Code 调试配置
└── .gitignore                  # Git 忽略配置
```

---

## 重要文件详解

### 1. 顶层诊断脚本

#### `diagnose_signal.py` - 基础信号诊断
| 属性 | 说明 |
|------|------|
| **功能** | 分析 LoRa 信号的基本特征 |
| **主要能力** | 读取 IQ 数据文件；生成整体频谱图和时域图；使用 upchirp 相关检测前导码；分析单个 chirp 的去啁啾效果；检测 SFD 位置；测试不同采样率假设 |
| **输出** | `save_fig/overall_spectrogram.png`, `save_fig/preamble_analysis.png`, `save_fig/single_chirp_analysis.png` |
| **使用方法** | `python diagnose_signal.py` |

#### `diagnose_signal2.py` - 深度信号诊断
| 属性 | 说明 |
|------|------|
| **功能** | 更全面的信号参数分析和优化 |
| **主要能力** | 测试多种 SF/BW/fs 配置组合；自动寻找最佳配置；频率偏移估计和校正；实际带宽分析；生成详细分析图 |
| **输出** | `save_fig/detailed_analysis.png` |
| **使用方法** | `python diagnose_signal2.py` |

---

### 2. 工具脚本 (`utils/`)

#### `bin_read.py` - 包检测与可视化基础工具
| 属性 | 说明 |
|------|------|
| **功能** | 检测数据包并生成前导码可视化 |
| **核心函数** | `detect_packets()`: 使用前导码+SFD检测包位置；`dechirp_symbol()`: 符号去啁啾；`process_bin_file()`: 处理单个bin文件 |
| **输出** | 前导码时频图、相位连续图 |
| **可配置参数** | CF(中心频率)、BW(带宽)、SF(扩频因子)、fs(采样率)、preamble_symbols(前导码符号数) |

#### `find_correct_sf.py` - SF 自动检测工具
| 属性 | 说明 |
|------|------|
| **功能** | 自动检测正确的扩频因子 (SF) |
| **工作原理** | 测试 SF7-SF12，通过 dechirp 后的 FFT 峰值锐度和集中度评分 |
| **评分指标** | Sharpness（锐度）: 峰值与周围平均的比值；Concentration（集中度）: 峰值能量占比 |
| **输出** | 最佳 SF 推荐；SF 对比表；`save_fig/sf_comparison_*.png` |
| **命令行** | `python find_correct_sf.py --input <file> --bw 125000 --fs 1000000` |

#### `lora_decode_bin.py` - 解码器命令行入口
| 属性 | 说明 |
|------|------|
| **功能** | 纯解码功能，无可视化 |
| **参数** | `--sf`, `--bw`, `--fs`, `--preamble`, `--cr`, `--implicit`, `--no-crc`, `--detect-threshold` |
| **输出** | 解码后的数据包信息（hex、CRC、长度等） |
| **使用示例** | `python lora_decode_bin.py --input rawData/dong --sf 10 --bw 125000` |

#### `lora_decode_visual.py` - 完整可视化解码器
| 属性 | 说明 |
|------|------|
| **功能** | 解码 + 完整可视化分析 |
| **可视化内容** | 整体时频图（标记Preamble/SFD/Header/Payload边界）；Preamble dechirped FFT；SFD dechirped FFT；Preamble 相位连续性；Header/Payload 符号峰值 |
| **输出** | `save_fig/<filename>_pkt<N>_analysis.png` |
| **使用示例** | `python lora_decode_visual.py --input rawData/dong --sf 11` |

#### `lora_decode_visual_simple.py` - 简化可视化解码器
| 属性 | 说明 |
|------|------|
| **功能** | 只显示前导码和 SFD 的简化分析 |
| **可视化内容** | 前导码+SFD时频图；Preamble vs SFD dechirped FFT 对比 |
| **特点** | 界面简洁，便于快速检查对齐情况 |
| **输出** | `save_fig/<filename>_pkt<N>_preamble.png` |

#### `test_sampling_rate.py` - 采样率测试工具
| 属性 | 说明 |
|------|------|
| **功能** | 测试不同采样率设置，找到能正确显示16个前导码chirp的配置 |
| **测试方法** | 通过能量峰值检测估计可见chirp数量 |
| **输出** | 最佳采样率推荐；`save_fig/preamble_test_*.png` |

---

### 3. 核心解码库 (`utils/lora_decoder/`)

这是一个完整的 LoRa 解码 Python 包，设计思路参考 gr-lora。

#### `config.py` - 配置类
```python
LoRaConfig(
    sf=9,              # 扩频因子 (6-12)
    bw=125000,         # 带宽 (Hz)
    fs=1000000,        # 采样率 (Hz)
    preamble_symbols=16,  # 前导码符号数
    implicit_header=False, # 是否隐式头
    coding_rate=1,     # 编码率 1-4 对应 4/5-4/8
    has_crc=True,      # 是否有CRC
)
```
**属性**: `num_bins`, `num_bins_header`, `symbol_samples`

#### `types.py` - 数据类型
| 类 | 说明 |
|----|------|
| `SyncResult` | 同步结果（packet_start, sfd_pos, header_start, score） |
| `LoRaHeader` | 解析后的头信息（length, coding_rate, has_crc, raw） |
| `LoRaFrame` | 完整数据帧（payload, mac_crc, snr_db等） |

#### `chirp.py` - Chirp 信号
| 函数 | 功能 |
|------|------|
| `upchirp(cfg, n_samples)` | 生成上啁啾信号 |
| `downchirp(cfg, n_samples)` | 生成下啁啾信号 |
| `dechirp(symbol, cfg)` | 对符号进行去啁啾 |

#### `sync.py` - 前导码同步 (`PreambleSynchronizer`)
**算法流程**:
1. **粗检测**: 相邻 chirp 归一化相关找候选位置
2. **精细化**: upchirp 互相关定位精确起点
3. **SFD 检测**: 在预期位置用 downchirp 搜索

**可调参数**:
- `detect_threshold`: 粗检测阈值 (默认 0.90)
- `upchirp_corr_threshold`: upchirp 相关阈值 (默认 0.30)
- `search_margin_ratio`: SFD 搜索范围比例

#### `demod.py` - 符号解调 (`SymbolDemodulator`)
**功能**: 
- 使用 downchirp 去啁啾
- FFT 转换到频域
- 峰值检测得到 symbol value
- 支持 reduced rate (Header 使用 SF-2)
- Gray 解码 (word = bin_idx ^ (bin_idx >> 1))

#### `coding.py` - 编码解码 (`BitPipelineDecoder`)
**解码流程**:
1. **Deinterleave** (解交织): 对角线读取矩阵
2. **Deshuffle** (解扰码): 按固定 pattern 重排比特
3. **Dewhiten** (解白化): 使用 PN9 序列 XOR
4. **Hamming Decode** (汉明解码): 纠错并提取数据位

#### `receiver.py` - 高层接收器 (`LoRaReceiver`)
**主入口**:
- `decode(data)`: 对 numpy 数组解码
- `decode_file(file_path)`: 从 .bin 文件解码

**完整解码流程**:
```
IQ Data → Sync → Demod Header → Decode Header → Demod Payload → Decode Payload → Extract CRC → LoRaFrame
```

---

## 典型使用场景

### 场景1: 分析未知参数的信号
```bash
# 1. 先用 diagnose_signal2.py 找出最佳配置
python diagnose_signal2.py

# 2. 用 find_correct_sf.py 确认 SF
python utils/find_correct_sf.py --input rawData/dong/1_0_9_11_14_16.bin

# 3. 用可视化解码器观察并解码
python utils/lora_decode_visual_simple.py --input rawData/dong --sf 9 --bw 125000
```

### 场景2: 批量解码已知参数的信号
```bash
# 直接解码整个目录
python utils/lora_decode_bin.py --input rawData/dong --sf 9 --bw 125000 --fs 1000000 --preamble 16 --cr 1
```

### 场景3: 检查前导码对齐
```bash
python utils/lora_decode_visual_simple.py --input rawData/dong/1_0_9_11_14_16.bin --sf 9 --detect-threshold 0.05 --max-packets 3
```

---

## 数据文件格式

### 输入文件
- **格式**: 复数 IQ 数据，float32 存储 (interleaved I/Q)
- **扩展名**: `.bin`
- **采样类型**: `np.complex64`
- **命名示例**: `1_0_9_11_14_16.bin` (可能编码了参数信息)

### 输出图表
- **位置**: `save_fig/` 目录
- **格式**: PNG (150 DPI)
- **命名**: `<原始文件名>_pkt<序号>_<类型>.png`

---

## 可调参数说明

| 参数 | 说明 | 典型值 |
|------|------|--------|
| SF | 扩频因子 | 7-12 (常用 9-11) |
| BW | 带宽 | 125kHz, 250kHz |
| fs | 采样率 | 500kHz, 1MHz, 2MHz |
| Preamble | 前导码符号数 | 8, 16 |
| CR | 编码率 | 1(4/5), 2(4/6), 3(4/7), 4(4/8) |

---

## 调试技巧

1. **信号检测不到**: 降低 `--detect-threshold` 和 `--upchirp-threshold`
2. **Header 解析失败**: 检查 SF、BW、fs 参数是否匹配
3. **CRC 错误**: 尝试不同的 CR 值，检查是否有频率偏移
4. **可视化对齐检查**: 看 preamble FFT 峰值是否在 bin 0 附近

---

## 依赖环境

```
numpy
matplotlib
scipy
```

---

## 已知限制

- 仅支持标准 LoRa 调制（单天线、非 MIMO）
- 同步检测依赖前导码质量
- 对频率偏移敏感（>1kHz 可能影响解码）
- 当前仅支持 implicit_header=False 模式

---

*文档生成时间: 2026-03-29*
*项目状态: 开发中，核心功能已可用*
