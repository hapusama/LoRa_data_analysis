import argparse
import glob
import os

from lora_decoder import LoRaConfig, LoRaReceiver


def main() -> None:
    parser = argparse.ArgumentParser(description="Python LoRa decoder pipeline (gr-lora style)")
    parser.add_argument("--input", default=r"d:\\Desktop\\data_analysis\\rawData\\dong", help=".bin file or folder")
    parser.add_argument("--sf", type=int, default=10)
    parser.add_argument("--bw", type=float, default=125000)
    parser.add_argument("--fs", type=float, default=1000000)
    parser.add_argument("--preamble", type=int, default=16)
    parser.add_argument("--cr", type=int, default=1, help="1..4 for 4/5..4/8")
    parser.add_argument("--implicit", action="store_true")
    parser.add_argument("--no-crc", action="store_true")
    parser.add_argument("--detect-threshold", type=float, default=0.1)
    parser.add_argument("--upchirp-threshold", type=float, default=0.10)
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
    rx = LoRaReceiver(
        cfg,
        detect_threshold=args.detect_threshold,
        upchirp_corr_threshold=args.upchirp_threshold,
    )

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.bin")))
    else:
        files = [args.input]

    for path in files:
        frames = rx.decode_file(path)
        print(f"[{os.path.basename(path)}] frames={len(frames)}")
        for i, frame in enumerate(frames, 1):
            crc_txt = f"0x{frame.mac_crc:04x}" if frame.mac_crc is not None else "None"
            print(
                f"  #{i} start={frame.start_index} len={len(frame.payload)} "
                f"cr=4/{4 + frame.header.coding_rate} crc={crc_txt} "
                f"payload_hex={frame.payload.hex()}"
            )


if __name__ == "__main__":
    main()
