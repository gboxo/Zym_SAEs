# Wrapper for Noelia FT-DMS activity prediction
# Deprecated: please use src.tools.oracles.activity_prediction with --mode kl
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(
        description="Wrapper for unified batch activity prediction"
    )
    parser.add_argument(
        "--cfg_path", type=str, required=True,
        help="Path to your YAML/JSON config"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for DataLoader"
    )
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "src.tools.oracles.activity_prediction",
        "--cfg_path", args.cfg_path,
        "--batch_size", str(args.batch_size)
    ]
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()


