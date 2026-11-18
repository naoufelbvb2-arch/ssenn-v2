"""Single-command verification of software invariants and all configured quality gates."""

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output", default="results.local.json")
    parser.add_argument("--flops-output")
    options = parser.parse_args()
    profile_args = ["--config", options.config] if options.config else []
    output_args = ["--output", options.flops_output] if options.flops_output else []
    unit = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    experiments = subprocess.run([sys.executable, "-m", "benchmarks.c01", *profile_args,
                                 "--output", options.output])
    flops = subprocess.run([sys.executable, "-m", "benchmarks.flops", *profile_args, *output_args])
    return int(bool(unit.returncode or experiments.returncode or flops.returncode))


if __name__ == "__main__":
    raise SystemExit(main())
