#!/usr/bin/env python3

from pathlib import Path
import hashlib
import sys

try:
    import kagglehub
except ImportError:
    sys.exit(
        "ERROR: kagglehub is not installed.\n"
        "Install it with:\n"
        "    python3 -m pip install kagglehub"
    )

DATASET = "mhadani/predictive-maintenance-aircraft-engine"

FILES = (
    "PM_train.csv",
    "PM_test.csv",
    "PM_truth.csv",
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "02_raw_data"
CHECKSUM_FILE = ROOT / "DATA_SHA256.txt"

OUTPUT.mkdir(parents=True, exist_ok=True)

print(f"Kaggle dataset: {DATASET}")
print(f"Destination   : {OUTPUT}")
print()

for filename in FILES:
    print(f"Downloading {filename} ...")
    path = kagglehub.dataset_download(
        DATASET,
        path=filename,
        output_dir=str(OUTPUT),
    )
    print(f"  {path}")

print()

# Read expected hashes generated from the exact files used in the paper.
expected = {}

if CHECKSUM_FILE.exists():
    for line in CHECKSUM_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue

        digest, filename = line.split(maxsplit=1)
        expected[Path(filename).name] = digest


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


print("Data integrity check")
print("--------------------")

failed = False

for filename in FILES:
    path = OUTPUT / filename

    if not path.exists():
        print(f"MISSING  {filename}")
        failed = True
        continue

    digest = sha256(path)
    reference = expected.get(filename)

    if reference is None:
        print(f"NO REF   {filename}  {digest}")
    elif digest == reference:
        print(f"OK       {filename}  {digest}")
    else:
        print(f"MISMATCH {filename}")
        print(f"  expected: {reference}")
        print(f"  actual  : {digest}")
        failed = True

if failed:
    sys.exit(
        "\nERROR: downloaded data do not exactly match the files "
        "used for the published analysis."
    )

print("\nAll dataset files match the reference SHA-256 values.")
