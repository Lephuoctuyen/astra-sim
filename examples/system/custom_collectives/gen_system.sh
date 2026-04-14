#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <system_layer_inputs.json> [output_dir]"
  exit 1
fi

MANIFEST_JSON="$1"
OUT_DIR="${2:-./generated_system_ets}"
SCRIPT_DIR=$(dirname "$(realpath "$0")")

mkdir -p "$OUT_DIR"

if [ ! -f "$MANIFEST_JSON" ]; then
  echo "ERROR: manifest JSON not found: $MANIFEST_JSON"
  exit 1
fi

if [ -z "${COLLECTIVEAPI_PATH:-}" ]; then
  echo "Enter the path to where the collectiveapi repo is cloned."
  read -r -p "Press Enter to use default path (\$HOME/collectiveapi), or type 'abort' or 'x' to exit: " COLLECTIVEAPI_PATH
fi

if [ -z "${COLLECTIVEAPI_PATH:-}" ]; then
  COLLECTIVEAPI_PATH="$HOME/collectiveapi"
fi

if [ "$COLLECTIVEAPI_PATH" = "abort" ] || [ "$COLLECTIVEAPI_PATH" = "x" ]; then
  echo "Aborted by user."
  exit 1
fi

echo "COLLECTIVEAPI_PATH: $COLLECTIVEAPI_PATH"

# Optional template directories for fallback mode
ALLREDUCE_TEMPLATE_DIR="${ALLREDUCE_TEMPLATE_DIR:-$SCRIPT_DIR/custom_ring_allreduce_8npus_1MB}"
ALLTOALL_TEMPLATE_DIR="${ALLTOALL_TEMPLATE_DIR:-}"

# Probe converter
CONVERTER=""
if [ -f "$COLLECTIVEAPI_PATH/chakra_converter/et_converter.py" ]; then
  CONVERTER="$COLLECTIVEAPI_PATH/chakra_converter/et_converter.py"
fi

# Probe allreduce MSCCLang generator
if [ -z "${MSCCLANG_ALLREDUCE_SCRIPT:-}" ]; then
  for CANDIDATE in \
    "$COLLECTIVEAPI_PATH/msccl-tools/examples/mscclang/allreduce_a100_ring.py" \
    "$COLLECTIVEAPI_PATH/msccl-tools/examples/mscclang/allreduce_ring.py" \
    "$COLLECTIVEAPI_PATH/msccl-tools/examples/mscclang/allreduce.py" \
    "$COLLECTIVEAPI_PATH/examples/mscclang/allreduce_a100_ring.py" \
    "$COLLECTIVEAPI_PATH/examples/mscclang/allreduce_ring.py" \
    "$COLLECTIVEAPI_PATH/examples/mscclang/allreduce.py"
  do
    if [ -f "$CANDIDATE" ]; then
      MSCCLANG_ALLREDUCE_SCRIPT="$CANDIDATE"
      break
    fi
  done
fi

# Probe alltoall MSCCLang generator
if [ -z "${MSCCLANG_ALLTOALL_SCRIPT:-}" ]; then
  for CANDIDATE in \
    "$COLLECTIVEAPI_PATH/msccl-tools/examples/mscclang/alltoall_a100_pairwise.py" \
    "$COLLECTIVEAPI_PATH/msccl-tools/examples/mscclang/alltoall_pairwise.py" \
    "$COLLECTIVEAPI_PATH/msccl-tools/examples/mscclang/alltoall.py" \
    "$COLLECTIVEAPI_PATH/examples/mscclang/alltoall_a100_pairwise.py" \
    "$COLLECTIVEAPI_PATH/examples/mscclang/alltoall_pairwise.py" \
    "$COLLECTIVEAPI_PATH/examples/mscclang/alltoall.py"
  do
    if [ -f "$CANDIDATE" ]; then
      MSCCLANG_ALLTOALL_SCRIPT="$CANDIDATE"
      break
    fi
  done
fi

echo "Detected resources:"
echo "  CONVERTER                 = ${CONVERTER:-<not found>}"
echo "  MSCCLANG_ALLREDUCE_SCRIPT = ${MSCCLANG_ALLREDUCE_SCRIPT:-<not found>}"
echo "  MSCCLANG_ALLTOALL_SCRIPT  = ${MSCCLANG_ALLTOALL_SCRIPT:-<not found>}"
echo "  ALLREDUCE_TEMPLATE_DIR    = ${ALLREDUCE_TEMPLATE_DIR:-<not set>}"
echo "  ALLTOALL_TEMPLATE_DIR     = ${ALLTOALL_TEMPLATE_DIR:-<not set>}"

if ! command -v chakra_jsonizer >/dev/null 2>&1; then
  echo "WARNING: chakra_jsonizer not found; JSON conversion will be skipped."
  HAVE_JSONIZER=0
else
  HAVE_JSONIZER=1
fi

python3 - <<'PY' "$MANIFEST_JSON" "$OUT_DIR" "$CONVERTER" "${MSCCLANG_ALLREDUCE_SCRIPT:-}" "${MSCCLANG_ALLTOALL_SCRIPT:-}" "${ALLREDUCE_TEMPLATE_DIR:-}" "${ALLTOALL_TEMPLATE_DIR:-}"
import json
import shutil
import subprocess
import sys
from pathlib import Path

manifest_json = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
converter = sys.argv[3]
allreduce_script = sys.argv[4]
alltoall_script = sys.argv[5]
allreduce_template_dir = sys.argv[6]
alltoall_template_dir = sys.argv[7]

with open(manifest_json, "r", encoding="utf-8") as f:
    manifest = json.load(f)

num_npus = int(manifest["num_npus"])
artifacts = manifest["artifacts"]

out_dir.mkdir(parents=True, exist_ok=True)

def run(cmd, stdout_path=None):
    print("[CMD]", " ".join(str(x) for x in cmd))
    if stdout_path is None:
        subprocess.run(cmd, check=True)
    else:
        with open(stdout_path, "w", encoding="utf-8") as fout:
            subprocess.run(cmd, check=True, stdout=fout)

def copy_template(template_dir: Path, algorithm_key: str, nranks: int):
    if not template_dir.is_dir():
        raise RuntimeError(f"Template directory not found: {template_dir}")

    missing = []
    for i in range(nranks):
        src = template_dir / f"custom_allreduce.{i}.et"
        if not src.exists():
            alt1 = template_dir / f"custom_alltoall.{i}.et"
            alt2 = template_dir / f"{algorithm_key}.{i}.et"
            if alt1.exists():
                src = alt1
            elif alt2.exists():
                src = alt2
            else:
                missing.append(str(src))
                continue

        dst = out_dir / f"{algorithm_key}.{i}.et"
        shutil.copyfile(src, dst)
        print(f"[COPY] {src} -> {dst}")

    if missing:
        raise RuntimeError(
            "Template ET files missing. Checked:\n" + "\n".join(missing)
        )

for art in artifacts:
    algorithm_key = art["algorithm_key"]
    comm_type = art["comm_type"]
    comm_size = int(art["comm_size"])
    nranks = int(art["num_npus"])

    xml_path = out_dir / f"{algorithm_key}.xml"
    et_prefix = out_dir / algorithm_key

    if comm_type == "ALLREDUCE":
        if allreduce_script and converter:
            run(
                ["python3", allreduce_script, str(nranks), "1", "1"],
                stdout_path=xml_path,
            )
            run([
                "python3", converter,
                "--input_filename", str(xml_path),
                "--output_filename", str(et_prefix),
                "--coll_size", str(comm_size),
                "--collective", "allreduce",
            ])
        else:
            print(
                f"[WARN] No MSCCLang allreduce generator found. "
                f"Falling back to template ETs for {algorithm_key}."
            )
            copy_template(Path(allreduce_template_dir), algorithm_key, nranks)

    elif comm_type == "ALLTOALL":
        if alltoall_script and converter:
            run(
                ["python3", alltoall_script, str(nranks), "1", "1"],
                stdout_path=xml_path,
            )
            run([
                "python3", converter,
                "--input_filename", str(xml_path),
                "--output_filename", str(et_prefix),
                "--coll_size", str(comm_size),
                "--collective", "alltoall",
            ])
        else:
            if not alltoall_template_dir:
                raise RuntimeError(
                    f"No ALLTOALL generator/template available for {algorithm_key}.\n"
                    f"Set MSCCLANG_ALLTOALL_SCRIPT if your repo has one, or set "
                    f"ALLTOALL_TEMPLATE_DIR to a directory containing template ETs."
                )
            print(
                f"[WARN] No MSCCLang alltoall generator found. "
                f"Falling back to template ETs for {algorithm_key}."
            )
            copy_template(Path(alltoall_template_dir), algorithm_key, nranks)

    else:
        raise RuntimeError(f"Unsupported comm_type: {comm_type}")

print("[OK] generated all ET artifacts")
PY

if [ "$HAVE_JSONIZER" -eq 1 ]; then
  echo "[INFO] Converting generated ET files to JSON..."
  python3 - <<'PY' "$MANIFEST_JSON" "$OUT_DIR"
import json
import subprocess
import sys
from pathlib import Path

manifest_json = Path(sys.argv[1])
out_dir = Path(sys.argv[2])

with open(manifest_json, "r", encoding="utf-8") as f:
    manifest = json.load(f)

for art in manifest["artifacts"]:
    key = art["algorithm_key"]
    nranks = int(art["num_npus"])
    for i in range(nranks):
        et_file = out_dir / f"{key}.{i}.et"
        json_file = out_dir / f"{key}.{i}.json"
        if et_file.exists():
            subprocess.run([
                "chakra_jsonizer",
                "--input_filename", str(et_file),
                "--output_filename", str(json_file),
            ], check=True)

print("[OK] converted ET files to JSON")
PY
fi

echo "[DONE] Output directory: $OUT_DIR"