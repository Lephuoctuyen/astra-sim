#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_int(x: str) -> int:
    return int(x.strip())


def parse_spec_file(spec_path: Path):
    """
    Expected format (based on your DLRM workload generator context):

    HYBRID_DLRM <num_groups> <num_npus>
    <layer_name> <fwd_compute> <fwd_comm_type> <fwd_comm_size> <inp_grad_compute> <inp_grad_comm_type> <inp_grad_comm_size> <weight_grad_compute> <weight_grad_comm_type> <weight_grad_comm_size>
    ...

    Example line:
    Embedding 1174 ALLTOALL 98304 0 ALLTOALL 98304 0 NONE 0
    """
    lines = []
    with open(spec_path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)

    if not lines:
        raise ValueError(f"SPEC file is empty: {spec_path}")

    header = lines[0].split()
    if len(header) < 3:
        raise ValueError(
            "First line must be: HYBRID_DLRM <num_groups> <num_npus>"
        )

    workload_name = header[0]
    num_groups = parse_int(header[1])
    num_npus = parse_int(header[2])

    layers = []
    for lineno, line in enumerate(lines[1:], start=2):
        toks = line.split()
        if len(toks) != 10:
            raise ValueError(
                f"Line {lineno}: expected 10 tokens, got {len(toks)}\n"
                f"Line: {line}"
            )

        layer = {
            "name": toks[0],
            "fwd_compute": parse_int(toks[1]),
            "fwd_comm_type": toks[2],
            "fwd_comm_size": parse_int(toks[3]),
            "bwd_compute": parse_int(toks[4]),
            "bwd_comm_type": toks[5],
            "bwd_comm_size": parse_int(toks[6]),
            "wgrad_compute": parse_int(toks[7]),
            "wgrad_comm_type": toks[8],
            "wgrad_comm_size": parse_int(toks[9]),
        }
        layers.append(layer)

    return workload_name, num_groups, num_npus, layers


def choose_algorithm(comm_type: str) -> str:
    if comm_type == "ALLREDUCE":
        return "ring"
    if comm_type == "ALLTOALL":
        return "pairwise"
    raise ValueError(f"Unsupported comm type: {comm_type}")


def build_algorithm_key(num_npus: int, comm_type: str, comm_size: int) -> str:
    algo = choose_algorithm(comm_type)
    if comm_type == "ALLREDUCE":
        return f"allreduce_{algo}_{num_npus}ranks_{comm_size}"
    if comm_type == "ALLTOALL":
        return f"alltoall_{algo}_{num_npus}ranks_{comm_size}"
    raise ValueError(f"Unsupported comm type: {comm_type}")


def extract_collectives(workload_name, num_groups, num_npus, layers):
    collectives = []

    # Forward pass: in-order
    for layer in layers:
        if layer["fwd_comm_type"] != "NONE":
            collectives.append({
                "node_name": f'{layer["name"]}_fwd_comm',
                "layer": layer["name"],
                "phase": "fwd",
                "comm_type": layer["fwd_comm_type"],
                "comm_size": layer["fwd_comm_size"],
                "algorithm": choose_algorithm(layer["fwd_comm_type"]),
                "algorithm_key": build_algorithm_key(
                    num_npus, layer["fwd_comm_type"], layer["fwd_comm_size"]
                ),
            })

    # Backward + wgrad: reverse order, to match workload generation order
    for layer in reversed(layers):
        if layer["bwd_comm_type"] != "NONE":
            collectives.append({
                "node_name": f'{layer["name"]}_bwd_comm',
                "layer": layer["name"],
                "phase": "bwd",
                "comm_type": layer["bwd_comm_type"],
                "comm_size": layer["bwd_comm_size"],
                "algorithm": choose_algorithm(layer["bwd_comm_type"]),
                "algorithm_key": build_algorithm_key(
                    num_npus, layer["bwd_comm_type"], layer["bwd_comm_size"]
                ),
            })

        if layer["wgrad_comm_type"] != "NONE":
            collectives.append({
                "node_name": f'{layer["name"]}_wgrad_comm',
                "layer": layer["name"],
                "phase": "wgrad",
                "comm_type": layer["wgrad_comm_type"],
                "comm_size": layer["wgrad_comm_size"],
                "algorithm": choose_algorithm(layer["wgrad_comm_type"]),
                "algorithm_key": build_algorithm_key(
                    num_npus, layer["wgrad_comm_type"], layer["wgrad_comm_size"]
                ),
            })

    # Deduplicate algorithm artifacts
    artifacts = {}
    for c in collectives:
        key = c["algorithm_key"]
        if key not in artifacts:
            artifacts[key] = {
                "algorithm_key": key,
                "comm_type": c["comm_type"],
                "comm_size": c["comm_size"],
                "algorithm": c["algorithm"],
                "num_npus": num_npus,
                "users": [],
            }
        artifacts[key]["users"].append(c["node_name"])

    return {
        "workload_name": workload_name,
        "num_groups": num_groups,
        "num_npus": num_npus,
        "algorithm_policy": {
            "ALLREDUCE": "ring",
            "ALLTOALL": "pairwise",
        },
        "collectives": collectives,
        "artifacts": list(artifacts.values()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate exact System Layer inputs from a DLRM SPEC file."
    )
    parser.add_argument(
        "--spec",
        required=True,
        help="Path to DLRM SPEC file",
    )
    parser.add_argument(
        "--output_json",
        default="system_layer_inputs.json",
        help="Output JSON manifest",
    )
    args = parser.parse_args()

    spec_path = Path(args.spec)
    workload_name, num_groups, num_npus, layers = parse_spec_file(spec_path)
    manifest = extract_collectives(workload_name, num_groups, num_npus, layers)

    out_path = Path(args.output_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] wrote {out_path}")
    print(f"[INFO] workload_name = {workload_name}")
    print(f"[INFO] num_groups    = {num_groups}")
    print(f"[INFO] num_npus      = {num_npus}")
    print(f"[INFO] collectives   = {len(manifest['collectives'])}")
    print(f"[INFO] unique ETs    = {len(manifest['artifacts'])}")


if __name__ == "__main__":
    main()