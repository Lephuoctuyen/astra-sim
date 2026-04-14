#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


# =============================================================================
# Path setup
# Expected location:
#   ~/astra-sim/examples/workload/generate_dlrm_et.py
# =============================================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ASTRA_SIM_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

CHAKRA_ROOT = os.path.join(ASTRA_SIM_ROOT, "extern", "graph_frontend", "chakra")
PROTO_DIR = os.path.join(CHAKRA_ROOT, "schema", "protobuf")
UTIL_DIR = os.path.join(CHAKRA_ROOT, "src", "third_party", "utils")

for p in [PROTO_DIR, UTIL_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from et_def_pb2 import (  # type: ignore
    ALL_REDUCE,
    ALL_TO_ALL,
    COMM_COLL_NODE,
    COMP_NODE,
    GlobalMetadata,
    Node,
    AttributeProto,
)
from protolib import encodeMessage  # type: ignore


# =============================================================================
# Workload spec
# Header:
#   HYBRID_DLRM <num_groups> <num_npus> <num_embedding_tables>
#
# Layer row:
#   <name> <group_id>
#   <fwd_us> <fwd_comm_type> <fwd_comm_size>
#   <bwd_us> <bwd_comm_type> <bwd_comm_size>
#   <wgrad_us> <wgrad_comm_type> <wgrad_comm_size>
#   <update_us>
#
# group_id = -1 means use the default DP group of the current rank
# =============================================================================
SPEC = """
HYBRID_DLRM 4 8 8

MLP_Bottom_0   -1  1280  NONE       0        794   NONE       0        896   ALLREDUCE  13312    13
MLP_Bottom_1   -1  2560  NONE       0       2560   NONE       0       1792   ALLREDUCE 524288   524
MLP_Bottom_2   -1  2560  NONE       0       2560   NONE       0       1792   ALLREDUCE 524288   524
MLP_Bottom_3   -1   800  NONE       0       1280   NONE       0        800   ALLREDUCE  16384    16
MLP_Top_0      -1  6400  NONE       0       6400   NONE       0       4480   ALLREDUCE 1064960 1065
MLP_Top_1      -1  2560  NONE       0       2560   NONE       0       1792   ALLREDUCE 524288   524
MLP_Top_2      -1  2560  NONE       0       2560   NONE       0       1792   ALLREDUCE 524288   524
""".strip()


COMM_TYPE_MAP = {
    "NONE": None,
    "ALLREDUCE": ALL_REDUCE,
    "ALLTOALL": ALL_TO_ALL,
}


# =============================================================================
# Data classes
# =============================================================================
@dataclass
class LayerSpec:
    name: str
    group_id: int
    fwd_compute: int
    fwd_comm_type: str
    fwd_comm_size: int
    bwd_compute: int
    bwd_comm_type: str
    bwd_comm_size: int
    wgrad_compute: int
    wgrad_comm_type: str
    wgrad_comm_size: int
    update_compute: int


@dataclass
class WorkloadSpec:
    workload_name: str
    num_groups: int
    num_npus: int
    num_embedding_tables: int
    layers: List[LayerSpec] = field(default_factory=list)


# =============================================================================
# Attribute helpers for your schema
# =============================================================================
def _attr_int(name: str, value: int, doc: str = "") -> AttributeProto:
    return AttributeProto(name=name, int64_val=int(value), doc_string=doc)


def _attr_str(name: str, value: str, doc: str = "") -> AttributeProto:
    return AttributeProto(name=name, string_val=str(value), doc_string=doc)


def _attr_bool(name: str, value: bool, doc: str = "") -> AttributeProto:
    return AttributeProto(name=name, bool_val=bool(value), doc_string=doc)


def _attr_ints(name: str, values: Sequence[int], doc: str = "") -> AttributeProto:
    vals = [int(v) for v in values]
    attr = AttributeProto(name=name, doc_string=doc)

    # In your schema, int64_list is a message wrapper, not a raw Python list field.
    if not hasattr(attr, "int64_list"):
        raise ValueError("AttributeProto has no int64_list field in this schema")

    sub = attr.int64_list
    repeated_field_name = None
    for f in sub.DESCRIPTOR.fields:
        if f.label == f.LABEL_REPEATED:
            repeated_field_name = f.name
            break

    if repeated_field_name is None:
        raise ValueError("int64_list exists but contains no repeated field")

    getattr(sub, repeated_field_name).extend(vals)
    return attr


# =============================================================================
# Parsing / validation
# =============================================================================
def strip_comments_and_blank(spec_text: str) -> List[str]:
    out = []
    for raw in spec_text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def parse_spec(spec_text: str) -> WorkloadSpec:
    lines = strip_comments_and_blank(spec_text)
    if not lines:
        raise ValueError("SPEC is empty")

    header = lines[0].split()
    if len(header) != 4:
        raise ValueError(
            "Header must be: HYBRID_DLRM <num_groups> <num_npus> <num_embedding_tables>"
        )

    workload_name = header[0]
    num_groups = int(header[1])
    num_npus = int(header[2])
    num_embedding_tables = int(header[3])

    layers: List[LayerSpec] = []
    for line in lines[1:]:
        toks = line.split()
        if len(toks) != 12:
            raise ValueError(f"Bad layer row ({len(toks)} tokens): {line}")

        layers.append(
            LayerSpec(
                name=toks[0],
                group_id=int(toks[1]),
                fwd_compute=int(toks[2]),
                fwd_comm_type=toks[3],
                fwd_comm_size=int(toks[4]),
                bwd_compute=int(toks[5]),
                bwd_comm_type=toks[6],
                bwd_comm_size=int(toks[7]),
                wgrad_compute=int(toks[8]),
                wgrad_comm_type=toks[9],
                wgrad_comm_size=int(toks[10]),
                update_compute=int(toks[11]),
            )
        )

    spec = WorkloadSpec(
        workload_name=workload_name,
        num_groups=num_groups,
        num_npus=num_npus,
        num_embedding_tables=num_embedding_tables,
        layers=layers,
    )
    validate_spec(spec)
    return spec


def validate_spec(spec: WorkloadSpec) -> None:
    if spec.workload_name != "HYBRID_DLRM":
        raise ValueError("Only HYBRID_DLRM is supported")
    if spec.num_groups <= 0:
        raise ValueError("num_groups must be > 0")
    if spec.num_npus <= 0:
        raise ValueError("num_npus must be > 0")
    if spec.num_embedding_tables <= 0:
        raise ValueError("num_embedding_tables must be > 0")
    if spec.num_groups > spec.num_npus:
        raise ValueError("num_groups cannot exceed num_npus")
    if spec.num_npus % spec.num_groups != 0:
        raise ValueError("num_npus must be divisible by num_groups")
    if not spec.layers:
        raise ValueError("Need at least one MLP layer")

    seen: Set[str] = set()
    for layer in spec.layers:
        if layer.name in seen:
            raise ValueError(f"Duplicate layer name: {layer.name}")
        seen.add(layer.name)

        if layer.group_id < -1:
            raise ValueError(f"Layer {layer.name}: group_id must be >= -1")
        if layer.group_id >= spec.num_groups:
            raise ValueError(
                f"Layer {layer.name}: group_id={layer.group_id} >= num_groups={spec.num_groups}"
            )

        for comm_name in [layer.fwd_comm_type, layer.bwd_comm_type, layer.wgrad_comm_type]:
            if comm_name not in COMM_TYPE_MAP:
                raise ValueError(f"Layer {layer.name}: unsupported comm type {comm_name}")

        for k, v in [
            ("fwd_compute", layer.fwd_compute),
            ("fwd_comm_size", layer.fwd_comm_size),
            ("bwd_compute", layer.bwd_compute),
            ("bwd_comm_size", layer.bwd_comm_size),
            ("wgrad_compute", layer.wgrad_compute),
            ("wgrad_comm_size", layer.wgrad_comm_size),
            ("update_compute", layer.update_compute),
        ]:
            if v < 0:
                raise ValueError(f"Layer {layer.name}: {k} must be >= 0")


# =============================================================================
# Group / sharding helpers
# =============================================================================
def ranks_in_group(rank: int, num_groups: int, num_npus: int) -> Tuple[int, List[int]]:
    group_size = num_npus // num_groups
    group_id = rank // group_size
    start = group_id * group_size
    return group_id, list(range(start, start + group_size))


def participants_for_group(group_id: int, num_groups: int, num_npus: int) -> List[int]:
    group_size = num_npus // num_groups
    start = group_id * group_size
    return list(range(start, start + group_size))


def tables_owned_by_rank(rank: int, num_npus: int, num_tables: int) -> List[int]:
    return [t for t in range(num_tables) if (t % num_npus) == rank]


def group_for_layer(layer: LayerSpec, default_group_id: int) -> int:
    return default_group_id if layer.group_id == -1 else layer.group_id


def participants_for_layer(
    layer: LayerSpec,
    rank: int,
    num_groups: int,
    num_npus: int,
) -> Tuple[int, List[int]]:
    default_gid, _ = ranks_in_group(rank, num_groups, num_npus)
    gid = group_for_layer(layer, default_gid)
    return gid, participants_for_group(gid, num_groups, num_npus)


def overlap_deps(main_tail: Optional[int], async_tail: Optional[int]) -> List[int]:
    deps: List[int] = []
    if main_tail is not None:
        deps.append(main_tail)
    if async_tail is not None:
        deps.append(async_tail)
    return deps


# =============================================================================
# Workload shaping
# =============================================================================
def embedding_lookup_runtime_us(table_id: int, owner_rank: int, rank: int) -> int:
    base = 220 + 20 * (table_id % 5)
    if owner_rank == rank:
        return base
    return max(base // 4, 12)


def embedding_exchange_size_bytes(table_id: int) -> int:
    return 32768 * (1 + (table_id % 3))


# =============================================================================
# ET writer
# =============================================================================
class ETWriter:
    def __init__(self, fh):
        self.fh = fh
        self.next_id = 0
        self.dep_map: Dict[int, List[int]] = {}

    def _new_node(self, name: str, node_type: int) -> Node:
        node = Node()
        node.id = self.next_id
        node.name = name
        node.type = node_type
        self.next_id += 1
        return node

    def _set_deps(self, node: Node, dep_ids: Sequence[int]) -> None:
        deps = [int(d) for d in dep_ids]
        node.data_deps.extend(deps)
        self.dep_map[int(node.id)] = deps

    def add_comp(
        self,
        name: str,
        runtime_us: int,
        dep_ids: Optional[Sequence[int]] = None,
        attrs: Optional[Iterable[AttributeProto]] = None,
    ) -> int:
        deps = list(dep_ids or [])
        node = self._new_node(name, COMP_NODE)
        self._set_deps(node, deps)

        node.duration_micros = int(runtime_us)
        node.attr.append(_attr_bool("is_cpu_op", False))
        if attrs:
            node.attr.extend(list(attrs))

        encodeMessage(self.fh, node)
        return int(node.id)

    def add_comm(
        self,
        name: str,
        comm_type_str: str,
        comm_size: int,
        dep_ids: Optional[Sequence[int]] = None,
        group_id: Optional[int] = None,
        participants: Optional[Sequence[int]] = None,
        overlap_slot: Optional[str] = None,
        attrs: Optional[Iterable[AttributeProto]] = None,
    ) -> Optional[int]:
        deps = list(dep_ids or [])
        if comm_type_str == "NONE":
            return deps[-1] if deps else None

        comm_enum = COMM_TYPE_MAP[comm_type_str]
        if comm_enum is None:
            raise ValueError(f"Unsupported communication type: {comm_type_str}")

        node = self._new_node(name, COMM_COLL_NODE)
        self._set_deps(node, deps)

        node.attr.append(_attr_bool("is_cpu_op", False))
        node.attr.append(_attr_int("comm_type", int(comm_enum)))
        node.attr.append(_attr_int("comm_size", int(comm_size)))
        node.attr.append(_attr_str("comm_mode", comm_type_str))

        if group_id is not None:
            node.attr.append(_attr_int("comm_group", int(group_id)))
        if participants is not None:
            parts = list(participants)
            node.attr.append(_attr_int("comm_group_size", len(parts)))
            node.attr.append(_attr_ints("participant_ranks", parts))
        if overlap_slot is not None:
            node.attr.append(_attr_str("overlap_slot", overlap_slot))

        if attrs:
            node.attr.extend(list(attrs))

        encodeMessage(self.fh, node)
        return int(node.id)

    def validate_graph(self) -> None:
        all_ids = set(self.dep_map.keys())
        for node_id, deps in self.dep_map.items():
            for dep in deps:
                if dep not in all_ids:
                    raise ValueError(f"Node {node_id} depends on missing node {dep}")
                if dep >= node_id:
                    raise ValueError(f"Node {node_id} depends on future/same node {dep}")


# =============================================================================
# Generation
# =============================================================================
def generate_one_rank(rank: int, spec: WorkloadSpec) -> str:
    output_dir = "DLRM_new"
    os.makedirs(output_dir, exist_ok=True)
    out_name = os.path.join(output_dir, f"{spec.workload_name.lower()}.{rank}.et")

    default_group_id, default_participants = ranks_in_group(
        rank, spec.num_groups, spec.num_npus
    )
    owned_tables = tables_owned_by_rank(rank, spec.num_npus, spec.num_embedding_tables)

    with open(out_name, "wb") as f:
        encodeMessage(f, GlobalMetadata(version="0.0.4"))
        w = ETWriter(f)

        special_id = w.add_comp(
            f"rank{rank}_special",
            1,
            dep_ids=[],
            attrs=[
                _attr_int("rank", rank),
                _attr_int("num_groups", spec.num_groups),
                _attr_int("num_npus", spec.num_npus),
                _attr_int("default_dp_group", default_group_id),
                _attr_ints("default_dp_participants", default_participants),
                _attr_ints("owned_embedding_tables", owned_tables),
            ],
        )

        # ---------------------------------------------------------------------
        # Embedding forward
        # ---------------------------------------------------------------------
        embedding_lookup_nodes: List[int] = []
        embedding_comm_nodes: List[int] = []

        for table_id in range(spec.num_embedding_tables):
            owner_rank = table_id % spec.num_npus

            lookup_id = w.add_comp(
                f"EmbeddingTable_{table_id}_lookup_rank{rank}",
                embedding_lookup_runtime_us(table_id, owner_rank, rank),
                dep_ids=[special_id],
                attrs=[
                    _attr_str("layer_kind", "embedding_table"),
                    _attr_str("phase", "forward"),
                    _attr_int("embedding_table_id", table_id),
                    _attr_int("owner_rank", owner_rank),
                    _attr_bool("owns_table_shard", owner_rank == rank),
                ],
            )
            embedding_lookup_nodes.append(lookup_id)

            comm_id = w.add_comm(
                f"EmbeddingTable_{table_id}_alltoall_rank{rank}",
                "ALLTOALL",
                embedding_exchange_size_bytes(table_id),
                dep_ids=[lookup_id],
                group_id=default_group_id,
                participants=default_participants,
                overlap_slot=f"emb_fwd_{table_id}",
                attrs=[
                    _attr_str("phase", "forward"),
                    _attr_str("semantic", "embedding_feature_exchange"),
                    _attr_int("embedding_table_id", table_id),
                ],
            )
            if comm_id is not None:
                embedding_comm_nodes.append(comm_id)

        interaction_id = w.add_comp(
            f"rank{rank}_feature_interaction",
            600 + 25 * len(owned_tables),
            dep_ids=embedding_lookup_nodes,
            attrs=[
                _attr_str("layer_kind", "feature_interaction"),
                _attr_str("phase", "forward"),
                _attr_int("owned_embedding_table_count", len(owned_tables)),
                _attr_int("total_embedding_table_count", spec.num_embedding_tables),
            ],
        )

        fwd_anchor = w.add_comp(
            f"rank{rank}_fwd_anchor",
            1,
            dep_ids=[interaction_id] + embedding_comm_nodes,
            attrs=[_attr_str("phase", "forward_join")],
        )

        # ---------------------------------------------------------------------
        # Forward MLP
        # ---------------------------------------------------------------------
        main_tail: Optional[int] = fwd_anchor
        async_comm_tail: Optional[int] = None

        for idx, layer in enumerate(spec.layers):
            layer_group_id, participants = participants_for_layer(
                layer, rank, spec.num_groups, spec.num_npus
            )

            comp_id = w.add_comp(
                f"{layer.name}_fwd_comp_rank{rank}",
                layer.fwd_compute,
                dep_ids=overlap_deps(main_tail, None),
                attrs=[
                    _attr_str("layer_kind", "mlp"),
                    _attr_str("phase", "forward"),
                    _attr_str("layer_name", layer.name),
                    _attr_int("layer_group_id", layer_group_id),
                ],
            )

            comm_id = w.add_comm(
                f"{layer.name}_fwd_comm_rank{rank}",
                layer.fwd_comm_type,
                layer.fwd_comm_size,
                dep_ids=[comp_id],
                group_id=layer_group_id,
                participants=participants,
                overlap_slot=f"mlp_fwd_{idx}",
                attrs=[
                    _attr_str("phase", "forward"),
                    _attr_str("semantic", "activation_exchange"),
                    _attr_str("layer_name", layer.name),
                ],
            )

            main_tail = comp_id
            async_comm_tail = comm_id

        bwd_seed = w.add_comp(
            f"rank{rank}_bwd_seed",
            1,
            dep_ids=overlap_deps(main_tail, async_comm_tail),
            attrs=[_attr_str("phase", "backward_seed")],
        )

        # ---------------------------------------------------------------------
        # Backward / weight gradient
        # ---------------------------------------------------------------------
        update_ready: Dict[str, int] = {}
        main_tail = bwd_seed
        async_comm_tail = None

        for rev_idx, layer in enumerate(reversed(spec.layers)):
            layer_group_id, participants = participants_for_layer(
                layer, rank, spec.num_groups, spec.num_npus
            )

            bwd_comp_id = w.add_comp(
                f"{layer.name}_bwd_comp_rank{rank}",
                layer.bwd_compute,
                dep_ids=overlap_deps(main_tail, None),
                attrs=[
                    _attr_str("layer_kind", "mlp"),
                    _attr_str("phase", "backward"),
                    _attr_str("layer_name", layer.name),
                ],
            )

            bwd_comm_id = w.add_comm(
                f"{layer.name}_bwd_comm_rank{rank}",
                layer.bwd_comm_type,
                layer.bwd_comm_size,
                dep_ids=[bwd_comp_id],
                group_id=layer_group_id,
                participants=participants,
                overlap_slot=f"mlp_bwd_{rev_idx}",
                attrs=[
                    _attr_str("phase", "backward"),
                    _attr_str("semantic", "activation_grad_exchange"),
                    _attr_str("layer_name", layer.name),
                ],
            )

            wgrad_comp_id = w.add_comp(
                f"{layer.name}_wgrad_comp_rank{rank}",
                layer.wgrad_compute,
                dep_ids=[bwd_comp_id],
                attrs=[
                    _attr_str("layer_kind", "mlp"),
                    _attr_str("phase", "wgrad"),
                    _attr_str("layer_name", layer.name),
                ],
            )

            wgrad_comm_id = w.add_comm(
                f"{layer.name}_wgrad_comm_rank{rank}",
                layer.wgrad_comm_type,
                layer.wgrad_comm_size,
                dep_ids=[wgrad_comp_id],
                group_id=layer_group_id,
                participants=participants,
                overlap_slot=f"mlp_wgrad_{rev_idx}",
                attrs=[
                    _attr_str("phase", "wgrad"),
                    _attr_str("semantic", "weight_grad_sync"),
                    _attr_str("layer_name", layer.name),
                ],
            )

            ready_id = w.add_comp(
                f"{layer.name}_ready_for_update_rank{rank}",
                1,
                dep_ids=[wgrad_comp_id] + ([wgrad_comm_id] if wgrad_comm_id is not None else []),
                attrs=[
                    _attr_str("phase", "update_join"),
                    _attr_str("layer_name", layer.name),
                ],
            )
            update_ready[layer.name] = ready_id

            main_tail = wgrad_comp_id
            async_comm_tail = bwd_comm_id if bwd_comm_id is not None else wgrad_comm_id

        # ---------------------------------------------------------------------
        # Update
        # ---------------------------------------------------------------------
        update_nodes: List[int] = []
        for layer in spec.layers:
            upd_id = w.add_comp(
                f"{layer.name}_update_rank{rank}",
                layer.update_compute,
                dep_ids=[update_ready[layer.name]],
                attrs=[
                    _attr_str("layer_kind", "optimizer"),
                    _attr_str("phase", "update"),
                    _attr_str("layer_name", layer.name),
                ],
            )
            update_nodes.append(upd_id)

        final_deps = list(update_nodes)
        if async_comm_tail is not None:
            final_deps.append(async_comm_tail)

        w.add_comp(
            "iteration_done",
            1,
            dep_ids=final_deps,
            attrs=[_attr_str("phase", "done")],
        )

        w.validate_graph()

    return out_name


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    spec = parse_spec(SPEC)

    print(f"workload_name         = {spec.workload_name}")
    print(f"num_groups            = {spec.num_groups}")
    print(f"num_npus              = {spec.num_npus}")
    print(f"num_embedding_tables  = {spec.num_embedding_tables}")
    print("Generating ET files...")

    outputs: List[str] = []
    for rank in range(spec.num_npus):
        outputs.append(generate_one_rank(rank, spec))

    print("Done.")
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()