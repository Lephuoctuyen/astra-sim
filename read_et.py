#!/usr/bin/env python3
import sys
from pathlib import Path
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import protobuf classes từ repo astra-sim
from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (
    GlobalMetadata,
    Node,
)


def read_varint(f):
    """Đọc protobuf varint dùng làm độ dài message."""
    shift = 0
    result = 0
    while True:
        b = f.read(1)
        if not b:
            return None  # EOF
        byte = b[0]
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result
        shift += 7
        if shift > 63:
            raise ValueError("Varint quá dài hoặc file lỗi format")


def read_delimited_message(f, message_cls):
    """Đọc 1 message dạng length-delimited protobuf."""
    size = read_varint(f)
    if size is None:
        return None
    payload = f.read(size)
    if len(payload) != size:
        raise EOFError("Không đọc đủ payload của protobuf message")
    msg = message_cls()
    msg.ParseFromString(payload)
    return msg


def attr_value(attr):
    """Lấy giá trị dễ đọc từ AttributeProto."""
    candidates = [
        "bool_val",
        "int32_val",
        "int64_val",
        "uint32_val",
        "uint64_val",
        "float_val",
        "double_val",
        "string_val",
    ]
    for name in candidates:
        if hasattr(attr, name):
            try:
                if attr.HasField(name):
                    return getattr(attr, name)
            except ValueError:
                # Với proto3/scalar đôi khi HasField không dùng được
                value = getattr(attr, name)
                if value not in (0, 0.0, False, ""):
                    return value
    return None


def dump_node(node, idx, lines):
    lines.append(f"\n=== Node #{idx} ===")

    # In các field cơ bản nếu có
    for field_name in ["id", "name", "type", "duration_micros", "num_ops", "comm_size"]:
        if hasattr(node, field_name):
            value = getattr(node, field_name)
            if value not in (0, "", [], None):
                lines.append(f"{field_name}: {value}")

    # In dependencies nếu có
    for dep_field in ["data_deps", "ctrl_deps", "parent", "children"]:
        if hasattr(node, dep_field):
            value = getattr(node, dep_field)
            if value:
                lines.append(f"{dep_field}: {list(value)}")

    # In attributes
    if hasattr(node, "attr") and node.attr:
        lines.append("attributes:")
        for a in node.attr:
            key = getattr(a, "name", "<unknown>")
            val = attr_value(a)
            lines.append(f"  - {key}: {val}")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <path_to_et_file> <output_file>")
        sys.exit(1)

    et_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not et_path.exists():
        print(f"Không tìm thấy file: {et_path}")
        sys.exit(1)

    lines = []

    with et_path.open("rb") as f:
        # Message đầu tiên thường là GlobalMetadata
        meta = read_delimited_message(f, GlobalMetadata)
        if meta is None:
            output_path.write_text("File rỗng.\n", encoding="utf-8")
            print(f"Đã ghi kết quả ra: {output_path}")
            return

        lines.append("=== GlobalMetadata ===")
        if hasattr(meta, "version"):
            lines.append(f"version: {meta.version}")
        else:
            lines.append(str(meta))

        idx = 0
        while True:
            try:
                node = read_delimited_message(f, Node)
            except EOFError as e:
                lines.append(f"\n[!] Lỗi khi đọc node cuối: {e}")
                break

            if node is None:
                break

            dump_node(node, idx, lines)
            idx += 1

        lines.append(f"\nĐọc xong {idx} node.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Đã ghi kết quả ra: {output_path}")


if __name__ == "__main__":
    main()