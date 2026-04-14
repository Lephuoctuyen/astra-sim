from et_def_pb2 import Node, AttributeProto, COMM_SEND_NODE, COMM_RECV_NODE
from protolib import encodeMessage

NUM_RANKS = 8
COMM_SIZE = 98304  # có thể thay đổi nếu cần

def create_send(node_id, peer):
    n = Node()
    n.id = node_id
    n.name = f"send_to_{peer}"
    n.type = COMM_SEND_NODE
    n.attr.append(AttributeProto(name="peer", int64_val=peer))
    n.attr.append(AttributeProto(name="comm_size", int64_val=COMM_SIZE))
    return n

def create_recv(node_id, peer):
    n = Node()
    n.id = node_id
    n.name = f"recv_from_{peer}"
    n.type = COMM_RECV_NODE
    n.attr.append(AttributeProto(name="peer", int64_val=peer))
    n.attr.append(AttributeProto(name="comm_size", int64_val=COMM_SIZE))
    return n

def generate_rank(rank):
    filename = f"custom_pairwise_alltoall_8npus/custom_alltoall.{rank}.et"
    f = open(filename, "wb")

    node_id = 0

    for peer in range(NUM_RANKS):
        if peer == rank:
            continue

        send = create_send(node_id, peer)
        encodeMessage(f, send)
        node_id += 1

        recv = create_recv(node_id, peer)
        encodeMessage(f, recv)
        node_id += 1

    f.close()

if __name__ == "__main__":
    import os
    os.makedirs("custom_pairwise_alltoall_8npus", exist_ok=True)

    for r in range(NUM_RANKS):
        generate_rank(r)

    print("Done generating ALLTOALL ET templates")
