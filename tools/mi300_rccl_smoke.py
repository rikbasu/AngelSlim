#!/usr/bin/env python3
"""Minimal RCCL all-reduce smoke for a multi-GPU ROCm training host."""

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    value = torch.tensor(float(dist.get_rank() + 1), device="cuda")
    dist.all_reduce(value)
    torch.cuda.synchronize()
    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"rank {dist.get_rank()}: {value.item()} != {expected}")
    print(
        f"rank={dist.get_rank()} device={torch.cuda.get_device_name(local_rank)} "
        f"all_reduce={value.item()}"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
