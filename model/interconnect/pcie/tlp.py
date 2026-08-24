"""Lightweight PCIe TLP abstraction (Part 8)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from model.interconnect.pcie.config import PCIeConfig, PCIeMPS

# Analytical 3-DW TLP header (ASSUMPTION — not full PCIe spec).
TLP_HEADER_BYTES = 12
COMPLETION_HEADER_BYTES = 12


class TLPType(str, Enum):
    MEMORY_READ = "MEMORY_READ"
    MEMORY_WRITE = "MEMORY_WRITE"
    COMPLETION = "COMPLETION"
    COMPLETION_WITH_DATA = "COMPLETION_WITH_DATA"


@dataclass
class TLP:
    type: TLPType
    payload_bytes: int
    address: int = 0
    requester: str = ""
    completer: str = ""
    sequence_number: int = 0
    tag: int = 0
    mps: int = 256

    @property
    def on_wire_bytes(self) -> int:
        if self.type in (TLPType.MEMORY_READ, TLPType.COMPLETION):
            return TLP_HEADER_BYTES
        if self.type == TLPType.COMPLETION_WITH_DATA:
            return COMPLETION_HEADER_BYTES + self.payload_bytes
        return TLP_HEADER_BYTES + self.payload_bytes


def _mps_value(mps: int | PCIeMPS) -> int:
    return mps.value if isinstance(mps, PCIeMPS) else int(mps)


def fragment_transfer(
    size_bytes: int,
    mps: int | PCIeMPS,
    *,
    tlp_type: TLPType = TLPType.MEMORY_WRITE,
    address: int = 0,
    requester: str = "",
    completer: str = "",
) -> list[TLP]:
    """
    Split a logical transfer into payload TLPs bounded by MPS.

    Example: 4096 B with MPS=256 → 16 MEMORY_WRITE TLPs.
    """
    if size_bytes <= 0:
        return []

    chunk = _mps_value(mps)
    tlps: list[TLP] = []
    remaining = size_bytes
    offset = 0
    seq = 0
    while remaining > 0:
        payload = min(chunk, remaining)
        tlps.append(
            TLP(
                type=tlp_type,
                payload_bytes=payload,
                address=address + offset,
                requester=requester,
                completer=completer,
                sequence_number=seq,
                tag=seq,
                mps=chunk,
            )
        )
        remaining -= payload
        offset += payload
        seq += 1
    return tlps


def fragment_read_transfer(
    size_bytes: int,
    mps: int | PCIeMPS,
    mrrs: int,
    *,
    address: int = 0,
    requester: str = "",
    completer: str = "",
) -> tuple[list[TLP], list[TLP]]:
    """
    PCIe read: memory read request TLPs (bounded by MRRS) plus
    completion-with-data TLPs (bounded by MPS).
    """
    if size_bytes <= 0:
        return [], []

    mps_val = _mps_value(mps)
    read_requests: list[TLP] = []
    completions: list[TLP] = []

    remaining = size_bytes
    offset = 0
    seq = 0
    while remaining > 0:
        req_payload = min(mrrs, remaining)
        read_requests.append(
            TLP(
                type=TLPType.MEMORY_READ,
                payload_bytes=req_payload,
                address=address + offset,
                requester=requester,
                completer=completer,
                sequence_number=seq,
                tag=seq,
                mps=mps_val,
            )
        )
        remaining -= req_payload
        offset += req_payload
        seq += 1

    remaining = size_bytes
    offset = 0
    seq = 0
    while remaining > 0:
        cpl_payload = min(mps_val, remaining)
        completions.append(
            TLP(
                type=TLPType.COMPLETION_WITH_DATA,
                payload_bytes=cpl_payload,
                address=address + offset,
                requester=completer,
                completer=requester,
                sequence_number=seq,
                tag=seq,
                mps=mps_val,
            )
        )
        remaining -= cpl_payload
        offset += cpl_payload
        seq += 1

    return read_requests, completions


def write_on_wire_bytes(size_bytes: int, mps: int | PCIeMPS) -> int:
    tlps = fragment_transfer(size_bytes, mps, tlp_type=TLPType.MEMORY_WRITE)
    return sum(tlp.on_wire_bytes for tlp in tlps)


def read_on_wire_bytes(size_bytes: int, mps: int | PCIeMPS, mrrs: int) -> int:
    requests, completions = fragment_read_transfer(size_bytes, mps, mrrs)
    return sum(tlp.on_wire_bytes for tlp in requests) + sum(tlp.on_wire_bytes for tlp in completions)


def compute_transfer_on_wire_bytes(
    config: PCIeConfig,
    payload_bytes: int,
    *,
    is_read: bool,
) -> int:
    """Total on-wire bytes including TLP header overhead."""
    if payload_bytes <= 0:
        return 0
    if is_read:
        return read_on_wire_bytes(payload_bytes, config.mps, config.mrrs.value)
    return write_on_wire_bytes(payload_bytes, config.mps)


def tlp_summary(size_bytes: int, mps: int | PCIeMPS) -> dict[str, int | float]:
    tlps = fragment_transfer(size_bytes, mps, tlp_type=TLPType.MEMORY_WRITE)
    on_wire = sum(t.on_wire_bytes for t in tlps)
    header_overhead = on_wire - size_bytes
    return {
        "payload_bytes": size_bytes,
        "tlp_count": len(tlps),
        "on_wire_bytes": on_wire,
        "header_overhead_bytes": header_overhead,
        "efficiency": size_bytes / on_wire if on_wire else 1.0,
    }
