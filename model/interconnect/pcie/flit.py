"""Lightweight PCIe Gen6 FLIT abstraction (Part 9)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from model.interconnect.pcie.config import PCIeGeneration

FLIT_SIZE_BYTES = 256
FLIT_TLP_PAYLOAD_AREA_BYTES = 220
FLIT_DL_INFO_BYTES = 8
FLIT_CRC_BYTES = 4
FLIT_FEC_BYTES = 24


@dataclass
class FLIT:
    """256-byte Gen6 FLIT with analytical overhead areas."""

    index: int
    tlp_payload_bytes: int
    dl_info_bytes: int = FLIT_DL_INFO_BYTES
    crc_bytes: int = FLIT_CRC_BYTES
    fec_bytes: int = FLIT_FEC_BYTES

    @property
    def overhead_bytes(self) -> int:
        return self.dl_info_bytes + self.crc_bytes + self.fec_bytes

    @property
    def total_bytes(self) -> int:
        return FLIT_SIZE_BYTES

    @property
    def efficiency(self) -> float:
        return self.tlp_payload_bytes / FLIT_SIZE_BYTES if FLIT_SIZE_BYTES else 0.0


def pack_bytes_into_flits(on_wire_bytes: int) -> list[FLIT]:
    """
    Pack serialized TLP byte stream into 256 B FLITs.

    TLP payload area per FLIT is capped at FLIT_TLP_PAYLOAD_AREA_BYTES;
    a large TLP may span multiple FLITs.
    """
    if on_wire_bytes <= 0:
        return []

    flits: list[FLIT] = []
    remaining = on_wire_bytes
    index = 0
    while remaining > 0:
        payload = min(FLIT_TLP_PAYLOAD_AREA_BYTES, remaining)
        flits.append(FLIT(index=index, tlp_payload_bytes=payload))
        remaining -= payload
        index += 1
    return flits


def flit_on_wire_bytes(on_wire_bytes: int) -> int:
    """Total bytes transmitted after Gen6 FLIT encapsulation."""
    if on_wire_bytes <= 0:
        return 0
    flit_count = math.ceil(on_wire_bytes / FLIT_TLP_PAYLOAD_AREA_BYTES)
    return flit_count * FLIT_SIZE_BYTES


def apply_flit_overhead(on_wire_bytes: int, generation: PCIeGeneration | str) -> int:
    """Apply FLIT packing for Gen6; pass through for earlier generations."""
    gen = generation.value if isinstance(generation, PCIeGeneration) else str(generation)
    if gen != "Gen6":
        return on_wire_bytes
    return flit_on_wire_bytes(on_wire_bytes)


def flit_summary(on_wire_bytes: int) -> dict[str, int | float]:
    flits = pack_bytes_into_flits(on_wire_bytes)
    total = len(flits) * FLIT_SIZE_BYTES
    payload = sum(f.tlp_payload_bytes for f in flits)
    return {
        "flit_count": len(flits),
        "flit_size_bytes": FLIT_SIZE_BYTES,
        "tlp_on_wire_bytes": on_wire_bytes,
        "flit_on_wire_bytes": total,
        "flit_overhead_bytes": total - on_wire_bytes,
        "efficiency": on_wire_bytes / total if total else 1.0,
        "payload_carried_bytes": payload,
    }
