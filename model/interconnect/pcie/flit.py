"""Lightweight PCIe Gen6 FLIT abstraction (Part 9).

256-byte FLIT layout (PCIe 6.0 / pcie_impl.md):

    |<-------- 236 B TLP/DLP payload -------->|<- DLP ->|<-- CRC -->|<- FEC ->|
    |              usable payload              |  6 B    |   8 B     |  6 B     |
    |<---------------------- 256 B total FLIT ------------------------------->|
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from model.interconnect.pcie.config import PCIeGeneration

FLIT_SIZE_BYTES = 256
FLIT_TLP_PAYLOAD_AREA_BYTES = 236
FLIT_DLP_BYTES = 6
FLIT_CRC_BYTES = 8
FLIT_FEC_BYTES = 6
GEN6_FLIT_EFFICIENCY = FLIT_TLP_PAYLOAD_AREA_BYTES / FLIT_SIZE_BYTES


@dataclass
class FLIT:
    """256-byte Gen6 FLIT with analytical overhead areas."""

    index: int
    tlp_payload_bytes: int
    dlp_bytes: int = FLIT_DLP_BYTES
    crc_bytes: int = FLIT_CRC_BYTES
    fec_bytes: int = FLIT_FEC_BYTES

    @property
    def overhead_bytes(self) -> int:
        return self.dlp_bytes + self.crc_bytes + self.fec_bytes

    @property
    def total_bytes(self) -> int:
        return FLIT_SIZE_BYTES

    @property
    def efficiency(self) -> float:
        return self.tlp_payload_bytes / FLIT_SIZE_BYTES if FLIT_SIZE_BYTES else 0.0


def flit_count_for_bytes(on_wire_bytes: int, payload_area_bytes: int = FLIT_TLP_PAYLOAD_AREA_BYTES) -> int:
    """Number of FLITs required to carry ``on_wire_bytes`` of TLP/DLP data."""
    if on_wire_bytes <= 0:
        return 0
    return math.ceil(on_wire_bytes / payload_area_bytes)


def pack_bytes_into_flits(on_wire_bytes: int) -> list[FLIT]:
    """
    Pack serialized TLP byte stream into 256 B FLITs.

    Each FLIT carries up to 236 B in the TLP/DLP payload region; a large TLP
    may span multiple FLITs (one TLP != one FLIT).
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
    return flit_count_for_bytes(on_wire_bytes) * FLIT_SIZE_BYTES


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
        "flit_payload_area_bytes": FLIT_TLP_PAYLOAD_AREA_BYTES,
        "dlp_bytes": FLIT_DLP_BYTES,
        "crc_bytes": FLIT_CRC_BYTES,
        "fec_bytes": FLIT_FEC_BYTES,
        "eta_flit": GEN6_FLIT_EFFICIENCY,
        "tlp_on_wire_bytes": on_wire_bytes,
        "flit_on_wire_bytes": total,
        "flit_overhead_bytes": total - on_wire_bytes,
        "efficiency": on_wire_bytes / total if total else 1.0,
        "payload_carried_bytes": payload,
    }
