from __future__ import annotations

from typing import Tuple

import numpy as np

from boltz.data.types import Record, Structure

ALIGNMENT_PREFIXES = ("A", "B")
RMSD_PREFIXES = ("C", "D")


def ensure_cyclic_period_field(chains: np.ndarray) -> np.ndarray:
    if "cyclic_period" in chains.dtype.names:
        return chains

    dtype = chains.dtype.descr + [("cyclic_period", "<i4")]
    new_chains = np.empty(chains.shape, dtype=dtype)
    for name in chains.dtype.names:
        new_chains[name] = chains[name]
    new_chains["cyclic_period"] = 0
    return new_chains


def build_chain_masks(structure: Structure, record: Record) -> Tuple[np.ndarray, np.ndarray]:
    """Build alignment and RMSD masks per atom using manifest metadata."""

    num_atoms = len(structure.atoms)
    alignment_mask = np.zeros(num_atoms, dtype=bool)
    rmsd_mask = np.zeros(num_atoms, dtype=bool)

    if record is None or not getattr(record, "chains", None):
        return alignment_mask, rmsd_mask

    asym_id_to_chain = {int(chain["asym_id"]): chain for chain in structure.chains}

    chain_map = {
        (chain.chain_name or "").upper(): chain
        for chain in record.chains
        if getattr(chain, "valid", True)
    }

    def resolve_chain(target: str):
        key = target.upper()
        direct = chain_map.get(key)
        if direct is not None:
            return direct

        preferred = chain_map.get(f"{key}1")
        if preferred is not None:
            return preferred

        matches = [
            chain_map[name]
            for name in chain_map
            if name[: len(key)] == key
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def mark_mask(names: tuple[str, ...], mask: np.ndarray) -> None:
        for name in names:
            record_chain = resolve_chain(name)
            if record_chain is None:
                continue

            chain_array = asym_id_to_chain.get(int(record_chain.chain_id))
            if chain_array is None:
                continue

            start = int(chain_array["atom_idx"])
            end = start + int(chain_array["atom_num"])
            mask[start:end] = True

    mark_mask(ALIGNMENT_PREFIXES, alignment_mask)
    mark_mask(RMSD_PREFIXES, rmsd_mask)

    return alignment_mask, rmsd_mask
