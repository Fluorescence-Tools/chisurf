"""External database cross-references for sample entities (PRD-39).

This module connects an :class:`~mfdb.models.EntityDefinition` to its
source database records — **UniProt** for the canonical sequence/organism and
**PDB** for the structure — and turns the difference between a construct sequence
and its reference sequence into structured
:class:`~mfdb.models.MutationDefinition` records (the standard PDBx
``struct_ref_seq_dif`` mechanism).

All network access is optional and offline-safe: fetch helpers return ``None`` on
failure so callers degrade gracefully to manual entry. The diffing logic
(:func:`diff_sequences`) is pure and requires no network.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

from mfdb.models import MutationDefinition

__all__ = [
    "ONE_TO_THREE",
    "diff_sequences",
    "fetch_uniprot",
    "fetch_sifts_uniprot_mapping",
]

UNIPROT_REST = "https://rest.uniprot.org/uniprotkb/{accession}.json"
SIFTS_REST = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"

#: Single-letter → three-letter amino-acid code (used to build ``comp_id`` values
#: and author labels such as ``S48C``).
ONE_TO_THREE: dict[str, str] = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def diff_sequences(
    construct: str,
    reference: str,
    *,
    numbering_start: int = 1,
) -> list[MutationDefinition]:
    """Diff a construct sequence against its reference, returning mutations.

    Compares the two one-letter sequences position-by-position and emits one
    :class:`MutationDefinition` per substitution. This covers the dominant
    smFRET case — point substitutions (e.g. engineered cysteines ``S48C``),
    which do not change sequence length.

    Parameters
    ----------
    construct : str
        Construct (as-measured) one-letter sequence.
    reference : str
        Reference / wild-type one-letter sequence (e.g. from UniProt).
    numbering_start : int, optional
        Residue number assigned to the first position (default ``1``).

    Returns
    -------
    list of MutationDefinition
        One record per differing position. ``auth_name`` follows the
        ``<wt><pos><mut>`` convention (e.g. ``"S48C"``).

    Raises
    ------
    ValueError
        If the sequences differ in length. Indels require an explicit
        alignment, which is out of scope for this positional diff.
    """
    construct = (construct or "").strip().upper()
    reference = (reference or "").strip().upper()
    if len(construct) != len(reference):
        raise ValueError(
            "diff_sequences requires equal-length sequences "
            f"(construct={len(construct)}, reference={len(reference)}); "
            "use an alignment for insertions/deletions"
        )

    mutations: list[MutationDefinition] = []
    for offset, (mut_aa, wt_aa) in enumerate(zip(construct, reference)):
        if mut_aa == wt_aa:
            continue
        seq_id = numbering_start + offset
        mutations.append(
            MutationDefinition(
                seq_id=seq_id,
                mut_comp_id=ONE_TO_THREE.get(mut_aa, mut_aa),
                wt_comp_id=ONE_TO_THREE.get(wt_aa, wt_aa),
                auth_name=f"{wt_aa}{seq_id}{mut_aa}",
                kind="engineered_mutation",
            )
        )
    return mutations


def fetch_uniprot(
    accession: str,
    *,
    cache_dir: Optional[Path] = None,
    timeout: float = 10.0,
) -> Optional[dict]:
    """Fetch sequence, organism and entry name from the UniProt REST API.

    The raw JSON response is cached on disk (when ``cache_dir`` is given) so
    repeated lookups are offline. Network or parse failures return ``None`` —
    callers must treat external references as optional.

    Parameters
    ----------
    accession : str
        UniProt accession, e.g. ``"P00720"``.
    cache_dir : Path, optional
        Directory for the on-disk JSON cache. If ``None``, no caching.
    timeout : float, optional
        Network timeout in seconds.

    Returns
    -------
    dict or None
        ``{"accession", "entry_name", "organism", "sequence"}`` on success,
        otherwise ``None``.
    """
    accession = (accession or "").strip()
    if not accession:
        return None
    payload = _fetch_json(
        UNIPROT_REST.format(accession=accession),
        cache_dir=cache_dir,
        cache_name=f"uniprot_{accession}.json",
        timeout=timeout,
    )
    if not isinstance(payload, dict):
        return None
    return {
        "accession": accession,
        "entry_name": payload.get("uniProtkbId"),
        "organism": (payload.get("organism") or {}).get("scientificName"),
        "sequence": (payload.get("sequence") or {}).get("value"),
    }


def fetch_sifts_uniprot_mapping(
    pdb_id: str,
    *,
    chain_id: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    timeout: float = 10.0,
) -> Optional[list[dict]]:
    """Fetch PDB→UniProt residue mappings from the EBI SIFTS REST API.

    Resolves a PDB chain to its UniProt accession and the residue-numbering
    offset, used to align author/PDB numbering (e.g. ``S48C``) into UniProt
    numbering (``struct_ref_seq`` offsets, PRD-39). Offline-safe: returns
    ``None`` on any network/parse failure.

    Parameters
    ----------
    pdb_id : str
        PDB identifier, e.g. ``"2lzm"`` (case-insensitive).
    chain_id : str, optional
        Restrict the result to a single author chain. If ``None``, all chains
        are returned.
    cache_dir : Path, optional
        Directory for the on-disk JSON cache.
    timeout : float, optional
        Network timeout in seconds.

    Returns
    -------
    list of dict or None
        One entry per chain segment with keys ``accession``, ``chain_id``,
        ``pdb_start``, ``pdb_end``, ``unp_start``, ``unp_end``. ``None`` on
        failure.
    """
    pdb_id = (pdb_id or "").strip().lower()
    if not pdb_id:
        return None
    payload = _fetch_json(
        SIFTS_REST.format(pdb_id=pdb_id),
        cache_dir=cache_dir,
        cache_name=f"sifts_{pdb_id}.json",
        timeout=timeout,
    )
    if not isinstance(payload, dict):
        return None

    segments: list[dict] = []
    uniprot = ((payload.get(pdb_id) or {}).get("UniProt")) or {}
    for accession, entry in uniprot.items():
        for mapping in entry.get("mappings", []):
            seg_chain = mapping.get("chain_id") or mapping.get("struct_asym_id")
            if chain_id is not None and seg_chain != chain_id:
                continue
            segments.append(
                {
                    "accession": accession,
                    "chain_id": seg_chain,
                    "pdb_start": (mapping.get("start") or {}).get("author_residue_number"),
                    "pdb_end": (mapping.get("end") or {}).get("author_residue_number"),
                    "unp_start": mapping.get("unp_start"),
                    "unp_end": mapping.get("unp_end"),
                }
            )
    return segments


def _fetch_json(
    url: str,
    *,
    cache_dir: Optional[Path],
    cache_name: str,
    timeout: float,
) -> Optional[object]:
    """Fetch and JSON-parse ``url``, with an optional on-disk cache.

    Returns the parsed payload, or ``None`` on any network/parse/IO failure so
    callers degrade gracefully (offline-safe).
    """
    cache_file: Optional[Path] = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / cache_name
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except (OSError, ValueError):
                pass

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, OSError):
        return None

    if cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(payload))
        except OSError:
            pass
    return payload
