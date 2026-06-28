"""AI-assisted triage for fluorophore data curation.

Runs deterministic sanity checks first, then uses the configured LLM
provider (via ``ai_settings``) to propose category, quality, canonical
name, duplicate clusters, and an approval recommendation. AI proposals
are written to a review queue — the AI never auto-approves.

Usage::

    from chisurf.core.fluorescence.curation.ai_triage import run_triage

    results = run_triage(db, probe_ids=[1, 2, 3])
    # or process all unverified
    results = run_triage(db, status="unverified")
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Deterministic checks (run in plain Python, no LLM) ──────────────


def _check_qy(qy: float | None) -> list[str]:
    issues = []
    if qy is None:
        issues.append("missing QY")
    elif qy < 0:
        issues.append(f"negative QY: {qy}")
    elif qy > 1.0:
        issues.append(f"QY > 1: {qy}")
    return issues


def _check_ext_coeff(ec: float | None) -> list[str]:
    issues = []
    if ec is None:
        issues.append("missing extinction coefficient")
    elif ec <= 0:
        issues.append(f"non-positive extinction coefficient: {ec}")
    elif ec > 1_000_000:
        issues.append(f"implausibly high extinction coefficient: {ec}")
    return issues


def _check_stokes_shift(
    abs_max: float | None, em_max: float | None
) -> list[str]:
    issues = []
    if abs_max is not None and em_max is not None:
        if em_max < abs_max:
            issues.append(
                f"negative Stokes shift: em_max={em_max} < abs_max={abs_max}"
            )
    return issues


def _check_wavelength_range(
    abs_max: float | None, em_max: float | None
) -> list[str]:
    issues = []
    for label, val in [("abs_max", abs_max), ("em_max", em_max)]:
        if val is not None and (val < 200 or val > 1000):
            issues.append(f"{label}={val} outside [200, 1000] nm")
    return issues


def run_deterministic_checks(probe: dict) -> dict:
    """Run numeric sanity checks on a probe dict.

    Parameters
    ----------
    probe : dict
        Probe data with keys like 'abs_max', 'em_max', 'qy', 'ext_coeff'.

    Returns
    -------
    dict
        ``{"issues": [...], "proposed_quality": "low"|"medium"|"high"}``
    """
    issues: list[str] = []
    abs_max = _safe_float(probe.get("abs_max"))
    em_max = _safe_float(probe.get("em_max"))
    qy = _safe_float(probe.get("qy"))
    ext_coeff = _safe_float(probe.get("ext_coeff"))

    issues.extend(_check_qy(qy))
    issues.extend(_check_ext_coeff(ext_coeff))
    issues.extend(_check_stokes_shift(abs_max, em_max))
    issues.extend(_check_wavelength_range(abs_max, em_max))

    has_abs_spec = bool(probe.get("has_abs", probe.get("absorption")))
    has_em_spec = bool(probe.get("has_em", probe.get("emission")))

    if not has_abs_spec:
        issues.append("missing absorption spectrum")
    if not has_em_spec:
        issues.append("missing emission spectrum")

    if not issues:
        proposed_quality = "high"
    elif len(issues) <= 2 and all(
        "missing" not in i for i in issues
    ):
        proposed_quality = "medium"
    else:
        proposed_quality = "low"

    return {"issues": issues, "proposed_quality": proposed_quality}


def _safe_float(v: object) -> float | None:
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(str(v).replace(",", ""))
    except (ValueError, TypeError):
        return None


# ── LLM-based proposal ────────────────────────────────────────────


_AI_TRIAGE_PROMPT = """You are a fluorescence spectroscopy expert reviewing a fluorophore database entry. Your task is to analyze this probe and return a JSON object with your assessment.

Probe data:
{probe_json}

Deterministic checks already found: {issues}

Return valid JSON only with exactly these fields:
```json
{{
  "category": "organic_dye|protein|nucleic_acid|quantum_dot|nanoparticle|other",
  "canonical_name": "the most standard name for this fluorophore",
  "duplicate_of": "name of duplicate if this appears to be a duplicate of another entry, or null",
  "recommendation": "approve|needs_review|reject",
  "rationale": "brief explanation for the recommendation"
}}
```
"""


_LOCAL_HOST_MARKERS = ("localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal")


def _llm_available(provider: str | None = None) -> bool:
    """Return whether a usable LLM provider is configured.

    A provider is usable when it declares a ``base_url`` + ``model`` and either
    carries an ``api_key`` or is a local endpoint (Ollama / LMStudio run keyless
    on localhost). The default (unconfigured) ``ai_settings`` points at the OpenAI
    cloud with an empty key — that is *not* usable, so an unconfigured install
    degrades to deterministic-only instead of 401-hammering a cloud endpoint once
    per probe.
    """
    from chisurf.core.settings.ai_settings import get_api_settings

    settings = get_api_settings(provider)
    base_url = (settings.get("base_url") or "").strip().rstrip("/")
    model = (settings.get("text_model") or settings.get("model") or "").strip()
    api_key = (settings.get("api_key") or "").strip()
    if not base_url or not model:
        return False
    host_is_local = any(marker in base_url for marker in _LOCAL_HOST_MARKERS)
    return bool(api_key) or host_is_local


def _call_llm(prompt: str, provider: str | None = None) -> str | None:
    """Call the configured LLM and return the response text.

    Uses the existing ``chisurf.core.settings.ai_settings`` infrastructure
    with the same ``requests.post`` pattern as ``agent_panel.py``.

    Parameters
    ----------
    prompt : str
        The prompt to send.
    provider : str, optional
        Provider key override. Uses the default configured provider if None.

    Returns
    -------
    str or None
        Response text, or None on failure.
    """
    import requests

    from chisurf.core.settings.ai_settings import get_api_settings

    if not _llm_available(provider):
        logger.debug("AI triage: no usable LLM provider — skipping LLM step")
        return None

    settings = get_api_settings(provider)
    base_url = (settings.get("base_url") or "").strip().rstrip("/")
    api_key = (settings.get("api_key") or "").strip()
    model = (settings.get("text_model") or settings.get("model") or "").strip()

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": settings.get("temperature", 0.3),
        "max_tokens": settings.get("max_tokens", 1024),
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning("AI triage LLM call failed: %s", exc)
        return None


def _parse_llm_reply(reply: str) -> dict | None:
    """Parse the LLM's JSON reply, with fallback.

    Tries ``json.loads`` first; on failure, attempts to extract a JSON
    block from markdown fences.
    """
    text = reply.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding any {...}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ── Main triage entry point ─────────────────────────────────────


def run_triage(
    db,
    probe_ids: list[int] | None = None,
    status: str = "unverified",
    provider: str | None = None,
    batch_size: int = 10,
) -> list[dict]:
    """Run AI-assisted triage on probes.

    Deterministic checks run for every probe. The LLM step runs only
    when a provider is configured (no provider = deterministic only).

    Parameters
    ----------
    db : FluorophoreDatabase or MFDatabase
        Database instance with probe access.
    probe_ids : list of int, optional
        Specific probe IDs to triage. If None, uses status filter.
    status : str, default='unverified'
        Filter by verification_status when probe_ids is None.
    provider : str, optional
        LLM provider key override.
    batch_size : int, default=10
        Number of probes to process per batch.

    Returns
    -------
    list of dict
        Triage results, one per probe.
    """
    if probe_ids:
        placeholders = ",".join("?" * len(probe_ids))
        rows = db.conn.execute(
            f"SELECT * FROM probes WHERE probe_id IN ({placeholders}) AND deleted_at IS NULL",
            probe_ids,
        ).fetchall()
    else:
        rows = db.conn.execute(
            "SELECT * FROM probes WHERE verification_status = ? AND deleted_at IS NULL ORDER BY probe_id",
            (status,),
        ).fetchall()

    if not rows:
        logger.info("AI triage: no probes to process")
        return []

    llm_ready = _llm_available(provider)
    if not llm_ready:
        logger.info(
            "AI triage: no usable LLM provider configured — running deterministic "
            "checks only (configure a provider in AI Settings to enable LLM triage)"
        )

    results = []
    for i, row in enumerate(rows):
        probe = dict(row)
        props = {}
        try:
            prop_rows = db.conn.execute(
                "SELECT property_name, property_value FROM optical_properties WHERE probe_id = ? AND deleted_at IS NULL",
                (int(probe["probe_id"]),),
            ).fetchall()
            for pr in prop_rows:
                props[str(pr["property_name"])] = str(pr["property_value"])
        except Exception:
            pass

        probe.update(props)
        probe["has_abs"] = bool(
            db.conn.execute(
                "SELECT 1 FROM spectra WHERE probe_id = ? AND spectrum_type = 'absorption' AND deleted_at IS NULL",
                (int(probe["probe_id"]),),
            ).fetchone()
        )
        probe["has_em"] = bool(
            db.conn.execute(
                "SELECT 1 FROM spectra WHERE probe_id = ? AND spectrum_type = 'emission' AND deleted_at IS NULL",
                (int(probe["probe_id"]),),
            ).fetchone()
        )

        # Deterministic checks
        deterministic = run_deterministic_checks(probe)
        result = {
            "probe_id": int(probe["probe_id"]),
            "name": str(probe.get("chromophore_name", probe.get("name", ""))),
            "issues": deterministic["issues"],
            "proposed_quality": deterministic["proposed_quality"],
            "llm_proposal": None,
            "recommendation": "needs_review",
            "rationale": "",
        }

        # LLM step (batched to avoid overwhelming small models)
        if i % batch_size == 0 and deterministic["issues"]:
            logger.info(
                "AI triage: processed %d/%d probes (batch %d)",
                i,
                len(rows),
                i // batch_size + 1,
            )

        if llm_ready and deterministic["proposed_quality"] != "high":
            llm_reply = _call_llm(
                _AI_TRIAGE_PROMPT.format(
                    probe_json=json.dumps(
                        {
                            "name": result["name"],
                            "category": probe.get("category"),
                            "abs_max": probe.get("abs_max"),
                            "em_max": probe.get("em_max"),
                            "qy": probe.get("qy"),
                            "ext_coeff": probe.get("ext_coeff"),
                            "has_absorption": probe["has_abs"],
                            "has_emission": probe["has_em"],
                        },
                        indent=2,
                    ),
                    issues="; ".join(deterministic["issues"]),
                ),
                provider=provider,
            )
            if llm_reply:
                parsed = _parse_llm_reply(llm_reply)
                if parsed:
                    result["llm_proposal"] = parsed
                    result["recommendation"] = parsed.get(
                        "recommendation", "needs_review"
                    )
                    result["rationale"] = parsed.get("rationale", "")
                    result["proposed_category"] = parsed.get("category")
                    result["canonical_name"] = parsed.get("canonical_name")

        results.append(result)

    logger.info(
        "AI triage: completed %d probes (%d with issues)",
        len(results),
        sum(1 for r in results if r["issues"]),
    )
    return results
