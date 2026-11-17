"""Membership-overlap stable cluster identity (carry IDs across runs).

HDBSCAN labels carry no cross-run meaning and the persist path is a generation
swap (insert the new generation, then delete the prior one; every cluster mints
a fresh ObjectId `_id` each run). To give multi-member
clusters a *stable* identity that survives the wipe, we match each NEW
multi-member cluster against the PREVIOUS generation by shared member articles: a
new cluster that overlaps a previous one strongly enough inherits that previous
cluster's `stableClusterId`; otherwise it mints a new UUID. Splits/merges resolve
naturally — the highest-overlap pairing wins under a greedy one-to-one match and
the losers mint (or die out).

Singletons (size 1 noise) get NO stableClusterId — their identity is the article
id itself.

This module is PURE compute (no Mongo, no IO): the caller reads the previous
generation, feeds it in, and writes the results back. That keeps the matcher
trivially testable and keeps the read strictly before the persist wipe.
"""
from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass


@dataclass
class StableIdMatch:
    """Result of `match_stable_ids`.

    - `stable_ids`: one stableClusterId per input cluster, aligned to the input
      order (carried from the previous generation, or a freshly minted UUID).
    - `carried`: how many clusters inherited a previous stableClusterId.
    - `minted`: how many clusters minted a fresh UUID.
    """

    stable_ids: list[str]
    carried: int
    minted: int


def match_stable_ids(
    new_clusters: list[list],
    prev_membership: dict[str, str],
    prev_stable_ids: dict[str, str],
    *,
    min_shared: int,
    min_overlap_ratio: float,
) -> StableIdMatch:
    """Assign a stableClusterId to each NEW multi-member cluster.

    Args:
      new_clusters: member articleIds per NEW multi-member cluster (one list per
        cluster; ids may be ObjectId or str — compared as str internally).
      prev_membership: { str(articleId) -> str(prevClusterId) } from the previous
        generation's multi-member clusters (built from cluster-article-link rows).
      prev_stable_ids: { str(prevClusterId) -> stableClusterId }.
      min_shared: minimum shared member count to consider a carry (STABLE_MIN_SHARED).
      min_overlap_ratio: shared count must also be >= this fraction of the SMALLER
        of the two clusters' sizes (STABLE_MIN_OVERLAP_RATIO).

    A carry is accepted when `shared >= min_shared` AND
    `shared >= min_overlap_ratio * min(|new|, |prev|)` AND neither the new cluster
    nor the previous cluster is already taken. Matching is greedy one-to-one:
    candidate (new, prev) pairs are ranked by shared count desc, then overlap ratio
    desc, then new index and prev id for deterministic tie-breaks. Unmatched new
    clusters mint `str(uuid.uuid4())`.
    """
    # Previous cluster sizes = number of links carrying each prevClusterId. Equal
    # to the stored clusterSize (both derive from the same membership).
    prev_sizes: dict[str, int] = Counter(prev_membership.values())

    # Build candidate (new_idx, prev_id) pairs with their shared-member overlap.
    candidates: list[tuple[int, float, int, str]] = []  # (shared, ratio, new_idx, prev_id)
    for new_idx, members in enumerate(new_clusters):
        overlap: Counter = Counter()
        for article_id in members:
            prev_id = prev_membership.get(str(article_id))
            if prev_id is not None:
                overlap[prev_id] += 1
        new_size = len(members)
        for prev_id, shared in overlap.items():
            denom = min(new_size, prev_sizes[prev_id])
            ratio = shared / denom if denom else 0.0
            candidates.append((shared, ratio, new_idx, prev_id))

    # Highest overlap wins; deterministic tie-breaks so a given input always yields
    # the same pairing (uuid minting aside).
    candidates.sort(key=lambda c: (-c[0], -c[1], c[2], c[3]))

    assigned: dict[int, str] = {}
    taken_new: set[int] = set()
    taken_prev: set[str] = set()
    for shared, _ratio, new_idx, prev_id in candidates:
        if new_idx in taken_new or prev_id in taken_prev:
            continue
        if shared < min_shared:
            continue
        denom = min(len(new_clusters[new_idx]), prev_sizes[prev_id])
        if shared < min_overlap_ratio * denom:
            continue
        stable = prev_stable_ids.get(prev_id)
        if stable is None:
            continue
        assigned[new_idx] = stable
        taken_new.add(new_idx)
        taken_prev.add(prev_id)

    stable_ids: list[str] = []
    minted = 0
    for new_idx in range(len(new_clusters)):
        sid = assigned.get(new_idx)
        if sid is None:
            sid = str(uuid.uuid4())
            minted += 1
        stable_ids.append(sid)

    return StableIdMatch(stable_ids=stable_ids, carried=len(assigned), minted=minted)
