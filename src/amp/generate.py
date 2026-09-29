"""AMP Challenge 2027 entry: order-2 Markov generation over known antibacterials, ranked by the
classic biophysical determinants of antimicrobial activity and of the safety window.

DESIGN RATIONALE, stated because the method must be disclosed.

The organizers' example generator samples random K/P strings. That satisfies the schema and nothing
else. The two things that actually determine whether a cationic antimicrobial peptide kills bacteria
are well established and cheap to compute:

  1. NET POSITIVE CHARGE, which drives association with the anionic bacterial membrane. The reference
     set of 39,448 known antibacterials is 20.6% K+R, against ~11% in an average proteome.
  2. AMPHIPATHICITY, the segregation of hydrophobic and polar faces once helical. Measured here as the
     Eisenberg hydrophobic moment at 100 degrees per residue.

The third consideration is the one that decides the safety window HC50/MIC50 that Phase 2 scores:
excessive raw hydrophobicity drives haemolysis as readily as it drives potency. So hydrophobicity is
scored toward a target band rather than maximised, which is a deliberate trade of predicted potency
for a predicted safety margin.

Sequence realism comes from an order-2 Markov chain fit to the reference actives, so local motifs
(KKIL, GKII, and similar) appear at their natural frequency instead of being assembled from
independent draws. Lengths are drawn from the reference length distribution.

NOVELTY. The library is filtered against exact matches to the reference set. The top 100 are
additionally required to sit at Levenshtein ratio <= 0.8 from every reference sequence, which is the
organizers' own novelty rule, so the ranked set cannot be a paraphrase of a known peptide.

DETERMINISM. One rng, seeded from SEED. Every collection is sorted before iteration. See SEED.md.

TRAINING DATA. Only `data/antibacterial.fasta` as shipped in the organizers' template repo
(39,448 sequences, all 8-50 aa). No other corpus, no pretrained model, no external database.

AI ASSISTANCE. This code was written with AI assistance (Claude), as the competition rules permit and
require to be disclosed.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SEED = 20260930
AA = "ACDEFGHIKLMNPQRSTVWY"

# Eisenberg consensus hydrophobicity scale.
EISENBERG = {
    "A": 0.62, "C": 0.29, "D": -0.90, "E": -0.74, "F": 1.19, "G": 0.48, "H": -0.40,
    "I": 1.38, "K": -1.50, "L": 1.06, "M": 0.64, "N": -0.78, "P": 0.12, "Q": -0.85,
    "R": -2.53, "S": -0.18, "T": -0.05, "V": 1.08, "W": 0.81, "Y": 0.26,
}
# Chou-Fasman helix propensity, higher favours helix.
HELIX = {
    "A": 1.42, "C": 0.70, "D": 1.01, "E": 1.51, "F": 1.13, "G": 0.57, "H": 1.00,
    "I": 1.08, "K": 1.16, "L": 1.21, "M": 1.45, "N": 0.67, "P": 0.57, "Q": 1.11,
    "R": 0.98, "S": 0.77, "T": 0.83, "V": 1.06, "W": 1.08, "Y": 0.69,
}
CHARGED_POS = {"K": 1.0, "R": 1.0, "H": 0.1}   # H mostly neutral at pH 7.4
CHARGED_NEG = {"D": -1.0, "E": -1.0}

# Target band for mean hydrophobicity. Above this, haemolysis risk rises faster than potency.
H_TARGET_LO, H_TARGET_HI = 0.05, 0.45
CHARGE_TARGET_LO, CHARGE_TARGET_HI = 4.0, 9.0


def read_fasta(path):
    seqs, cur = [], None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                cur = None
            elif line:
                seqs.append(line.upper())
    return [s for s in seqs if s and all(c in AA for c in s)]


def fit_markov(seqs, order=2):
    """-> (start_counts, trans) as plain dicts of Counters, order-2 over amino acids."""
    starts = Counter()
    trans = defaultdict(Counter)
    for s in seqs:
        if len(s) < order + 1:
            continue
        starts[s[:order]] += 1
        for i in range(len(s) - order):
            trans[s[i:i + order]][s[i + order]] += 1
    return starts, trans


def _probs(counter, rng_keys):
    ks = sorted(counter)
    tot = float(sum(counter[k] for k in ks))
    return ks, np.array([counter[k] / tot for k in ks])


def net_charge(s):
    return sum(CHARGED_POS.get(c, 0.0) for c in s) + sum(CHARGED_NEG.get(c, 0.0) for c in s)


def mean_hydrophobicity(s):
    return sum(EISENBERG[c] for c in s) / len(s)


def hydrophobic_moment(s, deg_per_res=100.0):
    """Eisenberg hydrophobic moment, normalised per residue. Higher = more amphipathic."""
    ang = np.deg2rad(deg_per_res) * np.arange(len(s))
    h = np.array([EISENBERG[c] for c in s])
    return float(np.hypot((h * np.sin(ang)).sum(), (h * np.cos(ang)).sum()) / len(s))


def helix_propensity(s):
    return sum(HELIX[c] for c in s) / len(s)


def _band(x, lo, hi):
    """1.0 inside [lo, hi], decaying outside. Rewards a target band, not a maximum."""
    if x < lo:
        return max(0.0, 1.0 - (lo - x) / max(1e-9, abs(lo) + 1.0))
    if x > hi:
        return max(0.0, 1.0 - (x - hi) / max(1e-9, abs(hi) + 1.0))
    return 1.0


def score_one(s):
    """SUPERSEDED, kept only so the retired scorer stays auditable. Do not select with this.

    Measured on 2,904 panel-matched unmodified sequences with published MIC values, this composite ranks
    WORSE than raw net charge alone (AUC 0.595 vs 0.678). Its moment term (weight 0.30, AUC 0.514) and
    helix term (weight 0.10, AUC 0.498) are indistinguishable from noise, and banding the charge throws
    away signal the raw value carries. See PREREG_SELECTION_2.md section 1.
    """
    q = net_charge(s)
    h = mean_hydrophobicity(s)
    mu = hydrophobic_moment(s)
    hel = helix_propensity(s)
    return (0.40 * _band(q, CHARGE_TARGET_LO, CHARGE_TARGET_HI)
            + 0.30 * min(1.0, mu / 0.55)
            + 0.20 * _band(h, H_TARGET_LO, H_TARGET_HI)
            + 0.10 * min(1.0, (hel - 0.85) / 0.35))


def _zrank(v):
    """Rank-standardise within the candidate pool. Deterministic given the pool and its order."""
    v = np.asarray(v, dtype=float)
    order = np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)
    return (order - order.mean()) / (order.std() + 1e-12)


def score_pool(sequences):
    """The scorer actually used: rank(net charge) - rank(mean hydrophobicity). Higher is better.

    Two terms, no fitted weights, and every element of it was measured before being adopted. It was
    chosen against three pre-registered gates on a cluster-disjoint held-out half (PREREG_SELECTION_2.md):

      * safety window HC50/MIC50 of the selected 100: median log SW 1.541 vs 1.229 for the retired
        composite, i.e. about a 1.8x wider window;
      * measured panel success rate: 0.682 vs 0.615, so the window is not bought with potency;
      * a fitted three-descriptor ridge beat this by 0.009 log10, inside the pre-registered simplicity
        margin, so the fitted weights were discarded in favour of this.

    WHY HYDROPHOBICITY IS SUBTRACTED RATHER THAN BANDED. Measured on 501 sequences with paired HC50 and
    panel MIC, controlling for net charge (the two correlate -0.729, so raw correlations are confounded):
    hydrophobicity costs haemolysis substantially (partial rho -0.366 on log HC50), netting -0.239 on log
    safety window. That direction survives four robustness stresses (see robustness_sw.py).

    NOT because the relationship is monotone. An earlier version of this docstring said so and that was
    wrong: Chen et al. 2007 (Antimicrob Agents Chemother 51:1398-1406) establishes an optimum
    hydrophobicity WINDOW for potency at constant net charge, and our own data agrees once charge is held
    fixed - binned at charge +4 to +5, measured success rate rises from 0.551 in our band to 0.665 at
    hydrophobicity 0.05-0.25. The monotone reading was the charge confound surviving into a conclusion.

    The selection stays low-hydrophobicity for a narrower, explicit reason: the Optimal Selectivity
    category ranks on mean HC50/MIC50 and EXCLUDES peptides inactive on every strain rather than scoring
    them zero, so its objective is E[SW | active] and the dead fraction barely enters. On the paired subset
    at charge +3 to +7 our band gives E[log SW | active] 1.839 against 1.239 for the old band, with the
    highest active fraction (0.923) of any bin. This entry therefore optimises one category and concedes
    the four that average Success Rate over all 25 peptides. That is a choice, not an oversight.
    """
    qs = [net_charge(s) for s in sequences]
    hs = [mean_hydrophobicity(s) for s in sequences]
    return _zrank(qs) - _zrank(hs)


def generate(n_sequences, ref_path, seed=SEED):
    refs = read_fasta(ref_path)
    refset = set(refs)
    lengths = np.array([len(s) for s in refs])
    starts, trans = fit_markov(refs, order=2)
    skeys, sprob = _probs(starts, None)
    tcache = {k: _probs(v, None) for k, v in sorted(trans.items())}

    rng = np.random.default_rng(seed)
    out, seen = [], set()
    tries = 0
    max_tries = n_sequences * 60
    while len(out) < n_sequences and tries < max_tries:
        tries += 1
        L = int(lengths[rng.integers(len(lengths))])
        s = skeys[rng.choice(len(skeys), p=sprob)]
        while len(s) < L:
            key = s[-2:]
            if key not in tcache:
                break
            ks, ps = tcache[key]
            s += ks[rng.choice(len(ks), p=ps)]
        if len(s) < 8 or len(s) > 50:
            continue
        if s in seen or s in refset:
            continue
        seen.add(s)
        out.append(s)
    if len(out) < n_sequences:
        raise RuntimeError("only generated %d of %d unique novel sequences in %d tries"
                           % (len(out), n_sequences, tries))
    return out, refs


_ALIGNERS = {}


def _aligners():
    """BLOSUM62 local and global aligners, built once. Deterministic."""
    if not _ALIGNERS:
        from Bio import Align
        from Bio.Align import substitution_matrices
        m = substitution_matrices.load("BLOSUM62")
        for mode in ("local", "global"):
            a = Align.PairwiseAligner()
            a.substitution_matrix = m
            a.open_gap_score = -11
            a.extend_gap_score = -1
            a.mode = mode
            _ALIGNERS[mode] = a
    return _ALIGNERS["local"], _ALIGNERS["global"]


def _identity_ok(s, refs, threshold=0.8, prefilter=0.45):
    """True when `s` is below `threshold` SEQUENCE IDENTITY to every reference, on all three readings.

    WHY THIS EXISTS. The competition's rule is "no more than 80% sequence identity, computed via MMseqs2
    pairwise alignment". A Levenshtein edit ratio is NOT that: identity is computed over an alignment, so
    two sequences can sit below 0.8 edit ratio while still aligning at >80% identity over a well-covered
    region. Measured on an earlier top-100 that passed the edit-ratio filter: 0 of 100 exceeded 0.80 by
    full-length global identity, but 14 of 100 exceeded it by matches/shorter-length and 24 of 100 by
    local identity at coverage >= 0.8. Non-compliant candidates are replaced by the organizers with the
    next valid entry, so that was up to a quarter of the ranked set silently diluted.

    Rather than bet on one reading of the rule, a candidate must clear the threshold under ALL of them.

    One caveat, measured and disclosed: the reference here is the template's own 39,448 antibacterials.
    The rule names the MarLys database (~102,000 sequences, thirteen databases), which we could not
    obtain. Building a union with DRAMP 3.0 and GRAMPA added only ~9% of new sequences (43,025 unique
    against 39,448), which bounds how much that gap can matter, but it does not close it.
    """
    import Levenshtein
    loc, glo = _aligners()
    for r in refs:
        if abs(len(r) - len(s)) > 20:
            continue
        if Levenshtein.ratio(s, r) < prefilter:
            continue
        try:
            A = loc.align(s, r)[0]
        except Exception:
            continue
        ta, tb = A[0], A[1]
        L = len(ta)
        if not L:
            continue
        m = sum(1 for x, y in zip(ta, tb) if x == y and x != "-")
        shorter = float(min(len(s), len(r)))
        if m / shorter > threshold:
            return False
        if L / shorter >= 0.8 and m / float(L) > threshold:
            return False
        try:
            G = glo.align(s, r)[0]
        except Exception:
            continue
        ga, gb = G[0], G[1]
        if len(ga) and sum(1 for x, y in zip(ga, gb) if x == y and x != "-") / float(len(ga)) > threshold:
            return False
    return True


def pick_top(sequences, refs, k, novelty=0.8, max_internal=0.7):
    """Best-scoring k that clear the novelty bar AND are not near-duplicates of each other.

    The internal-diversity cap (`max_internal`) is new and is a deliberate, unfitted choice. Two reasons:

      * The competition draws the 25 assayed peptides UNIFORMLY AT RANDOM from the top-100, so the order
        of the 100 cannot affect any score and a set of near-duplicates wastes draws on one idea.
      * Measured on the organizers' own Phase 1 scorer (seqme), the previous top-100 scored 0.760 on
        Diversity against 0.850 for the library it came from, with FBD 6.7x and MMD 52x worse. The
        qualification ranking uses the library AND the top candidate list, so a narrow set costs twice.
    """
    import Levenshtein

    # MEASURED-ENVELOPE CONSTRAINT. score_pool is an unbounded linear score, so maximising it over
    # 50,000 candidates walks straight off the end of the evidence: the first unconstrained run selected
    # poly-arginine strings at charge +18 (e.g. RRRRWIRDLAKTMQHPPRRQPKKRRKRRRGCR), with 44 of 100 outside
    # the range where the charge/hydrophobicity relationship was ever measured. Those are
    # cell-penetrating-peptide motifs, not antimicrobials, and nothing in the data speaks to them.
    # Bounds are the 1st-99th percentile envelope of the 2,904 labelled panel sequences the scorer was
    # validated on. Selection happens INSIDE the evidence, not past its edge.
    # Set at the 30th-70th percentile of the labelled distribution, not the 1st-99th, and that choice
    # was measured rather than guessed. Pushing to the 99th percentile maximises predicted safety window
    # (pred log SW 0.825 vs 0.304 here) but collapses the organizers' own Phase 1 property-conformity
    # metric from 0.496 to 0.029, against 0.489 for held-out real AMPs. Qualification for the assay comes
    # before any Phase 2 gain, so the rule applied was: take the most aggressive envelope whose seqme
    # conformity is still at least that of real AMPs. That is this one. It keeps +0.247 log10 of the
    # +0.262 measured safety-window improvement and also raises top-100 diversity (0.823 vs 0.760).
    ENV_CHARGE_LO, ENV_CHARGE_HI = -1.0, 5.0
    ENV_HYDRO_LO, ENV_HYDRO_HI = -0.05, 0.65
    # The two axes were later decoupled and re-measured, because tuning them with one shared percentile
    # had conflated them. Two results. (a) Gram-negative activity rises monotonically with net charge in
    # the labelled data (success rate 0.606 at charge 4-6, 0.770 at 6-8, 0.812 at 8-10) and the panel is
    # 15/20 Gram-negative, so a higher charge cap looks attractive - but EVERY variant that lifts median
    # charge above +5 collapses seqme property conformity (0.303 at cap +7, 0.197 at +9, against 0.489 for
    # real AMPs). Real AMPs have median charge near +4, so selecting for +7 makes the set distributionally
    # unlike real AMPs by construction. That conflict is structural and the charge cap stays at +5.
    # (b) The hydrophobicity floor moved from -0.19 to -0.05, which measured better on all three Phase 1
    # metrics (conformity 0.592 vs 0.536, diversity 0.825 vs 0.819, MMD 0.0112 vs 0.0166) at essentially
    # unchanged safety window (E[log SW | active] 1.812 vs 1.839, a 6% difference). Tuning stopped here
    # deliberately: these constants are now two steps of selection against a proxy for the Phase 1 scorer,
    # and a third would be fitting to the metric rather than to the biology.


    # COMPOSITION GUARD, caps set at the 95th percentile of the reference antibacterials themselves, so
    # the selected set stays inside the composition space where 95% of real AMPs live. This exists because
    # extremising any linear score invites composition tricks: an earlier unconstrained run put 7
    # tryptophans and a run of glutamines in the top sequence, since glutamine's Eisenberg value (-0.85)
    # drags mean hydrophobicity down without making a better antimicrobial.
    #
    # Cysteine is excluded outright. The competition requires linear peptides with free termini, and a
    # free thiol invites disulfide dimerisation during synthesis and QC; Phase 1 also scores
    # "empirically derived synthesizability constraints", so this is measured on both sides.
    CAP_MAX_SINGLE, CAP_W, CAP_AROMATIC, CAP_Q = 0.500, 0.238, 0.333, 0.111

    def _frac(s, aas):
        return sum(s.count(c) for c in aas) / float(len(s))

    def in_envelope(s):
        q = net_charge(s)
        h = mean_hydrophobicity(s)
        if not (ENV_CHARGE_LO <= q <= ENV_CHARGE_HI and ENV_HYDRO_LO <= h <= ENV_HYDRO_HI):
            return False
        if "C" in s:
            return False
        if max(s.count(c) for c in set(s)) / float(len(s)) > CAP_MAX_SINGLE:
            return False
        return (_frac(s, "W") <= CAP_W and _frac(s, "FWY") <= CAP_AROMATIC
                and _frac(s, "Q") <= CAP_Q)

    eligible = [s for s in sequences if in_envelope(s)]
    if len(eligible) < k:
        raise RuntimeError("only %d of %d candidates fall inside the measured envelope"
                           % (len(eligible), len(sequences)))
    scores = score_pool(eligible)
    ranked = [s for _sc, s in sorted(zip(-scores, eligible), key=lambda t: (t[0], t[1]))]
    # bucket references by length so the novelty scan is not 39k comparisons per candidate
    bylen = defaultdict(list)
    for r in refs:
        bylen[len(r)].append(r)
    top = []
    for s in ranked:
        if len(top) >= k:
            break
        ok = True
        # Which reference lengths can possibly violate the bar. Levenshtein.ratio is
        # (la + lb - dist) / (la + lb) with substitutions costing 2, so the most a pair of lengths can
        # score is 2 * min(la, lb) / (la + lb). Requiring that bound to reach `novelty` gives
        #     lb >= la * novelty / (2 - novelty)   and   lb <= la * (2 - novelty) / novelty,
        # i.e. [2/3 la, 3/2 la] at novelty = 0.8. The previous version scanned a fixed +/-12 window,
        # which is NARROWER than that bound for any candidate longer than 36 residues: at la = 45 it
        # scanned from 33 while the bound starts at 30, so a violating reference at length 31 or 32
        # would never have been compared. The shipped top-100 happens to contain no such pair -- the
        # organizers' own exhaustive check passes on it -- but the screen was not sound, and a screen
        # that is only accidentally correct is not a screen.
        la = len(s)
        lo = int(la * novelty / (2.0 - novelty))          # floor is the safe direction here
        hi = int(la * (2.0 - novelty) / novelty) + 1      # +1 so the bound is inclusive
        for lr in range(lo, hi + 1):
            if not ok:
                break
            for r in bylen.get(lr, ()):
                # `>=`, not `>`. The validator fails on `> 0.8`, so a candidate sitting at exactly
                # 0.800000 passes by a margin of zero and survives only because two independent
                # floating-point divisions agree. The transformer library produced exactly such a
                # candidate (VNWKKLFKGVKKIL against WKKLFKKLKIL, 20/25 = 0.8 exactly). Screening at
                # `>=` makes our bar strictly tighter than the one we are judged by, which is the
                # only side of that boundary worth being on.
                if Levenshtein.ratio(s, r) >= novelty:
                    ok = False
                    break
        if ok and not _identity_ok(s, refs, novelty):
            ok = False
        if ok and max_internal is not None:
            for t in top:
                if Levenshtein.ratio(s, t) > max_internal:
                    ok = False
                    break
        if ok:
            top.append(s)
    if len(top) < k:
        raise RuntimeError("only %d of %d candidates cleared the novelty and diversity bars "
                           "(loosen max_internal)" % (len(top), k))
    return top


def _write_fasta(seqs, path):
    with open(path, "w", newline="\n") as fh:
        for i, s in enumerate(seqs, start=1):
            fh.write(">seq%d\n%s\n" % (i, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sequences", type=int, default=50_000)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--ref", default="./data/antibacterial.fasta")
    a = ap.parse_args()

    out_dir = Path(Path(sys.argv[0]).stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs, refs = generate(a.n_sequences, a.ref, seed=a.seed)
    _write_fasta(seqs, out_dir / "library.fasta")
    print("library: %d sequences -> %s" % (len(seqs), out_dir / "library.fasta"))

    top = pick_top(seqs, refs, a.top_k)
    _write_fasta(top, out_dir / "top.fasta")
    print("top:     %d sequences -> %s" % (len(top), out_dir / "top.fasta"))
    qs = [net_charge(t) for t in top]
    hs = [mean_hydrophobicity(t) for t in top]
    print("top-100 net charge   median %+.1f  range %+.1f to %+.1f"
          % (sorted(qs)[len(qs) // 2], min(qs), max(qs)))
    print("top-100 hydrophobic. median %+.2f  range %+.2f to %+.2f"
          % (sorted(hs)[len(hs) // 2], min(hs), max(hs)))
    print("top-1 %s" % top[0])


if __name__ == "__main__":
    main()
