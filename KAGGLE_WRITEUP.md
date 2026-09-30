# AMP Challenge 2027 submission writeup — transformer entry

Paste-ready. Everything below is checkable against the repository.

**Team / author:** Braxton Mensah, Indiana University Bloomington, `bsmensah@iu.edu`
**Repository:** https://github.com/braxtonmensah/amp-challenge-2027-entry-lm (public, MIT)
**Entry point:** `uv sync` then `uv run generate`
**Category emphasis:** Optimal Selectivity (safety window HC50/MIC50)

**This is the second of two entries.** The rules permit one entry per sufficiently different model. The
companion entry, https://github.com/braxtonmensah/amp-challenge-2027-entry, generates with an order-2
Markov chain. This one generates with a transformer trained from scratch. They share the selection rule
exactly and share nothing in the generator, which makes the generative model the only variable between
the two submissions.

They are separate repositories because the validator hardcodes `ENTRY_POINT = "generate"` and reads
`<repo>/generate/library.fasta`. A second entry point under any other name inside one repository is never
invoked, so the second entry would silently regenerate the first entry's library.

**Declared because Section 2.3 runs a pairwise overlap analysis across submitted libraries and top-100
lists to detect collusion or duplicate submissions.** Two entries from one author should be checked, so
here are the numbers, measured rather than asserted:

| comparison between the two entries | result |
|---|---|
| identical sequences in the two 50,000-libraries | **2 of 50,000** (0.004%) |
| identical sequences in the two top-100 lists | **0 of 100** |
| highest Levenshtein ratio between any cross-entry top-100 pair | **0.667** |
| cross-entry top-100 pairs at ratio >= 0.8 | **0 of 10,000** |

The two full-library collisions are reported rather than rounded away: two short cationic sequences that
both generators independently reached. At 50,000 draws each over a 20-letter alphabet a handful of
collisions is what genuine independence predicts, and zero would be the more surprising number. An
earlier revision of this table reported **0**, which was true of the companion entry's previous library
and became false when that library was regenerated against its potent-AMP corpus. Corrected here rather
than left to be discovered.

The **method documentation deliberately overlaps**, by design rather than oversight. The selection rule,
the three guards, the identity screen and the retractions are identical by construction in both entries,
precisely so the generative model is the only variable between them. The generation sections, the
libraries and the candidate lists are entirely distinct.

---

## Abstract

A 4-layer decoder-only transformer (0.81M parameters) trained from scratch on the competition's 39,448
reference antibacterials generates a library of 50,000 novel linear peptides. On a held-out split it
reaches perplexity 7.72 against 14.79 for an order-2 Markov chain and 20.00 for uniform random, and its
library sits closer to the reference distribution than the Markov library on FBD and MMD. The top-100 are
selected by the same deliberately minimal scorer used in the companion entry,
`rank(net charge) - rank(mean hydrophobicity)`, chosen by measurement against 2,904 published
MIC-labelled peptides rather than asserted. Two things distinguish this submission: a larger transformer
that scored better perplexity was **trained, measured and rejected** for memorisation, and
reproducibility is verified across two platforms with different torch builds rather than twice on one
machine.

## Method

**Generation.** Decoder-only transformer, 4 layers, width 128, 4 heads, learned positional embeddings,
0.81M parameters. Trained on `data/antibacterial.fasta` only, 5% held out for validation, no pretrained
weights and no other corpus. Sampled autoregressively at temperature 1.0 with no top-k or nucleus
truncation, so the library comes from the model's full distribution rather than a sharpened one. Length
is learned, not imposed: the model emits its own end-of-sequence token. Exact reference matches and
internal duplicates are excluded. 56,800 draws yield 50,000 unique novel sequences, an 88% keep rate.
Trained weights ship in `checkpoint/peptide_lm.pt`.

| density model, same corpus and held-out split | perplexity |
|---|---|
| uniform random over 20 residues | 20.00 |
| order-2 Markov chain (companion entry) | 14.79 |
| **this transformer** | **7.72** |

**A bigger model was trained, measured, and rejected.** Width 256, 6 layers, 4.76M parameters, best
validation perplexity 6.93 — better than the shipped model's 7.72. It was thrown away. Its train loss
fell to 1.21 while validation rose to 2.10, a 0.88 gap with validation degrading after its peak, which is
memorisation; and this competition scores novelty against the very corpus it memorised. Its library
measured a **worse** MMD than even the Markov chain. Bigger lost. Training both sizes is what caught it.

**Selection.** Identical to the companion entry: `rank(net charge) - rank(mean hydrophobicity)`, two
terms, no fitted weights, restricted by three guards — a measured envelope (net charge [-1, +5], mean
Eisenberg hydrophobicity [-0.05, +0.65]), a composition guard at the 95th percentile of the reference
actives with cysteine excluded outright, and an internal pairwise-Levenshtein cap of 0.7 because the 25
assayed peptides are drawn uniformly at random and near-duplicates waste draws.

**Novelty screening.** Candidates must clear 80% identity under three definitions simultaneously
(matches/shorter-length, local alignment at coverage >= 0.8, and full-length global alignment) by BLOSUM62
alignment, because a Levenshtein edit ratio is not sequence identity.

## Reproducibility, checked across platforms rather than twice on one machine

The rules say organizers verify by running `uv sync` and the entry point *"on a Linux workstation with a
single GPU"*, comparing against the submitted library. That is a different machine from the one that
produced the submission, so two identical runs on one machine does not establish it. A transformer adds a
hazard a Markov chain does not have: every sampled token depends on floating-point logits, and
matrix-multiply reduction order is not guaranteed to match across builds or thread counts.

Sampling is therefore pinned to the CPU even when a GPU is present, because CUDA and CPU draw different
random streams. Leaving a GPU idle is the intended trade.

**We measured float32 failing this, and fixed it.** Same seed, same torch version, a Windows `+cpu` build
and a Linux `+cu130` build produced libraries differing in exactly one sequence of 50,000:

    Linux   float32: NLVQFEMQILGQLTINAIENPQPK S QHLQK
    Windows float32: NLVQFEMQILGQLTINAIENPQPK W QHLQR
    both    float64: NLVQFEMQILGQLTINAIENPQPK W QHLQR

| check | precision | result |
|---|---|---|
| Windows `2.14.0+cpu` vs Linux `2.14.0+cu130` | float32 | **DIVERGED, 1 sequence of 50,000** |
| **AMD EPYC 7742 vs Intel Xeon Gold 6248** | **float64** | **byte-identical, library and top** |
| two consecutive runs, same node | float64 | byte-identical |
| thread count 1 vs 8, same machine | both | identical |
| fresh clone of the public repo on Linux, `compileall` + import | - | pass |

Sampling therefore runs in float64, which takes the logit perturbation from ~1e-7 to ~1e-16 and the
expected flips per library from about one to about 1e-10, at roughly 2.4x runtime. The two float64 rows
are on different CPU architectures, AMD Zen 2 and Intel Cascade Lake, which select different matmul
kernels and hence different reduction orders: exactly the condition float32 failed under.

Reaching float64 exposed a latent bug. The causal mask was built by `torch.full` with no dtype, so it
stayed float32 while the model moved to float64. At one sampling position that shifted the output
distribution by 0.14 in probability and made EOS the most likely token where float32 ranked it outside the
top six. With the mask following `x.dtype`, the two precisions agree to 7.7e-08. The shipped float32 path
was never affected because there the dtypes coincide, which is why it went unnoticed.

**What the fix did not buy.** One sequence in 50,000 changed, and every Phase 1 metric below is unchanged
to six decimal places. This is a reproducibility fix, not a quality improvement, and we do not present it
as one.

The last row exists because an earlier push of this entry did **not** parse on a fresh clone: `\n` escapes
in the FASTA writer had been flattened into real newlines, and it went unnoticed locally because the
running process had already imported an earlier copy. Compiling the committed blob, not the working tree,
is now part of the check.

## Phase 1 self-measurement (`seqme`, against a disjoint reference half)

Measured on the files this repository ships, in one run, so all five rows are comparable.

| query set | Uniqueness | Diversity | Novelty | Conformity | FBD | MMD |
|---|---|---|---|---|---|---|
| held-out real AMPs (ceiling) | 1.000 | 0.8530 | 1.0 | 0.4918 | 0.00485 | 0.000502 |
| **this entry's library** (transformer) | 1.000 | **0.8563** | 1.0 | **0.4850** | **0.00555** | **0.000732** |
| **this entry's top-100** | 1.000 | 0.8057 | 1.0 | 0.5798 | 0.0550 | 0.01708 |
| companion entry's library (Markov) | 1.000 | 0.8528 | 1.0 | 0.5759 | 0.00913 | 0.001218 |
| companion entry's top-100 | 1.000 | 0.8255 | 1.0 | 0.5965 | 0.0529 | 0.01118 |

The two entries split the result and neither dominates. At library level the transformer wins three of
four: 39% closer on FBD, 40% better on MMD, and diversity 0.8563 marginally above the real-AMP ceiling of
0.8530. Conformity is the one the Markov chain wins (0.5759 against 0.4850) and the comparison is
genuinely ambiguous, because real AMPs score 0.4918 — the transformer lands *on* the reference value while
the Markov library is more property-conforming than real peptides are, and the published rules do not say
which the aggregation rewards. At top-100 level the Markov entry is better on diversity and MMD; selection
is identical in both entries, so that is a property of the candidate pool rather than the scorer.

Earlier figures circulated for this entry (Diversity 0.858, Conformity 0.490, FBD 0.00568, MMD 0.00097)
are **withdrawn**: they were measured on a development library from a code path that sizes batches
adaptively and uses CUDA when available, not on what the submitted entry point produces.

**Caveat.** `seqme`'s FBD emitted `LinAlgWarning: Matrix is singular` for both 100-sequence sets — a
400-dimensional dipeptide embedding cannot yield a full-rank covariance from 100 points — so the top-100
FBD and MMD values are order-of-magnitude indicators, not precise numbers.

## Two defects in the novelty screen, found by checking the margin rather than the verdict

The validator fails similarity on `ratio > 0.8` and our screen rejected on the same `> 0.8`, a bar of
zero width. The transformer library landed a candidate exactly on it: `VNWKKLFKGVKKIL` against reference
`WKKLFKKLKIL`, 20/25 = **0.800000**, which passed only because two independent floating-point divisions
agreed. Screening at `>=` makes our bar strictly tighter than the one we are judged by; the worst
surviving ratio is 0.7917.

Second, the screen scanned a fixed +/-12 reference-length window. The real bound is `[2/3 la, 3/2 la]` at
ratio 0.8, which is wider than +/-12 for any candidate longer than 36 residues: at length 45 the old code
started at 33 while the bound starts at 30, so a violating reference of length 31 or 32 was never
compared. No such pair exists in the shipped library — the organizers' exhaustive check passes — but the
screen was only accidentally correct.

Fixing both changed 2 of the 100 selected sequences.

## What we measured, and what we retracted

Carried over from the companion entry, because the selection rule is shared and the retractions apply to
both. An earlier version scored candidates with a four-term composite (banded charge 0.40, hydrophobic
moment 0.30, banded hydrophobicity 0.20, helix propensity 0.10). Measured against panel success rate
(MIC <= 16 uM, weighted 15:5 Gram-negative:Gram-positive to match the official panel):

| term | old weight | Spearman | AUC |
|---|---|---|---|
| net charge, raw | - | +0.287 | **0.678** |
| net charge, banded | 0.40 | +0.244 | 0.590 |
| hydrophobic moment | 0.30 | +0.022 | **0.514** |
| hydrophobicity, banded | 0.20 | -0.047 | 0.505 |
| helix propensity | 0.10 | -0.006 | **0.498** |
| the composite | | +0.158 | **0.595** |

The moment and helix terms were indistinguishable from noise and the composite ranked worse than net
charge alone. **We retract our earlier claim** that banding hydrophobicity buys a safety window.

**And we retract a second claim, against prior art.** We had argued the hydrophobicity/activity
relationship is monotone. That is wrong: Chen et al. 2007 (*Antimicrob Agents Chemother* 51:1398-1406)
establishes an optimum hydrophobicity window at constant net charge, and our own data agrees once charge
is held fixed (success 0.551 in our band against 0.665 at hydrophobicity 0.05-0.25, at charge +4 to +5).
Our monotone reading was the charge confound (r = -0.729) surviving into a conclusion. We keep the
low-hydrophobicity selection for a narrower reason: on 501 peptides with paired HC50 and panel MIC,
controlling for charge, hydrophobicity buys potency (partial rho -0.176 on log MIC) but costs haemolysis
about twice as much (-0.366 on log HC50), netting -0.239 on log safety window.

**A trained MIC model was tested and is NOT shipped.** Pre-registered in `PREREG_SELECTION.md`: a
descriptor ridge learned real signal (grouped-CV Spearman +0.361 against a -0.074 shuffled-label null)
but its top-100 beat the biophysical scorer by **+0.002** on measured success rate. The gate failed and we
report that rather than shipping the more sophisticated method. This says nothing about deep *generative*
models, which is why this entry exists.

**Why we target one category and concede four.** Optimal Selectivity ranks on mean HC50/MIC50 and
**excludes peptides inactive on every strain rather than scoring them zero**, so its objective is
E[SW | active] and the dead fraction barely enters; the other four categories average success rate over
all 25, where dead peptides drag the mean. Stated conflict: a larger sample (n=109, charge +4 to +5, MIC
only) favours the old band on success rate, 0.665 against 0.551, and is the more reliable estimate of the
potency question. So we most likely concede some success rate deliberately.

## Training data, external databases, and manual interventions

**Generation** uses one corpus: `data/antibacterial.fasta` as shipped in the template (39,448 sequences).
No pretrained model, no other corpus. Trained weights ship in the repository.

**External public data was used to choose the selection rule** (permitted by the rules): MIC from the
51,345 measurements aggregated in GRAMPA (DBAASP, DRAMP, YADAMP, APD, DADP), filtered to unmodified
free-termini 20-standard-AA 8-50aa peptides on panel species, giving 2,904 labelled sequences; HC50 from
Hemolytik-derived values, 501 sequences with both HC50 and panel MIC; DRAMP 3.0 used only to bound the
novelty reference gap (adds ~9% new sequence over the template). Excluding amidated entries is mandatory
here: this competition forbids terminal modification and amidation shifts MIC severalfold.

**A learned model runs at generation time in this entry** — that is the whole point of it — but no
learned model scores candidates. Selection is a closed-form two-term scorer with hard-coded constants
whose provenance is in `generate.py` comments.

**Manual interventions and filters**, all in `pick_top`: the measured envelope, the composition guard with
cysteine exclusion, the internal diversity cap, and the three-definition identity screen.

## Limitations

1. No experimental validation of anything here; every relationship was measured on published peptides.
2. Applied out of distribution by construction: candidates must sit below 80% identity to known AMPs while
   every fitted relationship comes from known AMPs.
3. The generator is a 0.81M-parameter model trained on 39,448 short sequences. It is a better density
   model of that corpus than a dipeptide chain, which is a low bar, and it is far from the published
   state of the art in AMP generation.
4. Perplexity is not activity. The transformer is measurably the better model of the corpus; nothing here
   demonstrates that this yields more active peptides, and Phase 2 is what would test it.
5. The three-definition identity screen prefilters on `abs(len(r) - len(s)) > 20` and on a Levenshtein
   floor of 0.45. Those prefilters are heuristics, not proven bounds like the one fixed above, so the
   alignment-based screen is best-effort; the Levenshtein screen that the validator actually runs is
   exhaustive and sound.
6. Novelty is screened against the template's 39,448 rather than MarLys, which we could not obtain.
7. We do not claim to beat a null: no published study MIC-tests random or composition-matched peptides at
   this competition's <= 16 uM threshold.

## AI assistance

This repository was written with AI assistance (Anthropic Claude), disclosed as the rules require. The
scientific judgements — rejecting the larger transformer for memorisation, retracting the
hydrophobicity-band rationale, discarding fitted weights, and submitting two models rather than guessing
which the withheld Phase 1 weights reward — are stated explicitly so a reviewer can disagree with them on
the record.
