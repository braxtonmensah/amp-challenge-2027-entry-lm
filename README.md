# AMP Challenge 2027 entry: a transformer language model, with a selection rule chosen by measurement

Braxton Mensah, Indiana University Bloomington (`bsmensah@iu.edu`).

    uv sync
    uv run generate

Writes `generate/library.fasta` (50,000 sequences) and `generate/top.fasta` (100).

---

## This is one of two entries, and what makes it a different model

The rules permit one entry per sufficiently different model: *"If a team has two or more sufficiently
different models, it may submit one entry per model."* This repository generates with a **4-layer
decoder-only transformer trained from scratch** on the competition corpus. The companion entry,
[amp-challenge-2027-entry](https://github.com/braxtonmensah/amp-challenge-2027-entry), generates with an
**order-2 Markov chain**. They share nothing in the generator and everything in the selection rule, which
is deliberate: it isolates the generative model as the only variable between the two submissions.

**Why two repositories rather than two entry points in one.** The organizers' validator hardcodes
`ENTRY_POINT = "generate"` and reads `<repo>/generate/library.fasta`. A second entry point under any
other name is never invoked, so two models in one repository would be regenerated as the same library
twice and the second entry would collapse into a duplicate of the first with no error raised. One
repository per model is the only arrangement the validator can distinguish.

**Which one is better is not a question we can answer, and that is the point.** The Phase 1 aggregation
weights are withheld. The transformer is the better density model by a wide margin and sits closer to the
real-AMP distribution; the Markov library scores higher on property conformity. Rather than guess which
the withheld weights reward, both are submitted.

## Disclosures, up front

**AI assistance.** This repository was written with AI assistance (Anthropic Claude), which the
competition rules permit and require to be disclosed.

**Training data for generation: one corpus only.** `data/antibacterial.fasta` as shipped in the
organizers' template, 39,448 sequences, all 8-50 residues. The transformer is trained from scratch on
that corpus and nothing else: no pretrained weights, no other sequence source, no tokenizer library. The
corpus is used for two things here, training the model and filtering for novelty. Lengths are not drawn
from an empirical distribution in this entry; the model emits its own end-of-sequence token, so the
length distribution is learned rather than imposed.

**External data used to choose the SELECTION RULE.** Public measured MIC and haemolysis values were used
to decide the form and the constants of the scoring function. The competition explicitly permits this
("teams are free to use any public peptide and/or AMP database"). Specifically:

* **MIC**: 51,345 measurements aggregated in GRAMPA from DBAASP, DRAMP, YADAMP, APD and DADP. Filtered
  to unmodified, free-termini, 20-standard-AA, 8-50 aa peptides on panel species, giving **2,904
  labelled sequences**. Filtering out modified and C-terminally amidated entries is not optional: this
  competition forbids terminal modification, and amidation adds about +1 charge and shifts MIC
  severalfold, so training on it would bias every prediction.
* **HC50**: Hemolytik-derived values, **501 sequences** with both HC50 and panel MIC.

**Trained weights DO ship in this entry, and a learned model DOES run at generation time.** This is the
opposite of the companion Markov entry and is stated plainly because the two repositories otherwise read
alike. `checkpoint/peptide_lm.pt` is the trained transformer, 0.81M parameters, and `uv run generate`
samples from it. The external MIC and haemolysis data below was used only to choose the **selection**
rule, which remains a closed-form two-term scorer with hard-coded constants; no learned model scores
candidates. Generation needs none of the external data and reproduces from this repository alone.

**Determinism, and why it is checked across platforms.** One seed, `SEED = 20260930`, and sampling is
pinned to the CPU even when a GPU is present, because CUDA and CPU draw different random streams. The
rules say reproducibility is verified by running `uv sync` and the entry point *"on a Linux workstation
with a single GPU"* and comparing against the submitted library. That is a different machine from the one
that produced the submission, so two identical runs on one machine does not establish it. A transformer
adds a hazard the Markov chain does not have: the sampled token depends on floating-point logits, and
matrix-multiply reduction order can differ between builds. This entry is therefore checked both ways, two
runs on one machine and one run on each of two platforms with different torch builds. See `SEED.md` and
the reproducibility table below.

## Method

### Generation

A **decoder-only transformer trained from scratch** on the reference actives: 4 layers, model width 128,
4 attention heads, learned positional embeddings, 0.81M parameters, trained on `data/antibacterial.fasta`
with a 5% held-out validation split. Sequences are sampled autoregressively at temperature 1.0 with no
top-k or nucleus truncation, so the library is drawn from the model's full distribution rather than a
sharpened one. The library excludes exact matches to the reference set and internal duplicates.

**Why a learned model at all, when the companion entry's Markov chain is deliberately weak.** An order-2
Markov chain conditions each residue on exactly the previous two. It cannot represent periodicity,
long-range charge patterning, or any dependency beyond a dipeptide, and those are the features that
distinguish an amphipathic helix from a random cationic string. The transformer is the honest upgrade, and
it is measurably a better density model of the corpus on a held-out split:

| model | held-out perplexity |
|---|---|
| uniform random over 20 residues | 20.00 |
| order-2 Markov chain (companion entry) | 14.79 |
| **this transformer** | **7.72** |

**Why the small model, when a bigger one scored better perplexity.** A larger variant (width 256, 6
layers, 4.76M parameters) reached a better best validation perplexity, 6.93 against 7.72. It was rejected.
Its train loss fell to 1.21 while validation rose to 2.10, a 0.88 gap with validation degrading after its
peak, which is memorisation; and this competition scores novelty against the very corpus it memorised. Its
library measured a **worse** MMD than even the Markov chain. Bigger lost. Training both sizes is what
caught it, and the rejected run is kept in the record rather than quietly dropped.

**Capacity is a novelty constraint here, not just a fitting choice.** 39,448 sequences of 8-50 residues
over a 20-letter alphabet is a small corpus, and every candidate must sit below 80% identity to all of it.
A model with enough capacity to reproduce its training set is worse than useless for this task, which is
the reason the selection above is made on distribution metrics and the identity screen rather than on
perplexity alone.

### Selection: what we measured, and what we retired

The first version of this entry ranked candidates with a four-term composite: banded net charge (0.40),
hydrophobic moment (0.30), banded hydrophobicity (0.20), Chou-Fasman helix propensity (0.10). **We then
measured each term against real MIC data and retired it.** Against measured panel success rate
(MIC <= 16 uM, weighted 15:5 Gram-negative:Gram-positive to match the official panel):

| term | old weight | Spearman | AUC |
|---|---|---|---|
| net charge, raw | - | +0.287 | **0.678** |
| net charge, banded +4..+9 | 0.40 | +0.244 | 0.590 |
| hydrophobic moment | 0.30 | +0.022 | **0.514** |
| mean hydrophobicity, banded | 0.20 | -0.047 | 0.505 |
| helix propensity | 0.10 | -0.006 | **0.498** |
| **the old composite** | | +0.158 | **0.595** |

The moment and helix terms were **indistinguishable from noise**, banding the charge **destroyed** signal
the raw value carries, and the composite as a whole ranked **worse than net charge alone**.

### A claim from the first version that we retract

The first README called the hydrophobicity band "the single most consequential choice in this entry",
justified by the assertion that hydrophobicity "drives haemolysis about as readily as it drives killing".
We measured it on 501 sequences with paired HC50 and panel MIC. Net charge and hydrophobicity correlate
**-0.729**, so raw correlations are confounded and only the partials mean anything:

| | raw | partial, controlling net charge |
|---|---|---|
| hydrophobicity -> log HC50 | -0.309 | **-0.366** |
| hydrophobicity -> log MIC50 | +0.256 | **-0.176 (sign flips)** |
| hydrophobicity -> log safety window | -0.454 | **-0.239** |

In the pooled data, at fixed charge, hydrophobicity buys potency as the literature says, while costing
HC50 roughly twice as much — so for the safety window the trade is **not worth taking**. Peptides inside
the old band measured a median log safety window of 1.089 against 1.593 outside it (Mann-Whitney
p = 2.8e-10).

#### Retracted: our claim that the relationship is monotone

An earlier version of this section argued that because the relationship is monotone, banding was "the
wrong instrument entirely". **That was wrong, and it is wrong against well-established prior art.**
Chen et al. 2007 (*Antimicrob Agents Chemother* 51:1398-1406) demonstrates, on a congeneric series at
**constant net charge**, an optimum hydrophobicity window for antimicrobial potency: past the optimum,
activity collapses through peptide self-association, and below it peptides go inactive. Haemolysis, by
contrast, is monotone in hydrophobicity. Our own data agrees once charge is properly held fixed — binned
at charge +4 to +5, measured success rate rises from 0.551 in our band to 0.665 at hydrophobicity
0.05-0.25. Our monotone reading was the charge confound (r = -0.729) surviving into a conclusion.

**We keep the low-hydrophobicity selection anyway, for a different and explicit reason.** The five
categories do not score the same way. Optimal Selectivity ranks on mean HC50/MIC50 and **excludes peptides
inactive on every strain rather than scoring them zero**, so the objective there is E[SW | active] and the
fraction of dead peptides barely enters. The other four categories average Success Rate over all 25, where
dead peptides do drag the mean. Measured on the paired subset at charge +3 to +7:

| mean hydrophobicity | n | frac active | E[log SW \| active] | mean success rate |
|---|---|---|---|---|
| -0.20 to -0.05 (**ours**) | 39 | 0.923 | **1.839** | 0.752 |
| -0.05 to 0.10 | 85 | 0.824 | 1.812 | 0.655 |
| 0.10 to 0.30 (old band) | 121 | 0.810 | **1.239** | 0.589 |

So this entry deliberately optimises one category and concedes four. It is not a claim that Chen et al.
are wrong; it is a claim that their optimum is the wrong optimum for the metric we are targeting.

**The conflict in our own numbers, stated rather than hidden.** The larger sample (n=109, charge +4 to +5,
MIC labels only) favours the old band on success rate, 0.665 against 0.551. The smaller paired subset
(n=39) favours ours, 0.752 against 0.589. The larger sample is the more reliable estimate of the potency
question, so we most likely give up some success rate. The safety-window direction is the robust one, and
it is the one we are selecting on.

#### Robustness, and a correction to the sentence above

`src/amp/robustness_sw.py` puts this through four stresses. Three of the four leave it intact and the
fourth forces a correction, so both are reported.

| stress | hydrophobicity -> log SW, controlling charge |
|---|---|
| all 501 paired | -0.239 |
| similarity-cluster half A (n=260) | -0.236 |
| similarity-cluster half B (n=241) | -0.235 |
| also controlling length | -0.233 |
| within DBAASP (n=274) / YADAMP (n=238) / DRAMP (n=219) | -0.245 / -0.303 / -0.223 |

**Parse validated.** The haemolysis file has malformed line endings and is parsed positionally, which is
exactly the kind of thing that invents a result. So `Hemolytik_data.csv` is parsed independently and
strictly, filtered by its *own* columns to Linear / C-ter Free / N-ter Free / Modified None and to
micromolar units only. On the 67 sequences the two parses share: **rank correlation +0.773, Pearson +0.849**
on log10. The parse is not manufacturing the relationship.

**The correction.** The *haemolysis cost* is robust: hydrophobicity -> log HC50 controlling charge sits
between -0.32 and -0.40 in every cluster half and every source database. The *potency benefit* is **not**:
hydrophobicity -> log MIC controlling charge ranges from **-0.006 in YADAMP** to -0.246 in DRAMP. So the
design conclusion ("penalise hydrophobicity") is well supported, but the mechanism as stated above is only
half supported — the cost is real, the benefit is not reliably present in this data. We prefer to leave
the original sentence standing with this correction beneath it rather than quietly rewrite it.

### The scorer that ships

    score = rank(net charge) - rank(mean hydrophobicity)        # within the candidate pool

Two terms, no fitted weights. It was adopted against three gates pre-registered in
`PREREG_SELECTION_2.md` and evaluated on a **cluster-disjoint held-out half** (clusters at Levenshtein
ratio >= 0.6, split so no sequence family appears in both halves):

| scorer, top-100 of held-out half | median log SW | mean success rate |
|---|---|---|
| random 100 | 1.157 | 0.591 |
| the retired composite | 1.288 | 0.615 |
| a fitted 3-descriptor ridge | 1.550 | 0.628 |
| **rank(charge) - rank(hydrophobicity)** | **1.541** | **0.682** |

The fitted ridge beat the simple form by 0.009 log10, inside the pre-registered simplicity margin, so
**the fitted weights were discarded**. A two-term expression beats the four-term composite on both
endpoints at once.

### Three guards, each because an unguarded version failed

Extremising any linear score walks it off the end of the evidence, and each of these was added after
watching that happen, not in anticipation:

1. **Measured envelope.** Selection is restricted to net charge in [-1, +5] and mean hydrophobicity in
   [-0.19, +0.65], the 30th-70th percentile band of the labelled distribution. Unconstrained, the scorer
   selected poly-arginine strings at charge +18 (`RRRRWIRDLAKTMQHPPRRQPKKRRKRRRGCR`) with 44 of 100
   outside any measured range; those are cell-penetrating-peptide motifs, not antimicrobials.
   The percentile was chosen by a stated rule, not by taste: **take the most aggressive envelope whose
   Phase 1 property-conformity is still at least that of real AMPs.** Pushing to the 99th percentile
   maximises predicted safety window (0.825 vs 0.304) but collapses `seqme` conformity from 0.496 to
   0.029 against 0.489 for held-out real AMPs. Qualification precedes any Phase 2 gain.
2. **Composition guard** at the 95th percentile of the reference actives (max single residue <= 0.500,
   W <= 0.238, aromatic FWY <= 0.333, Q <= 0.111), plus **cysteine excluded outright**. Without it the
   top sequence carried seven tryptophans and a run of glutamines, because glutamine's Eisenberg value
   (-0.85) drags mean hydrophobicity down without improving anything biological. Cysteine is excluded
   because a free thiol invites disulfide dimerisation in a linear free-termini peptide.
3. **Internal diversity cap** (pairwise Levenshtein ratio <= 0.7 within the 100). The 25 assayed
   peptides are drawn **uniformly at random** from the top-100, so the ordering of the 100 cannot affect
   any score and near-duplicates simply waste draws. This raised top-100 diversity from 0.760 to 0.825.

### Novelty, and a metric mismatch we found and fixed

The library excludes exact matches to the reference set. The top 100 are additionally screened on
**sequence identity**, not an edit ratio, because those are not the same thing and the difference was
costing us candidates.

The rule is "no more than 80% sequence identity, computed via MMseqs2 pairwise alignment". Identity is
computed over an *alignment*, so two peptides can sit below 0.8 Levenshtein ratio and still align above
80% identity over a well-covered region. Measured on an earlier top-100 that passed the edit-ratio filter:

| identity definition | candidates above 0.80 |
|---|---|
| full-length global alignment | 0 of 100 |
| matches / shorter sequence length | **14 of 100** |
| local alignment, coverage >= 0.8 | **24 of 100** |

Non-compliant candidates are replaced by the organizers with the next valid entry, so that was up to a
quarter of the ranked set silently diluted. Rather than bet on one reading of the rule, every candidate
must now clear 80% under **all three** definitions, by BLOSUM62 alignment (`_identity_ok`). The shipped
top-100 has **zero violations under all three**, with a maximum identity of 0.800.

**Residual gap, stated plainly.** The rule names the MarLys database (~102,000 sequences, thirteen
databases), which we could not obtain; we screen against the template's own 39,448 antibacterials. We
bounded how much that can matter rather than leaving it unquantified: a union with DRAMP 3.0 and GRAMPA
contains 43,025 unique sequences against the template's 39,448, i.e. those two databases add only about
9% of genuinely new sequence. That bounds the gap; it does not close it.

## Phase 1, measured on the organizers' own `seqme`

Computed against a **disjoint half** of the reference actives, so the real-AMP row is not scoring against
itself. Higher is better except FBD and MMD, which are distances.

Measured on the **files this repository ships**, not on a development library. That distinction cost a
correction: earlier figures for this entry (Diversity 0.858, Conformity 0.490, FBD 0.00568, MMD 0.00097)
came from a library produced by a `lm.py sample` development path that sizes batches adaptively and will
use CUDA when a GPU is present. The submitted entry point uses a fixed batch of 512 pinned to the CPU, so
it draws a different stream and is a different library. Those numbers are withdrawn in favour of these.

| query set | Uniqueness | Diversity | Novelty | Conformity | FBD | MMD |
|---|---|---|---|---|---|---|
| held-out **real AMPs** (ceiling) | 1.000 | 0.8530 | 1.0 | 0.4918 | 0.00485 | 0.000502 |
| **this entry's library** (transformer) | 1.000 | **0.8563** | 1.0 | **0.4850** | **0.00555** | **0.000732** |
| **this entry's top-100** | 1.000 | 0.8057 | 1.0 | 0.5798 | 0.0550 | 0.01708 |
| companion entry's library (Markov) | 1.000 | 0.8528 | 1.0 | 0.5759 | 0.00913 | 0.001218 |
| companion entry's top-100 | 1.000 | 0.8255 | 1.0 | 0.5965 | 0.0529 | 0.01118 |

All five rows were measured in one run against the same disjoint reference half, so they are comparable to
each other. Each set is subsampled to 2,000 with its own `Random(SEED)` rather than one shared stream, so
a row's draw does not depend on how many rows precede it; that makes these figures differ in the third
decimal from the companion entry's README, which consumed a shared stream.

**Read honestly, the two entries split the result, and neither dominates.**

* **At library level the transformer wins three of four.** It is 39% closer to the reference distribution
  on FBD (0.00555 against 0.00913) and 40% better on MMD (0.000732 against 0.001218), and its diversity
  0.8563 is marginally *above* the real-AMP ceiling of 0.8530.
* **Conformity is the one the Markov chain wins, and the comparison is ambiguous.** The Markov library
  scores 0.5759, the transformer 0.4850, and held-out real AMPs score 0.4918. So the transformer lands
  essentially *on* the real-AMP value while the Markov library is markedly **more** property-conforming
  than real peptides are. Whether the aggregation rewards matching the reference or exceeding it is not
  knowable from the published rules, and this is the single clearest reason to submit both rather than
  pick one.
* **At top-100 level the Markov entry is better**, on diversity (0.8255 against 0.8057) and MMD (0.0112
  against 0.0171). Selection is identical in both, so this is a property of the candidate pool, not of the
  scorer.

**A caveat on the top-100 rows.** `seqme`'s FBD implementation emitted `LinAlgWarning: Matrix is singular`
for both 100-sequence sets: a 400-dimensional dipeptide embedding cannot give a full-rank covariance from
100 points. The top-100 FBD and MMD figures are therefore order-of-magnitude indicators, not precise
values, and the gap between the two top-100 rows should not be read as finely as the library rows.

## Compliance, checked against the organizers' own validator

`scripts/verify_submission.py` from the template, imported and run directly against the shipped files:

| check | result |
|---|---|
| `_verify_sequences(library.fasta)` | PASS, 50,000 unique |
| `_verify_no_overlap` vs 39,448 references | PASS |
| `_verify_top(top.fasta, k=100)` | PASS |
| `_veritfy_max_simularity(<= 0.80)` | PASS |
| worst top-100 Levenshtein ratio vs any reference | **0.7917** (validator fails above 0.80) |
| identity <= 0.80 under all three alignment definitions | PASS, 0 violations |

Peptide constraints: 20 standard amino acids, 8-50 residues, linear, free termini, no duplicates. The
top-100 additionally contains no cysteine. Length 10-45, median 21. Net charge median +5, range +4 to +5.

### Two defects in the novelty screen, found by checking the margin rather than the verdict

The validator's similarity check fails on `ratio > 0.8`, and our screen rejected on the same `> 0.8`.
That is a bar of zero width, and the transformer library landed a candidate exactly on it:
`VNWKKLFKGVKKIL` against reference `WKKLFKKLKIL` scores 20/25 = **0.800000**. It passed only because two
independent floating-point divisions agreed. Screening at `>=` instead makes our bar strictly tighter
than the one we are judged by, and the worst surviving ratio is now 0.7917.

Second, the screen bucketed references by length and scanned a fixed +/-12 window around each candidate.
The real bound is `[la * r / (2 - r), la * (2 - r) / r]`, i.e. `[2/3 la, 3/2 la]` at r = 0.8, which is
WIDER than +/-12 for any candidate longer than 36 residues: at length 45 the old code started at 33 while
the bound starts at 30, so a violating reference of length 31 or 32 would never have been compared. The
shipped library contains no such pair -- the organizers' exhaustive check passes -- but the screen was
only accidentally correct, and that is not the same as correct.

Fixing both changed 2 of the 100 selected sequences. This is recorded because the first version of the
entry would have passed the automated check and still been wrong.

## Honest limitations

1. **No experimental validation of anything here.** Every relationship used was measured on *published*
   peptides, not on ours.
2. **Everything is applied out of distribution by construction.** Our candidates must sit below 80%
   identity to any known AMP, while every relationship we fitted comes from known AMPs. The
   cluster-held-out evaluation is the closest available estimate of that gap and is still optimistic.
3. **An order-2 Markov chain is a weak generative model.** It captures dipeptide context and nothing
   longer, and cannot represent tertiary or aggregation behaviour.
4. **A trained MIC model did not help and is not shipped.** Pre-registered in `PREREG_SELECTION.md`: a
   descriptor ridge model learned real signal (grouped-CV Spearman +0.361 against a -0.074 shuffled-label
   null) but its top-100 beat the biophysical score by **+0.002** on measured success rate. The primary
   gate failed and we report that rather than shipping the more sophisticated method.
5. **The haemolysis source file has malformed line endings** and was parsed heuristically (1,258
   sequences recovered, 501 joined), with values range-checked to 0 < log10 HC50 < 4. Section "A claim we
   retract" is conditional on that parse.
6. **We do not claim to beat a null.** No published study synthesises random or composition-matched
   peptides and MIC-tests them at this competition's <= 16 uM threshold, so no comparable null exists.
7. Nothing here was screened for protease stability or aggregation.

## Repository map

* `src/amp/generate_lm.py` - **the entry point** (`uv run generate`). Sampling from the trained model.
* `src/amp/lm.py` - the model, the training loop, and the corpus encoding.
* `checkpoint/peptide_lm.pt` - the trained weights this entry samples from, 0.81M parameters.
* `hpc/train_lm.sbatch` - the job that trained them, so the checkpoint is not an unexplained artifact.
* `src/amp/generate.py` - NOT the entry point here. Retained because it holds the shipped scorer, the
  three guards and the identity screen, which this entry imports unchanged from the companion entry.
* `PREREG_SELECTION.md` - pre-registration 1: should a trained MIC model replace the scorer? Gate failed.
* `PREREG_SELECTION_2.md` - pre-registration 2: are the scorer's own terms carrying signal? Gates passed.
* `src/amp/prep_labels.py` - builds panel-matched labels from GRAMPA.
* `src/amp/eval_selection.py`, `src/amp/eval_scorer.py` - the gate runners.
* `src/amp/predictor.py` - the retired predictor harness, kept for audit.

## Licence

MIT, see `LICENSE`.
