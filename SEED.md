# Determinism

One constant, `SEED = 20260930`, defined in `src/amp/generate_lm.py` and passed to `torch.manual_seed`
before sampling. No other source of randomness: no `random.seed()` elsewhere, no unseeded
`numpy.random`, no time- or PID-derived entropy, and no set or dict iteration order used to influence
output.

**Sampling is pinned to the CPU even when a GPU is present.** CUDA and CPU draw from different random
streams, so a GPU run would not reproduce a CPU run whatever the seed. The rules say reproducibility is
verified on "a Linux workstation with a single GPU"; this entry point ignores that GPU on purpose.

## Verify it

    uv sync
    uv run generate
    # regenerates generate/library.fasta and generate/top.fasta in place

Then compare against what the repository ships. Take the committed copies from git rather than the
working tree, since the run you just did has overwritten the working tree:

    git show HEAD:generate/library.fasta | sha256sum
    git show HEAD:generate/top.fasta     | sha256sum
    sha256sum generate/library.fasta generate/top.fasta

Expected, for the commit that ships these files:

    library.fasta  f43780fc41bab39c3c50134fd83a1b8007fb3f7d9d17245f4d860e6af1aa1204
    top.fasta      212120c70d723043cefe80037445c6586ceb4516409682fe9d73b875744a23a2

Budget about 35 minutes on 8 Linux cores in float64, roughly four times that on Windows. Sampling draws
~56,800 sequences to keep 50,000 after the length, duplicate and reference-overlap filters.

## Why a transformer needs this checked across machines, not just twice on one

A Markov chain samples from a table of counts; its only floating-point sensitivity is a cumulative sum.
This entry samples each token from logits produced by four transformer layers, and matrix-multiply
reduction order is not guaranteed identical across builds, thread counts or CPU architectures. A single
flipped comparison changes that token and every token after it in that sequence. So "two runs agree on
my machine" is not the claim that matters; the claim that matters is that a different machine agrees.

Measured, not assumed. The float32 row is the reason this entry samples in float64:

| check | precision | result |
|---|---|---|
| Windows `2.14.0+cpu` vs Linux `2.14.0+cu130`, same seed | float32 | **DIVERGED, 1 sequence of 50,000** |
| AMD EPYC 7742 vs Intel Xeon Gold 6248 | float64 | **byte-identical**, library and top |
| two consecutive runs, same node | float64 | byte-identical |
| thread count 1 vs 8, same machine | both | identical |
| fresh clone of this public repo on Linux, compiled and imported | - | pass |

The float32 divergence was a single sampled token flipping 25 residues into one sequence:

    Linux   float32: NLVQFEMQILGQLTINAIENPQPK S QHLQK
    Windows float32: NLVQFEMQILGQLTINAIENPQPK W QHLQR
    both    float64: NLVQFEMQILGQLTINAIENPQPK W QHLQR

So float64 shows the Linux float32 run was the one in error, not Windows. **The two float64 runs above
are on genuinely different CPU architectures**, an AMD Zen 2 and an Intel Cascade Lake, which select
different matrix-multiply kernels and therefore different reduction orders. That is the condition under
which float32 failed, and float64 survives it.

One honest note on what the fix bought. Switching to float64 changed exactly one sequence in 50,000 and
left every Phase 1 metric unchanged to six decimal places (diversity 0.856294, conformity 0.484954, FBD
0.005550, MMD 0.000732 before and after). It bought reproducibility, not performance, and it is not
presented as an improvement in library quality.

The shipped `library.fasta` is the file a Linux float64 run produced, so the submitted bytes are the
bytes a re-run reproduces rather than bytes that merely ought to match.
