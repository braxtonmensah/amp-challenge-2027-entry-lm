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

    library.fasta  b1c356a20752ce789127a57ed23b2762e599f64759884223e44671327f85b34c
    top.fasta      212120c70d723043cefe80037445c6586ceb4516409682fe9d73b875744a23a2

Budget about 14 minutes on 8 Linux cores, roughly four times that on Windows. Sampling draws ~56,800
sequences to keep 50,000 after the length, duplicate and reference-overlap filters.

## Why a transformer needs this checked across machines, not just twice on one

A Markov chain samples from a table of counts; its only floating-point sensitivity is a cumulative sum.
This entry samples each token from logits produced by four transformer layers, and matrix-multiply
reduction order is not guaranteed identical across builds, thread counts or CPU architectures. A single
flipped comparison changes that token and every token after it in that sequence. So "two runs agree on
my machine" is not the claim that matters; the claim that matters is that a different machine agrees.

Measured, not assumed:

| check | result |
|---|---|
| two consecutive runs, same Linux node, 8 cores | byte-identical, library and top |
| thread count 1 vs 8, same machine | identical |
| Windows `torch 2.14.0+cpu` vs Linux `torch 2.14.0+cu130`, same seed | identical keep counts at every checkpoint |
| fresh clone of this public repo on Linux, regenerated and compared to the committed files | see the README |

The shipped `library.fasta` is the file a Linux run produced, so the submitted bytes are the bytes a
Linux re-run reproduces rather than bytes that merely ought to match.
