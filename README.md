# Beyond Pocket

Code and results for *One Mistake, Three Layers: Persistence Rules Under
Depth Correlation*, submitted to RECOMB 2027.

The study probes pocket information at all 48 Pairformer blocks of a
frozen Boltz-1 trunk, stratifying 1,820 post-cutoff PLINDER targets by
protein-family familiarity and pocket familiarity. It contributes a
calibrated rule for reading a depth curve, and reports that the hardest
of the four regimes is a familiar protein family carrying a structurally
novel pocket.

## What is here, and what is not

`code/` holds every script that produced a reported number. `results/`
holds the JSON and table outputs those scripts wrote, so any figure in
the paper can be checked against the file it came from without rerunning
anything.

The representations themselves are not here: 1,820 systems at 48 layers
is roughly 40 GB. Neither are the PLINDER release files, which are
public and large. `results/probe_index.parquet` and
`results/per_system_auprc.parquet` are included because they are small
and because every regime contrast in the paper is computable from them.

## Pipeline

Scripts run in this order. Each is self-documenting: its docstring
states what it does, what it refuses to do, and why.

**Familiarity axes.** `build_quadrants.py` builds the protein-family
axis from PLINDER's chain-level sequence clustering.
`export_chain_clusters.py` reads the same tables as files rather than
through the package API, for the reason in *Provenance* below.
`build_continuous_sf.py` produces the continuous support count.
`quadrant_feasibility.py` is the pre-registration: it settles whether
all four cells can be populated, on the data alone, before any model
runs. `verify_quadrants.py` re-derives both axes from source and
compares against the stored labels.

**Extraction.** `export_sequences.py`, `prefetch_msas.py`,
`sample_extraction_set.py`, `extract_representations.py`. Line 162 of
`sample_extraction_set.py` is where the four cells are actually formed.
`boltz_probe.py` runs one protein through the trunk with hooks live and
reports what was captured; it is how the Boltz-2 checkpoint problem
reported in the paper was found. `load_representations.py` reads the
output — use it rather than `np.load`, since the arrays store bfloat16
bits as uint16.

**Probing.** `assemble_probe_data.py`, `run_depth_probes.py`,
`per_system_scores.py`.

**Onset.** `validate_onset_rule.py` fixes the persistence requirement on
synthetic curves with known ground truth, before any real representation
was seen. `calibrate_null.py` and `merge_null.py` build the permutation
null; `audit_onset_null.py` measures the rule's false-positive rate on
that null using a leave-one-out construction.

**Analysis.** `bootstrap_contrasts.py` gives percentile intervals from
receptor-family cluster resampling; `holm_contrasts.py` replays that
exact resampling and attaches Holm-corrected p-values over all eighteen
contrasts. The robustness checks are `design_factor_check.py`,
`alternative_sp_check.py`, `msa_covariate_check.py`,
`fn_composition_check.py`, the three `covalent_*_check.py` scripts,
`continuous_two_axis.py` and `readout_capacity_check.py`.

**Tables and figures.** The `generate_*.py` scripts and
`plot_depth_curves.py` read the JSON in `results/` and write the LaTeX
the paper includes. Several also emit `\newcommand` macro files, so the
numbers in the prose come from the same file as the numbers in the
table beside it.

**Audit.** `audit_cutoff_provenance.py` checks that every artefact came
from the 2021-09-30 cutoff rather than the 2023-06-01 one, testing the
question from the data rather than from filenames.

## Provenance, and three things a reader should know

**The PLINDER release is 2026-09, and the installed client will not
serve it.** `plinder-core` 0.2.27 defaults to release 2024-06 and
substitutes it without warning. This is not hypothetical: it produced
one wrong analysis during this work, reporting 43% target coverage that
looked like a property of the data rather than a wrong directory. Every
script that touches PLINDER now takes the release directory as an
argument and reads parquet files directly. `build_quadrants.py` calls
`pc.query_table()`, which the current package does not provide; it ran
correctly in October 2026 and is included as the record of how the
family axis was built, with `export_chain_clusters.py` as the working
path today.

**The per-ligand pocket-similarity scores have no producing script.**
They were computed in October 2026 and the code was not retained. The
rule they implement is recorded in
`results/quadrants_pocket_qcov_boltz1.json`, along with the full
threshold sweep, so the quantity is reproducible in principle — but by
re-implementation from that record, not by rerunning. The scores
themselves determine every cell assignment, so this is the weakest point
in the chain and is stated in the paper as well.

**One stale column.** `targets_with_sp_boltz1.parquet` carries a
`quadrant` column left over from an earlier construction at a different
threshold. It disagrees with `probe_index.parquet` on about half the
systems. The index is correct — recomputing from `fam_f` and `S_p` at
threshold 80 matches it exactly — and every analysis reads the index.
The column is not included here.

## What the audit can and cannot establish

`audit_cutoff_provenance.py` identifies the pipeline from the data:
systems released between 2021-09-30 and 2023-06-01 exist in the Boltz-1
target pool and cannot exist in the Boltz-2 one, so their presence is
decisive. It also checks that the probe index exists and is well formed, that
no JSON records the wrong cutoff, and that the extraction manifest
matches. Where the target table still carries a quadrant column it
compares that against the index too; ours does not, for the reason
in *Provenance* above.

It cannot tell whether a table in the manuscript still matches the file
on disk. That gap is how a table generated against the wrong cutoff
reached an earlier draft, and it is checked by hand.

`verify_quadrants.py` has a narrower scope than its name suggests. It
verifies that the stored labels are consistent with the supplied
per-ligand scores; it does not independently reconstruct how those
scores were computed, so it cannot establish comparison direction,
reference eligibility, self-exclusion or ligand coverage.

Running the audit on the authors' own results directory reports three
problems, all JSON files in directories named `SUPERSEDED` or
`boltz2`. They are kept as the record of a rejected pipeline and feed
nothing in the paper. The audit cannot tell a deliberately archived
file from a wrong one in use, which is the correct conservatism.

## Results files

The onset false-positive rates are in `results/audit_onset_fpr.tex` and
`audit_onset_summary.md` rather than a JSON, because
`audit_onset_null.py` wrote tables directly. Everything else in
`results/` is JSON.

## Environment

```
conda env create -f environment.yml
conda activate beyond-pocket
```

Only the extraction step needs Boltz and a GPU. Everything downstream
runs from the parquet and JSON files on a laptop.

## Citation

Please cite the paper. A BibTeX entry will be added here on
publication.

## License

MIT.
