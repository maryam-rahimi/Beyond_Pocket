#!/usr/bin/env python3
"""
Extract per-layer Boltz-1 trunk representations for the manifest systems.

For each system this writes, in half precision:

    s        (n_layers, L, 384)        per-residue track, every block
    z_pool   (n_layers, L, 3*128)      pooled pair summary, every block
    z_final  (L, L, 128)               full pair tensor, final block only

one .npz per system, plus a row in an index file.

WHAT THE PROBE ESTABLISHED, AND THIS RELIES ON
----------------------------------------------
The extraction contract was validated separately before this script was
written, and the facts it depends on are not re-derived here:

  * The trunk is `pairformer_module`, selected by exact name. Boltz-1
    carries a SECOND 48-block stack inside its confidence module, so
    choosing by depth would be a coin flip between them.
  * Hooks fire n_blocks x (recycling_steps + 1) times, once per block per
    recycle, in block order within each recycle.
  * Boltz-1's published weights load against its published code with 0
    missing and 0 unexpected keys. (Boltz-2's do not: its checkpoint
    lacks 128 trunk layer-norm parameters its model class expects, which
    would be randomly initialised and would make every representation
    partly noise while looking entirely normal. That check is repeated
    here rather than assumed.)

RECYCLING
---------
The trunk runs `recycling_steps + 1` times over the same input. Only the
FINAL pass is kept: that is the representation the structure module
consumes, and keeping an earlier one would silently substitute a
partially refined trunk state. The script takes the last `n_blocks`
captures and asserts the count first.

WHY STOP BEFORE THE STRUCTURE MODULE
------------------------------------
A hook raises once the expected captures have arrived, so diffusion never
runs. Besides being wasted work on every system, it fails outright at
`num_sampling_steps=1` (`linalg.svd` does not converge on a degenerate
matrix), which would discard good trunk captures because of an error in a
component downstream of them.

THE POOLING
-----------
Full z is (L, L, 128): about 2 GB per protein across 48 layers at this
length, which is not storable for the target set. The per-residue summary
keeps three channels -- row mean, column mean and the self-pair diagonal
-- following the zPocket readout, rather than a single row mean. The
diagonal in particular is not recoverable from either marginal, and a
plain row-mean would discard it.

Full z is still written for the final layer, because the pair-processing
heads need real pair structure and pooling is precisely what destroys it.

PROTEIN-ONLY INPUT
------------------
Only protein entities are written to the input. The study asks whether
the representation locates pockets without being shown them; giving a
co-folding model the ligand tells it where the ligand binds. That is a
property of this experiment's question, not a universal rule.

RESUMABILITY
------------
A run over thousands of systems will be interrupted. Completed systems
are skipped on restart, and a system counts as complete only if its file
reads back with the expected arrays and shapes -- not merely if the path
exists, since a truncated file from a killed process does exist. Failures
are recorded per system and never abort the run.

USAGE
-----
    python extract_representations.py \\
        --manifest extraction_manifest.parquet \\
        --outdir reps/ --limit 5          # try a handful first
    python extract_representations.py \\
        --manifest extraction_manifest.parquet --outdir reps/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ImportError as e:
    sys.exit(f"missing dependency: {e}")

# NOTE: this script deliberately does NOT import plinder. PLINDER and
# Boltz pin incompatible dependency trees (pyarrow/NumPy), and the
# PLINDER version that installs alongside Boltz exposes a different table
# API. Sequences are exported once by export_sequences.py, using the
# PLINDER-capable interpreter, and read from JSON here.

DEFAULT_MSA_URL = "https://api.colabfold.com"
EXPECTED_KEYS = ("s", "z_pool", "layers", "length", "dtype")
STORAGE_DTYPE = "bfloat16"


class TrunkDone(Exception):
    """Raised from a hook once the expected captures have arrived."""


def msa_fingerprint(root: Path) -> dict:
    """Depth and content hash of the alignments actually used.

    These representations are MSA-conditioned, and the MSA comes from a
    live server that may answer differently later. Without recording what
    the alignment was, a representation cannot be reproduced even with the
    same checkpoint and the same sequence -- so the depth and a hash of
    the alignment files go in the index alongside the timing.
    """
    # Boltz-1 writes its alignments as .npz under the msa directory, not
    # as .a3m/.csv. Hashing only the text formats found nothing at all on
    # a real run -- zero files, depth -1 -- so the set must match what is
    # actually produced AND what --save-msa preserves, which is this
    # whole directory.
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix in (".a3m", ".csv", ".npz"))
    if not files:
        return {"msa_files": 0, "msa_depth_total": -1,
                "msa_depth_min": -1, "msa_depth_median": -1,
                "msa_depths": "", "msa_sha256": ""}
    h = hashlib.sha256()
    per_file = []
    for f in files:
        h.update(f.name.encode())
        data = f.read_bytes()
        h.update(data)
        # Count '>' at line starts. data.count(b"\n>") misses a header on
        # the very first byte, undercounting every alignment by one.
        if f.suffix == ".a3m":
            per_file.append(sum(1 for ln in data.splitlines()
                                if ln.startswith(b">")))
        elif f.suffix == ".csv":
            per_file.append(max(0, data.count(b"\n") - 1))
        else:
            # .npz is Boltz's processed MSA object. Its schema is fixed:
            #
            #   @dataclass(frozen=True)
            #   class MSA(NumpySerializable):
            #       sequences: np.ndarray   # one structured record per
            #       deletions: np.ndarray   # aligned sequence
            #       residues:  np.ndarray
            #
            # so depth is len(sequences). An earlier version guessed at
            # "the row count of the largest 2-D array", which matches
            # nothing here because these are structured 1-D arrays -- and
            # so recorded a depth of zero for every system while looking
            # like it had read something. Reading the named field is
            # schema-based rather than heuristic, and missing it is fatal
            # rather than silently zero, because this is a planned
            # covariate and not decoration.
            with np.load(f, allow_pickle=False) as arr:
                if "sequences" not in arr.files:
                    raise RuntimeError(
                        f"{f.name}: expected Boltz MSA field 'sequences', "
                        f"found {list(arr.files)}")
                per_file.append(int(arr["sequences"].shape[0]))
    per_file = [d for d in per_file if d > 0]
    if not per_file:
        per_file = [0]
    # Per ALIGNMENT FILE, not demonstrably per biological chain: the
    # layout is not guaranteed to be one file per chain, so these are
    # reported as per-alignment depths. The minimum is still the useful
    # covariate, since one deep alignment can mask a shallow one and it
    # is the shallow one that limits what the trunk sees.
    med = float(np.median(per_file))
    return {"msa_files": len(files),
            "msa_depth_total": int(sum(per_file)),
            "msa_depth_min": int(min(per_file)),
            "msa_depth_median": float(med),
            "msa_depths": ";".join(str(d) for d in per_file),
            "msa_sha256": h.hexdigest()[:32]}


def _boltz_source_hash() -> str:
    """Hash the installed Boltz model source.

    The version string alone is not enough: boltz reports 2.2.1 whether
    installed from PyPI or from a repository checkout, and those are not
    the same code. Hashing the model package's .py files distinguishes
    them, which matters because the representation is a property of the
    implementation as much as of the weights.
    """
    try:
        import boltz.model as bm
        root = Path(bm.__file__).parent
        h = hashlib.sha256()
        for f in sorted(root.rglob("*.py")):
            h.update(f.relative_to(root).as_posix().encode())
            h.update(f.read_bytes())
        return h.hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return "unknown"


def sha256(path: Path) -> str:
    """Hash a file in chunks.

    The scientific claim is about representations from one specific
    frozen checkpoint, so the checkpoint's identity belongs in the
    provenance record rather than its filename, which could name
    anything.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(cache: Path, device: str):
    """Construct Boltz-1 from cleaned hyperparameters and load weights.

    `load_from_checkpoint` is NOT used: it reads the checkpoint's own
    hyper_parameters, so filtering unsupported keys would not reliably
    take effect. Constructing explicitly and loading the state dict keeps
    the contract visible -- and makes the 0-missing/0-unexpected check
    below meaningful rather than decorative.
    """
    import inspect
    from boltz.model.models.boltz1 import Boltz1

    ckpt = cache / "boltz1_conf.ckpt"
    if not ckpt.exists():
        sys.exit(f"checkpoint not found: {ckpt}")

    raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    hp = dict(raw.get("hyper_parameters", {}))
    ok = set(inspect.signature(Boltz1.__init__).parameters)
    dropped = sorted(set(hp) - ok)
    for k in dropped:
        hp.pop(k)

    model = Boltz1(**hp)
    missing, unexpected = model.load_state_dict(raw["state_dict"],
                                                strict=False)
    if missing or unexpected:
        sys.exit(
            f"state_dict mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected.\\nParameters the model expects "
            f"but the checkpoint lacks are randomly initialised, so every "
            f"representation\\nextracted from this pairing would be partly "
            f"noise. Refusing to proceed.")
    model.eval().to(device)
    del raw
    return model, dropped


def find_trunk(model):
    """The top-level `pairformer_module`, by exact name."""
    import torch.nn as nn
    for name, mod in model.named_modules():
        if name == "pairformer_module":
            layers = getattr(mod, "layers", None)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                return layers
    sys.exit("no module named exactly 'pairformer_module'. Boltz-1 carries "
             "a second\n48-block stack in its confidence module, so "
             "selecting by depth is unsafe;\nname the trunk explicitly "
             "before proceeding.")


def pool_z(z: torch.Tensor) -> torch.Tensor:
    """(L, L, C) -> (L, 3C): row mean, column mean, self-pair diagonal."""
    idx = torch.arange(z.shape[0], device=z.device)
    return torch.cat([z.mean(dim=1), z.mean(dim=0), z[idx, idx]], dim=-1)


def is_complete(path: Path, n_layers: int, require_final_z: bool) -> bool:
    """A file counts as done only if it reads back intact AND matches the
    current configuration.

    Existence alone would treat a file truncated by a killed process as
    complete, and the run would skip it forever. Checking only the layer
    count is nearly as bad in one specific way: a trial run with
    --no-final-z writes files lacking z_final, and a later production run
    that needs z_final would skip them as already done. The resume
    contract therefore has to include what this run is configured to
    produce, not just what some earlier run happened to write.
    """
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as d:
            if not set(EXPECTED_KEYS).issubset(set(d.files)):
                return False
            # The stored bits are uint16 and mean nothing without the
            # dtype that says how to read them. A file written under a
            # different storage dtype must not be accepted as complete,
            # or the collection silently mixes two encodings.
            if str(np.asarray(d["dtype"]).reshape(-1)[0]) != STORAGE_DTYPE:
                return False
            L = int(np.asarray(d["length"]).reshape(-1)[0])
            if d["s"].shape != (n_layers, L, 384):
                return False
            if d["z_pool"].shape != (n_layers, L, 384):
                return False
            if not np.array_equal(d["layers"], np.arange(n_layers)):
                return False
            if require_final_z:
                if "z_final" not in d.files:
                    return False
                if d["z_final"].shape != (L, L, 128):
                    return False
            return True
    except Exception:  # noqa: BLE001
        return False


def write_input(workdir: Path, name: str, chains: dict,
                msa_paths: list | None = None) -> Path:
    """Protein-only YAML, one entity per chain, in the chains' own order.

    When msa_paths is given, each entity names its pre-fetched
    alignment. Boltz queues a chain for server fetching only when it has
    no such path (main.py line 573), so naming the file suppresses the
    request entirely, which is the point of the cache.

    IDENTICAL SEQUENCES MUST NAME THE SAME MSA FILE
    -----------------------------------------------
    Boltz groups entities by sequence and then requires the grouped
    items to agree on their msa field (schema.py 1108-1124):

        for item in items:
            if item[entity_type].get("msa", 0) != msa:
                raise ValueError("All proteins with the same sequence
                                  must share the same MSA!")

    The constraint is on the PATH, not on entity structure. Separate
    entities with the same sequence are accepted as long as they point
    at the same file. The inline path never hit this because Boltz
    generated one alignment per distinct sequence; a per-chain cache
    writes one file per chain index, so identical chains got different
    paths and were refused.

    So identical chains are mapped to the first cached path among them.
    Those alignments were computed from the same sequence and are the
    same alignment.

    Chain order is preserved exactly. An earlier attempt merged
    identical sequences into one entity with a list of ids, which would
    have reordered any system whose identical chains are not adjacent
    (five in this manifest) while leaving the total residue count
    unchanged, so the length check would not have caught the resulting
    label misalignment.
    """
    first_path: dict = {}
    if msa_paths:
        for i, seq in enumerate(chains.values()):
            if seq not in first_path:
                first_path[seq] = msa_paths[i]

    lines = ["version: 1", "sequences:"]
    for cid, seq in chains.items():
        lines += ["  - protein:", f"      id: {cid}",
                  f"      sequence: {seq}"]
        if msa_paths:
            lines.append(f"      msa: {first_path[seq]}")
    p = workdir / f"{name}.yaml"
    p.write_text("\n".join(lines) + "\n")
    return p


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sequences", required=True,
                    help="sequences.json from export_sequences.py")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cache", default=str(Path.home() / ".boltz"))
    ap.add_argument("--msa-cache", required=True,
                    help="directory of pre-fetched alignments from "
                         "prefetch_msas.py, one subdirectory per system")
    ap.add_argument("--allow-msa-server", action="store_true",
                    help="permit live MSA fetching for systems missing "
                         "from the cache. OFF by default: inline fetching "
                         "left the GPU idle 93%% of the time, so a silent "
                         "fallback would quietly restore a 250-GPU-hour "
                         "run.")
    ap.add_argument("--msa-server-url", default=DEFAULT_MSA_URL)
    ap.add_argument("--recycling-steps", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int,
                    help="stop after this many NEW systems (for a trial)")
    ap.add_argument("--allow-missing-sequences", action="store_true",
                    help="proceed when some manifest systems have no "
                         "exported sequence; without this the run stops, "
                         "so a shortfall is investigated rather than "
                         "silently accepted")
    ap.add_argument("--no-final-z", action="store_true",
                    help="skip the full final-layer pair tensor, which is "
                         "most of the storage")
    ap.add_argument("--keep-work", action="store_true",
                    help="keep per-system working directories, including "
                         "the alignments, instead of deleting them")
    ap.add_argument("--save-msa", metavar="DIR",
                    help="archive the processed MSA objects actually "
                         "consumed during extraction, for "
                         "representation-level provenance. The raw "
                         "alignments themselves live in --msa-cache and "
                         "are identified by its cache_sha256.")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    repdir = outdir / "systems"
    repdir.mkdir(exist_ok=True)
    index_path = outdir / "index.csv"
    fail_path = outdir / "failures.csv"

    man = pd.read_parquet(args.manifest)
    # A duplicate system_id would be collapsed by expected_len below
    # while len(man) still counted it, so completeness could never be
    # reached and the cause would not be obvious.
    if man["system_id"].isna().any():
        sys.exit("manifest contains missing system_id values")
    if man["system_id"].duplicated().any():
        dup = man.loc[man["system_id"].duplicated(keep=False),
                      "system_id"].tolist()
        sys.exit(f"manifest contains duplicate system_id values, "
                 f"e.g. {dup[:5]}")
    if man["length"].isna().any():
        sys.exit("manifest contains missing lengths; the per-residue "
                 "alignment contract cannot be checked without them")
    print(f"manifest: {len(man):,} systems")
    # The downstream target is a per-residue vector, so the label length,
    # the input sequence length and the model's token count must all be
    # the same number. If they ever diverge the arrays still look valid
    # and every residue-level probe is silently misaligned, so this is
    # checked at both ends rather than assumed.
    expected_len = dict(zip(man.system_id, man.length))

    # --- sequences ------------------------------------------------------
    seq_path = Path(args.sequences)
    if not seq_path.exists():
        sys.exit(f"no such file: {seq_path}\nRun export_sequences.py with "
                 f"the PLINDER-capable interpreter first.")
    sequences = json.loads(seq_path.read_text())
    print(f"sequences: {len(sequences):,} systems exported")
    missing_seq = [s for s in man.system_id if s not in sequences]
    if missing_seq:
        # Default to stopping. Quietly extracting 1,917 of 1,918 and
        # reporting it as the study's sample is the kind of discrepancy
        # that is never noticed again; an explicit flag makes the
        # exception a decision rather than an accident.
        msg = (f"{len(missing_seq):,} of {len(man):,} manifest systems "
               f"have no exported sequence,\ne.g. {missing_seq[:3]}")
        if not args.allow_missing_sequences:
            sys.exit(msg + "\n\nInvestigate the exporter's problems file, "
                     "or pass --allow-missing-sequences\nto proceed "
                     "deliberately with a reduced set.")
        print(f"  {msg}\n  proceeding anyway (--allow-missing-sequences)")

    # --- MSA cache coverage ---------------------------------------------
    # Checked BEFORE the model loads, so a missing cache costs seconds
    # rather than being discovered after a queue wait and 90s of
    # checkpoint loading.
    cache_root = Path(args.msa_cache)
    if not cache_root.is_dir():
        sys.exit(f"--msa-cache {cache_root} is not a directory. Run "
                 f"prefetch_msas.py first.")

    # The cache's identity belongs in the run signature. Without it,
    # regenerating the alignments and resuming into the same output
    # directory would skip the earlier systems on a matching signature
    # and build the rest from different inputs -- one collection resting
    # on two MSA sets, with nothing recording the change. The fingerprint
    # is computed by the prefetch job so the GPU job reads one string
    # rather than hashing every cached file at startup.
    cm_path = cache_root / "cache_manifest.json"
    if not cm_path.exists():
        sys.exit(f"no cache_manifest.json in {cache_root}. Rerun "
                 f"prefetch_msas.py, which writes it, so the "
                 f"representations can record\nwhich alignments they "
                 f"were built from.")
    cache_manifest = json.loads(cm_path.read_text())

    # The cache hash says WHICH alignments exist; it says nothing about
    # which sequences they were built for. Without this check a cache
    # generated from one sequences.json could be applied to another:
    # every file would be present, the structure valid, and each protein
    # would be run against an alignment computed for a different
    # sequence.
    cur_seq_sha = sha256(seq_path)
    cached_seq_sha = cache_manifest.get("sequences_sha256")
    if cached_seq_sha != cur_seq_sha:
        sys.exit(
            f"the MSA cache was generated from a different "
            f"sequences.json.\n  cache: {cached_seq_sha}\n  input: "
            f"{cur_seq_sha}\n\nRegenerate the cache from the current "
            f"--sequences file, or point --sequences\nat the file the "
            f"cache was built from.")

    # NOTE on the cache's "complete" flag. It records whether the cache
    # covers the sequences file it was built from, which is a different
    # question from whether it covers THIS manifest. A manifest
    # deliberately restricted to cached systems is fully served by a
    # cache that calls itself incomplete. The per-system scan below is
    # the check that matters, so the flag is reported rather than
    # enforced, and is used to explain a failure if the scan finds gaps.
    cache_partial = not cache_manifest.get("complete", False)
    if cache_partial:
        print(f"note: the cache reports itself partial "
              f"({cache_manifest.get('systems_cached','?')} of "
              f"{cache_manifest.get('systems_expected','?')} systems "
              f"for the sequences file it was built from).\n"
              f"      What matters is whether it covers this manifest, "
              f"checked next.")
    # Honour the cache's own contract. prefetch_msas.py writes
    # .complete only after every chain CSV has been re-read through
    # Boltz's parser, so the marker is a stronger statement than the
    # filenames being present -- a partially copied directory can have
    # every name and no valid content.
    uncached = []
    for sysid, rec in sequences.items():
        if sysid not in set(man.system_id):
            continue
        d = cache_root / sysid
        n = len(rec["chains"])
        ok = (d / ".complete").exists() and all(
            (d / f"{i}.csv").exists() for i in range(n))
        if ok:
            try:
                meta = json.loads((d / ".complete").read_text())
                ok = int(meta.get("chains", -1)) == n
            except Exception:  # noqa: BLE001
                ok = False
        if not ok:
            uncached.append(sysid)
    # Counted over MANIFEST systems, not the whole sequences file: a
    # subset manifest against the full sequences file would otherwise
    # report a coverage figure for systems this run never touches.
    n_manifest = len(set(man.system_id))
    print(f"\nMSA cache: {n_manifest - len(uncached):,} of "
          f"{n_manifest:,} manifest systems have alignments on disk")
    if uncached:
        msg = (f"{len(uncached):,} systems are not cached, "
               f"e.g. {uncached[:3]}")
        if args.allow_msa_server:
            print(f"  {msg}\n  these will be fetched inline "
                  f"(--allow-msa-server), which is slow")
        else:
            tail = ("\n\nThe cache reports itself partial, which is "
                    "consistent with these gaps.\nEither run "
                    "prefetch_msas.py to completion, or restrict the "
                    "manifest to\nthe systems the cache covers."
                    if cache_partial else "")
            sys.exit(f"  {msg}\n\nFetching these inline would hold the "
                     f"GPU idle while waiting on the\nserver. Run "
                     f"prefetch_msas.py to completion first, or pass\n"
                     f"--allow-msa-server to accept the cost "
                     f"deliberately." + tail)

    # --- model ----------------------------------------------------------
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("CUDA unavailable; pass --device cpu to run (slowly) on CPU")
    print(f"\nloading model on {device} ...")
    t0 = time.time()
    model, dropped = load_model(Path(args.cache), device)
    layers = find_trunk(model)
    n_blocks = len(layers)
    print(f"  loaded in {time.time()-t0:.0f}s | trunk: {n_blocks} blocks "
          f"| 0 missing, 0 unexpected")
    if dropped:
        print(f"  dropped checkpoint kwargs (training-time): {dropped}")

    from boltz.main import process_inputs
    from boltz.data.module.inference import BoltzInferenceDataModule
    from boltz.data.types import Manifest

    # Captures hold POOLED, fp16, CPU tensors. Retaining the full pair
    # tensor from every block instead would need about 48 x L^2 x 128 x 4
    # bytes live at once -- roughly 16 GB at the 800-residue cap, on top
    # of the model's own activations, which does not fit a 32 GB card.
    # Pooling inside the hook drops peak retained memory from O(n*L^2*C)
    # to O(L^2*C) + O(n*L*C): only the block currently executing holds a
    # full pair tensor.
    captures: list = []
    final_z: dict = {}

    def make_hook(i, stop_after, keep_final_z):
        def hook(module, inputs, output):
            s_t, z_t = output[0], output[1]
            captures.append((
                i,
                s_t[0].detach().to("cpu", torch.bfloat16),
                pool_z(z_t[0].detach()).to("cpu", torch.bfloat16),
            ))
            if len(captures) >= stop_after:
                # This firing is the last block of the last recycle, so
                # this is the pair tensor the heads need.
                if keep_final_z:
                    final_z["z"] = z_t[0].detach().to(
                        "cpu", torch.bfloat16)
                raise TrunkDone
        return hook

    # --- run signature ---------------------------------------------------
    # Resume checks shapes, which says a file is well-formed but not that
    # it was produced under THIS configuration. Without that, changing the
    # checkpoint, the sequences, the recycling depth or the MSA server and
    # rerunning into the same directory would skip the old files on a
    # shape check and then overwrite provenance.json with the new
    # settings -- leaving a mixed collection whose provenance describes
    # all of it as the newer run. That is the precise class of quiet error
    # the rest of this pipeline exists to prevent.
    ckpt_path = Path(args.cache) / "boltz1_conf.ckpt"
    try:
        import importlib.metadata as _md
        _boltz_v = _md.version("boltz")
    except Exception:  # noqa: BLE001
        _boltz_v = "unknown"

    # Hardware belongs in the signature, not only in the closing record.
    # Kernel selection and reduction order differ between architectures
    # and between CPU and GPU, so representations from a V100 and an
    # H100 are not interchangeable even at identical software versions.
    # Recorded here, a resume onto different hardware is refused rather
    # than silently mixing two numerical regimes in one collection.
    if device.startswith("cuda"):
        _dev_idx = torch.cuda.current_device()
        _dev_name = torch.cuda.get_device_name(_dev_idx)
        _dev_cap = list(torch.cuda.get_device_capability(_dev_idx))
    else:
        _dev_name, _dev_cap = "cpu", None
    run_config = {
        "manifest_sha256": sha256(Path(args.manifest)),
        "sequences_sha256": sha256(seq_path),
        "checkpoint_sha256": sha256(ckpt_path),
        # The cache identity belongs HERE, in the config the resume
        # comparison reads -- not only in the final provenance record.
        # Recording it only at the end would let a regenerated cache pass
        # the signature check, skip the systems already extracted, and
        # build the rest from different alignments, with the closing
        # provenance naming the new cache as though it had produced all
        # of them.
        "msa_cache_sha256": cache_manifest["cache_sha256"],
        "msa_cache_partial": bool(cache_partial),
        "msa_pairing": cache_manifest.get("pairing", "unknown"),
        "msa_cache_boltz_source_sha256":
            cache_manifest.get("boltz_source_sha256", "unknown"),
        "recycling_steps": args.recycling_steps,
        "msa_server_url": args.msa_server_url,
        "final_z_stored": not args.no_final_z,
        "pooling": "row mean | column mean | self-pair diagonal",
        "storage_dtype": STORAGE_DTYPE,
        # The implementation belongs in the compatibility check, not only
        # in the final record: the same checkpoint run through a
        # different Boltz checkout or torch build can produce different
        # representations, and without these the signature would match
        # while the numbers had moved.
        "boltz_version": _boltz_v,
        "boltz_source_sha256": _boltz_source_hash(),
        "torch_version": torch.__version__,
        "device": device,
        "device_name": _dev_name,
        "cuda_version": torch.version.cuda,
        "cuda_capability": _dev_cap,
    }
    run_sig = hashlib.sha256(
        json.dumps(run_config, sort_keys=True).encode()).hexdigest()[:16]
    print(f"run signature: {run_sig}")

    prov_path = outdir / "provenance.json"
    if prov_path.exists():
        # Fail closed on an unreadable record. A corrupt provenance
        # means the configuration of the files already here cannot be
        # established, and proceeding would write this run's
        # configuration over an unknown one, making those files appear
        # to belong to it.
        try:
            prev_prov = json.loads(prov_path.read_text())
        except Exception as e:  # noqa: BLE001
            sys.exit(
                f"cannot read {prov_path.name}: {type(e).__name__}: "
                f"{e}\nRefusing to resume: the configuration of the "
                f"existing representations cannot be\nestablished, so "
                f"they cannot be shown compatible with this run.")
        prev_cfg = prev_prov.get("run_config")
        if not isinstance(prev_cfg, dict) or not prev_cfg:
            sys.exit(
                f"{prov_path.name} has no usable run_config. Refusing "
                f"to resume into a\ncollection whose configuration is "
                f"unknown.")
        if prev_cfg != run_config:
            diff = {k: (prev_cfg.get(k), run_config[k])
                    for k in run_config
                    if prev_cfg.get(k) != run_config[k]}
            sys.exit(
                f"{outdir} holds representations from a DIFFERENT "
                f"configuration.\nDiffering settings (existing -> "
                f"requested):\n" +
                "\n".join(f"  {k}: {a} -> {b}" for k, (a, b) in diff.items())
                + "\n\nResuming would mix two collections under one "
                  "provenance record. Use a\nnew --outdir, or delete the "
                  "existing one deliberately.")
    else:
        # The same failure with the record deleted rather than damaged:
        # outputs present and no provenance at all. Writing a fresh one
        # would assign this configuration to files of unknown origin.
        orphans = sorted(repdir.glob("*.npz"))
        if orphans or index_path.exists():
            sys.exit(
                f"{outdir} already contains "
                f"{len(orphans)} representation file(s)"
                f"{' and an index' if index_path.exists() else ''} but "
                f"no provenance.json.\nRefusing to assign this run's "
                f"configuration to files of unknown provenance.\nUse a "
                f"new --outdir, or remove the existing outputs "
                f"deliberately.")

    # Write the signature NOW, not at the end. Provenance written only on
    # completion leaves a window: a first run killed partway through
    # leaves NPZ files but no record, so a later run under a different
    # configuration finds nothing to compare against and silently mixes
    # the two. The completion fields are filled in afterwards.
    prov_path.write_text(json.dumps(
        {"started": datetime.now().isoformat(timespec="seconds"),
         "run_signature": run_sig, "run_config": run_config,
         "status": "in_progress"}, indent=2))

    # --- resume ----------------------------------------------------------
    done = set()
    if index_path.exists():
        prev = pd.read_csv(index_path)
        for sid in prev.system_id:
            if is_complete(repdir / f"{sid}.npz", n_blocks,
                           require_final_z=not args.no_final_z):
                done.add(sid)
        print(f"\nresuming: {len(done):,} systems already complete")

    todo = [s for s in man.system_id
            if s not in done and s in sequences]
    if args.limit:
        todo = todo[:args.limit]
    print(f"to extract: {len(todo):,}")

    work = outdir / "work"
    work.mkdir(exist_ok=True)

    rows, fails = [], []
    t_start = time.time()
    for n, sysid in enumerate(todo, 1):
        t_sys = time.time()
        sysdir = work / sysid.replace("/", "_")
        success = False
        msa_dir = None
        msa_info = {}
        try:
            rec = sequences.get(sysid)
            if rec is None:
                raise RuntimeError("no exported sequence")

            chain_seqs, order = {}, []
            for c in rec["chains"]:
                cid = chr(ord("A") + len(chain_seqs))
                chain_seqs[cid] = c["sequence"]
                order.append(c["instance"])

            want_len = int(expected_len[sysid])
            seq_len = sum(len(v) for v in chain_seqs.values())
            if seq_len != want_len:
                raise RuntimeError(
                    f"input/label length mismatch: sequences={seq_len}, "
                    f"manifest={want_len}")

            # Pre-fetched alignments, named per chain in the YAML. A
            # system missing from the cache is a hard failure rather than
            # a live fetch: inline fetching is what left the GPU idle 93%
            # of the time, and a silent fallback would reintroduce it one
            # system at a time without anything in the log saying so.
            cachedir = Path(args.msa_cache) / sysid
            msa_paths = [cachedir / f"{i}.csv"
                         for i in range(len(chain_seqs))]
            missing = [p for p in msa_paths if not p.exists()]
            if missing and not args.allow_msa_server:
                raise RuntimeError(
                    f"no cached MSA: {missing[0].name} absent from "
                    f"{cachedir}. Run prefetch_msas.py for this system, "
                    f"or pass --allow-msa-server to fetch inline "
                    f"(which is slow).")
            use_cache = not missing

            if sysdir.exists():
                shutil.rmtree(sysdir)
            sysdir.mkdir(parents=True)
            inp = write_input(sysdir, "t", chain_seqs,
                              msa_paths if use_cache else None)
            proc = sysdir / "processed"

            process_inputs(
                data=[inp], out_dir=proc,
                ccd_path=Path(args.cache) / "ccd.pkl",
                mol_dir=Path(args.cache) / "mols",
                msa_server_url=args.msa_server_url,
                msa_pairing_strategy="greedy",
                use_msa_server=not use_cache, boltz2=False)

            cands = sorted(proc.rglob("manifest.json"))
            if not cands:
                raise RuntimeError("no manifest produced")
            root = cands[0].parent
            # Fingerprint the directory the data module is actually given,
            # not every .a3m/.csv anywhere under processed/. Otherwise the
            # recorded hash could cover files Boltz never read.
            msa_dir = root / "msa"
            msa_info = msa_fingerprint(msa_dir)
            dm = BoltzInferenceDataModule(
                manifest=Manifest.load(cands[0]),
                target_dir=root / "structures",
                msa_dir=root / "msa", num_workers=0)
            dm.setup("predict")
            batch = next(iter(dm.predict_dataloader()))
            batch = {k: (v.to(device) if hasattr(v, "to") else v)
                     for k, v in batch.items()}

            stop_after = n_blocks * (args.recycling_steps + 1)
            captures.clear()
            final_z.clear()
            handles = [layers[i].register_forward_hook(
                make_hook(i, stop_after, not args.no_final_z))
                for i in range(n_blocks)]
            try:
                with torch.no_grad():
                    model(batch, recycling_steps=args.recycling_steps,
                          num_sampling_steps=1, diffusion_samples=1,
                          max_parallel_samples=1)
            except TrunkDone:
                pass
            finally:
                for h in handles:
                    h.remove()

            if len(captures) != stop_after:
                raise RuntimeError(
                    f"expected {stop_after} captures, got {len(captures)}")

            # Final recycle only: the last n_blocks captures, which the
            # probe confirmed arrive in block order.
            final = captures[-n_blocks:]
            idx = [i for i, _, _ in final]
            if idx != list(range(n_blocks)):
                raise RuntimeError(f"final recycle out of order: {idx[:6]}...")

            s_all = torch.stack([t for _, t, _ in final])
            zp_all = torch.stack([t for _, _, t in final])
            L = int(s_all.shape[1])
            if L != want_len:
                raise RuntimeError(
                    f"model/label length mismatch: model={L}, "
                    f"manifest={want_len}. The representation cannot be "
                    f"aligned to the labels.")

            # dtype=str, not object: an object array needs pickle to load,
            # and these files are read back with allow_pickle=False.
            # numpy has no bfloat16, so the raw bits are stored as
            # uint16 and "dtype" says how to read them back:
            #   torch.from_numpy(a).view(torch.bfloat16).float()
            payload = {"s": s_all.view(torch.uint16).numpy(),
                       "z_pool": zp_all.view(torch.uint16).numpy(),
                       "dtype": np.asarray([STORAGE_DTYPE], dtype=str),
                       "layers": np.arange(n_blocks),
                       "length": np.array([L]),
                       "chain_order": np.asarray(order, dtype=str)}
            if not args.no_final_z:
                if "z" not in final_z:
                    raise RuntimeError("final pair tensor was not captured")
                payload["z_final"] = final_z["z"].view(
                    torch.uint16).numpy()

            tmp = repdir / f".{sysid}.tmp.npz"
            np.savez_compressed(tmp, **payload)
            tmp.rename(repdir / f"{sysid}.npz")
            success = True
            captures.clear()
            final_z.clear()

            dt = time.time() - t_sys
            size = (repdir / f"{sysid}.npz").stat().st_size
            rows.append({"system_id": sysid, "length": L,
                         "n_layers": n_blocks, "bytes": size,
                         "seconds": round(dt, 1), "run_sig": run_sig,
                         **msa_info})
            if n % 10 == 0 or n == 1:
                rate = (time.time() - t_start) / n
                left = (len(todo) - n) * rate / 3600
                print(f"  [{n}/{len(todo)}] {sysid} L={L} "
                      f"{size/1e6:.0f}MB {dt:.1f}s | "
                      f"{rate:.1f}s/sys, ~{left:.1f}h remaining")
        except Exception as e:  # noqa: BLE001
            # One system failing must not end the run. Record and continue.
            captures.clear()
            final_z.clear()
            fails.append({"system_id": sysid, "error": f"{type(e).__name__}: {e}",
                          "traceback": traceback.format_exc()[-800:]})
            print(f"  [{n}/{len(todo)}] {sysid} FAILED: "
                  f"{type(e).__name__}: {str(e)[:90]}")
        finally:
            # Archive the alignment ONLY on success, and only the
            # directory that was fingerprinted. Saving on failure would
            # leave an archive from a failed attempt; a later retry gets
            # a fresh answer from the live server, writes the new hash to
            # index.csv, and finds the destination already present -- so
            # the recorded hash and the archived alignment would describe
            # different alignments, which is the exact provenance
            # corruption this archive exists to prevent.
            if success and args.save_msa and msa_dir and msa_dir.exists():
                try:
                    dest = Path(args.save_msa) / sysid.replace("/", "_")
                    if dest.exists():
                        # A retry succeeded where an earlier attempt
                        # failed. Replace rather than keep, then verify.
                        shutil.rmtree(dest)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(msa_dir, dest)
                    saved = msa_fingerprint(dest)
                    if saved["msa_sha256"] != msa_info.get("msa_sha256"):
                        raise RuntimeError(
                            f"archived MSA fingerprint {saved['msa_sha256']}"
                            f" != recorded {msa_info.get('msa_sha256')}")
                except Exception as e:  # noqa: BLE001
                    print(f"    (MSA archive problem for {sysid}: "
                          f"{type(e).__name__}: {e})")
            if not args.keep_work and sysdir.exists():
                shutil.rmtree(sysdir, ignore_errors=True)
            # Flush after every system so an interrupted run resumes
            # from the last success rather than the last checkpoint.
            if rows:
                # index.csv describes the files currently on disk, so a
                # system re-extracted after its NPZ failed validation
                # must replace its earlier row rather than sit beside
                # it. Appending would leave two success rows for one
                # system, with stale metadata in the first.
                #
                # failures.csv is different and stays append-only: it is
                # an event log, and a failure that was later repaired
                # still happened.
                new_rows = pd.DataFrame(rows)
                if index_path.exists():
                    # Fail closed. Swallowing a read error here would
                    # replace the whole index with this one row: the
                    # NPZ files would survive, so nothing would look
                    # wrong until assembly found an almost empty index.
                    try:
                        prev = pd.read_csv(index_path)
                    except Exception as e:  # noqa: BLE001
                        raise RuntimeError(
                            f"cannot read the existing index "
                            f"{index_path.name} to update it: "
                            f"{type(e).__name__}: {e}. Refusing to "
                            f"overwrite it; the representations on "
                            f"disk are unaffected.") from e
                    if "system_id" not in prev.columns:
                        raise RuntimeError(
                            f"{index_path.name} has no system_id "
                            f"column; refusing to overwrite it")
                    prev = prev[~prev.system_id.isin(
                        set(new_rows.system_id))]
                    out_df = pd.concat([prev, new_rows],
                                       ignore_index=True)
                else:
                    out_df = new_rows
                # Atomic replace: an interruption mid-write would
                # otherwise truncate the index of every completed
                # system, not just this one.
                tmp_idx = index_path.with_suffix(".csv.tmp")
                out_df.to_csv(tmp_idx, index=False)
                tmp_idx.replace(index_path)
                rows = []
            if fails:
                pd.DataFrame(fails).to_csv(
                    fail_path, mode="a" if fail_path.exists() else "w",
                    header=not fail_path.exists(), index=False)
                fails = []

    el = time.time() - t_start
    # Completeness is measured against the MANIFEST, with the same
    # contract resume uses. Counting *.npz in the directory would let a
    # stale or foreign file stand in for a missing system, and would not
    # re-check that each file is well-formed -- so 1,917 good files plus
    # one unrelated .npz could report the collection as complete.
    manifest_ids = set(man.system_id)
    valid = {sid for sid in manifest_ids
             if is_complete(repdir / f"{sid}.npz", n_blocks,
                            require_final_z=not args.no_final_z)}
    n_done = len(valid)
    total = sum((repdir / f"{sid}.npz").stat().st_size for sid in valid)
    stray = len([f for f in repdir.glob("*.npz")
                 if f.stem not in manifest_ids])
    if stray:
        print(f"note: {stray} .npz file(s) in {repdir.name} are not "
              f"manifest systems and are not counted")
    print(f"\n{n_done:,} of {len(man):,} manifest systems on disk, "
          f"{total/1e9:.1f} GB")
    print(f"elapsed {el/3600:.2f}h")

    # failures.csv is an append-only AUDIT LOG of failure events. A system
    # that failed once and succeeded on a later resume still has its old
    # row, so the row count is not the number of systems currently
    # missing. Both numbers are reported, because only the second one
    # says what is left to do.
    missing = sorted(manifest_ids - valid)
    if fail_path.exists():
        fl = pd.read_csv(fail_path)
        print(f"{len(fl):,} failure events logged in {fail_path.name} "
              f"(append-only audit log: a system that failed once and "
              f"succeeded\n  on a later resume keeps its old row, so this "
              f"is not the number outstanding)")
    if missing:
        print(f"{len(missing):,} manifest systems without a valid file, "
              f"e.g. {missing[:3]}")

    manifest_complete = (n_done == len(manifest_ids))
    unresolved = missing

    prov_path.write_text(json.dumps({
        "when": datetime.now().isoformat(timespec="seconds"),
        # "invocation_complete" means THIS run ended normally, which after
        # --limit is true with only a handful of systems extracted.
        # Whether the collection is finished is a separate question.
        "status": "invocation_complete",
        "manifest_complete": bool(manifest_complete),
        "unresolved_failures": len(unresolved),
        "run_signature": run_sig,
        "run_config": run_config,
        "systems_on_disk": int(n_done),
        "bytes_on_disk": int(total),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(Path(args.manifest)),
        "manifest_systems": int(len(man)),
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256(ckpt_path),
        "boltz": run_config["boltz_version"],
        "torch": torch.__version__,
        "torch_cuda": run_config["cuda_version"],
        "device": run_config["device_name"],
        "msa_source": ("cache" if not args.allow_msa_server
                       else "cache_with_server_fallback"),
        "msa_server_url": args.msa_server_url,
        "sequences": str(seq_path),
        "sequences_sha256": sha256(seq_path),
        "n_blocks": n_blocks,
        "recycling_steps": args.recycling_steps,
        "dropped_hyperparameters": dropped,
        "pooling": "row mean | column mean | self-pair diagonal",
        "dtype": "bfloat16 (stored as uint16 bits)",
        "final_z_stored": not args.no_final_z,
    }, indent=2))
    print(f"Wrote {prov_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
