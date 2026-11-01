#!/usr/bin/env python3
"""
Boltz extraction probe (Boltz-1 or Boltz-2).

Runs one protein through the chosen Boltz trunk IN THIS PROCESS, with forward
hooks live on every Pairformer block, and reports what was actually
captured. Nothing here is asserted from the architecture on paper: shapes,
dtypes, recycle counts and byte costs are all measured from the tensors that
came back, and the probe exits non-zero if it captured nothing.

WHY IN-PROCESS
--------------
An earlier version of this probe ran inference through the CLI in a
subprocess and separately attached hooks to a second, idle copy of the
model. That establishes only that hooks can be attached. It cannot show
that they fire, what they return, or how recycling interacts with them --
which are the three things worth knowing. This version builds features with
Boltz's own pipeline and calls the model directly, so the hooks run on the
forward pass being measured.

THE RECYCLING TEST
------------------
The trunk recycles: its output is fed back and the stack runs again. Hooks
therefore fire n_blocks x (recycling_steps + 1) times per protein, and a
naive "first capture per layer" would silently record an early recycle
rather than the representation the structure module consumes. The probe
runs twice, with recycling_steps=0 and 1, and checks that the capture count
scales as predicted. If it does not, the recycling model is wrong and the
extractor must not be written against it.

MSA
---
Boltz requires alignments; there is no implicit single-sequence mode
(main.py raises "Missing MSA's in input and --use_msa_server flag not
set"). This probe therefore supports only --use-msa-server, which fetches
them and needs network access from the compute node.

A --msa-dir option for precomputed alignments was removed rather than left
in place behind a warning. The data module accepts an msa_dir, but
process_inputs runs before it and is not passed one, so the route would
have failed during preprocessing whatever the directory contained. An
option that cannot work is worse than no option, and a flag asserting the
user has "verified" it does not change which code path runs.

INPUT SCOPE
-----------
The probe submits protein sequence alone. This matches the study's design,
which asks whether the representation locates pockets without being shown
them; a ligand in the input would answer a different question, since a
co-folding model given the binding partner has been told where it binds.
That is a choice about what the experiment measures, not a universal rule:
for a task where the complex is a legitimate input, ligand conditioning is
not leakage.

USAGE
-----
    python boltz_probe.py --use-msa-server --out probe_report.json
    python boltz_probe.py --use-msa-server --seq-file my_protein.fasta
    python boltz_probe.py --use-msa-server --recycling-steps 1
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

# Ubiquitin, 76 aa. Small enough that failure is quick, common enough that
# any MSA server will have deep alignments for it.
UBIQUITIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)

DEFAULT_MSA_URL = "https://api.colabfold.com"


def die(msg: str, code: int = 1):
    print(f"\nFAILED: {msg}", file=sys.stderr)
    sys.exit(code)


def read_sequence(args) -> str:
    if args.seq_file:
        p = Path(args.seq_file)
        if not p.exists():
            die(f"no such file: {p}")
        lines = [l.strip() for l in p.read_text().splitlines()
                 if l.strip() and not l.startswith(">")]
        seq = "".join(lines)
        if not seq:
            die(f"no sequence found in {p}")
        return seq
    return args.seq


def write_input(workdir: Path, seq: str, name: str) -> Path:
    """Protein-only YAML. No ligand entity is written; see INPUT SCOPE."""
    yaml = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {seq}
"""
    p = workdir / f"{name}.yaml"
    p.write_text(yaml)
    return p


def find_pairformer(model):
    """Find the trunk Pairformer stack.

    Collects every module holding a non-empty ModuleList whose qualified
    name mentions pairformer, prints them all, and then prefers the one
    named exactly "pairformer_module" -- the trunk. Size is only a
    tiebreak of last resort: Boltz-1 carries two 48-block stacks, the
    trunk and the confidence module's, so choosing by depth alone would
    be a coin flip between them. If the exact name is absent and the two
    largest tie, this fails rather than guessing.
    """
    import torch.nn as nn

    found = []
    for name, mod in model.named_modules():
        if "pairformer" not in name.lower():
            continue
        layers = getattr(mod, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            found.append((name, mod, layers))
    if not found:
        return None, None, None
    # Print every candidate before choosing, then prefer the TRUNK by name
    # rather than by size. Boltz-1 carries two 48-block stacks --
    # pairformer_module and confidence_module.pairformer_module -- so
    # "longest wins" is a coin flip between them, and hooking the
    # confidence stack would yield captures that look entirely plausible
    # and describe the wrong thing. The exact top-level name is the only
    # reliable discriminator.
    print("pairformer candidates:")
    for nm, _, ly in sorted(found, key=lambda t: len(t[2]), reverse=True):
        print(f"    {nm}: {len(ly)} blocks")

    exact = [f for f in found if f[0] == "pairformer_module"]
    if exact:
        if len(found) > 1:
            print("    -> selecting 'pairformer_module' by exact name "
                  "(the trunk),\n       not by block count")
        return exact[0]

    found.sort(key=lambda t: len(t[2]), reverse=True)
    sizes = [len(f[2]) for f in found]
    if len(sizes) > 1 and sizes[0] == sizes[1]:
        # Fail rather than warn. Announcing that a choice is arbitrary and
        # then making it anyway is the contradiction an audit probe exists
        # to avoid: everything downstream would be certified against a
        # stack picked by dict ordering.
        die("no module named exactly 'pairformer_module', and the two "
            "largest\nPairformer candidates tie in depth "
            f"({sizes[0]} blocks each). The trunk cannot be\nidentified "
            "safely from structure alone. Name it explicitly in this "
            "function\nbefore running the probe against this "
            "architecture.")
    print("    -> no module named exactly 'pairformer_module'; "
          "falling back to the\n       largest stack, which is "
          f"unambiguous here ({sizes[0]} blocks vs "
          f"{sizes[1] if len(sizes) > 1 else 0})")
    return found[0]


def summarize(t):
    """Shape, dtype and true byte size of a captured tensor."""
    return {
        "shape": tuple(int(x) for x in t.shape),
        "dtype": str(t.dtype),
        "bytes": int(t.numel() * t.element_size()),
    }


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", default=UBIQUITIN,
                    help="sequence to probe with (default: ubiquitin, 76 aa)")
    ap.add_argument("--seq-file",
                    help="FASTA file to read the sequence from instead; use "
                         "this rather than --seq to probe a longer protein")
    ap.add_argument("--truncate", type=int,
                    help="shorten the sequence to this many residues "
                         "(cannot lengthen it)")
    ap.add_argument("--workdir",
                    help="working directory; must NOT already exist. "
                         "Defaults to a timestamped directory.")
    ap.add_argument("--cache", default=str(Path.home() / ".boltz"))
    ap.add_argument("--use-msa-server", action="store_true",
                    help="fetch alignments from the MSA server (needs "
                         "network access from this node)")
    ap.add_argument("--msa-server-url", default=DEFAULT_MSA_URL)
    ap.add_argument("--recycling-steps", type=int, default=0,
                    help="recycling steps for the first pass (default 0)")
    ap.add_argument("--skip-recycle-test", action="store_true",
                    help="run only one pass; skips the recycling check")
    ap.add_argument("--model", choices=["boltz1", "boltz2"],
                    default="boltz1",
                    help="which trunk to probe. Default boltz1: at the time "
                         "of writing its published weights load cleanly "
                         "against the published code (0 missing keys), "
                         "whereas the boltz2 checkpoint is missing 128 "
                         "trunk layer-norm parameters that the installed "
                         "class expects -- representations from that pair "
                         "would be partly randomly initialised.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="probe_report.json")
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    if out_path.exists():
        die(f"{out_path} already exists; refusing to overwrite it. "
            f"Pass a different --out.")

    if not args.use_msa_server:
        die("This probe requires an MSA-backed input; Boltz has no implicit\n"
            "single-sequence mode (main.py raises \"Missing MSA's in input "
            "and\n--use_msa_server flag not set\"). Pass --use-msa-server.\n\n"
            "A precomputed-MSA route was removed rather than left in place: "
            "the data\nmodule takes an msa_dir, but process_inputs runs "
            "first and is not passed\none, so the option would have failed "
            "during preprocessing whatever the\ndirectory contained. To add "
            "it, read how this Boltz version consumes\nalignments:\n\n"
            "  python -c \"from boltz.main import process_inputs; import "
            "inspect;\\\n"
            "             print(inspect.getsource(process_inputs)[:6000])\""
            "\n\nthen wire the a3m into the YAML or the call explicitly.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    work = Path(args.workdir) if args.workdir else Path(f"probe_{stamp}")
    if work.exists():
        die(f"working directory {work} already exists; refusing to touch "
            f"it.\nRemove it yourself or pass a different --workdir.")
    work.mkdir(parents=True)

    seq = read_sequence(args)
    if args.truncate:
        if args.truncate > len(seq):
            print(f"note: --truncate {args.truncate} exceeds the sequence "
                  f"length ({len(seq)}); it cannot lengthen the sequence "
                  f"and is ignored.")
        else:
            seq = seq[:args.truncate]

    report = {
        "when": stamp,
        "host": platform.node(),
        "seq_len": len(seq),
        "workdir": str(work),
    }

    print("=" * 70)
    print(f"{args.model.upper()} EXTRACTION PROBE")
    print("=" * 70)

    # --- provenance ----------------------------------------------------
    import torch
    try:
        from importlib.metadata import version as _ver
        boltz_version = _ver("boltz")
    except Exception:  # noqa: BLE001
        boltz_version = "unknown"
    ckpt = Path(args.cache) / f"{args.model}_conf.ckpt"
    if not ckpt.exists():
        die(f"checkpoint not found at {ckpt}")
    ck_stat = ckpt.stat()
    report["provenance"] = {
        "model": args.model,
        "boltz": boltz_version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "checkpoint": str(ckpt),
        "checkpoint_bytes": ck_stat.st_size,
        "checkpoint_mtime": datetime.fromtimestamp(
            ck_stat.st_mtime).isoformat(timespec="seconds"),
    }
    print(f"boltz {boltz_version} | torch {torch.__version__} "
          f"(cuda {torch.version.cuda})")
    print(f"checkpoint {ckpt.name}, {ck_stat.st_size/1e9:.2f} GB")
    print(f"sequence   {len(seq)} residues")
    print(f"workdir    {work}")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        die("CUDA requested but unavailable. Pass --device cpu to probe "
            "on CPU\n(slow, but it does exercise the hooks).")
    if device.startswith("cuda"):
        print(f"device     {torch.cuda.get_device_name(0)}")
        report["device"] = torch.cuda.get_device_name(0)
    else:
        report["device"] = "cpu"

    # --- 1. features ---------------------------------------------------
    print("\n" + "-" * 70)
    print("1. building features")
    print("-" * 70)
    inp = write_input(work, seq, "probe")
    proc_dir = work / "processed"

    from boltz.main import process_inputs

    t0 = time.time()
    try:
        returned = process_inputs(
            data=[inp],
            out_dir=proc_dir,
            ccd_path=Path(args.cache) / "ccd.pkl",
            mol_dir=Path(args.cache) / "mols",
            msa_server_url=args.msa_server_url,
            msa_pairing_strategy="greedy",
            use_msa_server=args.use_msa_server,
            boltz2=(args.model == "boltz2"),
        )
    except Exception as e:  # noqa: BLE001
        report["feature_error"] = f"{type(e).__name__}: {e}"
        out_path.write_text(json.dumps(report, indent=2))
        die(f"feature processing failed: {type(e).__name__}: {e}\n"
            f"Wrote partial report to {out_path}.")
    feat_secs = time.time() - t0
    report["feature_seconds"] = round(feat_secs, 1)

    # process_inputs is annotated "-> Manifest" but returns None in boltz
    # 2.2.1, writing the manifest to disk instead. Trust the filesystem
    # rather than the annotation, and fall back to the return value in
    # case a future version starts honouring it.
    from boltz.data.types import Manifest
    manifest = returned
    if manifest is None:
        cands = sorted(proc_dir.rglob("manifest.json"))
        if not cands:
            report["feature_error"] = "no manifest.json produced"
            out_path.write_text(json.dumps(report, indent=2))
            die(f"process_inputs returned None and wrote no manifest.json "
                f"under\n{proc_dir}. Inspect that directory.")
        manifest = Manifest.load(cands[0])
        report["manifest_path"] = str(cands[0])
        struct_root = cands[0].parent
    else:
        struct_root = proc_dir
    report["struct_root"] = str(struct_root)
    print(f"processed in {feat_secs:.1f} s "
          f"({len(manifest.records)} record(s))")
    print(f"manifest    : {report.get('manifest_path', 'returned in-memory')}")

    # --- 2. model ------------------------------------------------------
    print("\n" + "-" * 70)
    print("2. loading model")
    print("-" * 70)
    # The published weights track Boltz's development branch while the pip
    # release lags, so the checkpoint can carry hyperparameters the
    # installed classes do not accept. Rather than silently passing
    # whatever the checkpoint holds, compare its arguments against the
    # installed signatures and drop only the ones that genuinely do not
    # exist -- reporting each, so a mismatch is never invisible.
    import inspect as _ins
    if args.model == "boltz1":
        from boltz.model.models.boltz1 import Boltz1 as ModelCls
    else:
        from boltz.model.models.boltz2 import Boltz2 as ModelCls

    raw_hp = dict(torch.load(str(ckpt), map_location="cpu",
                             weights_only=False).get("hyper_parameters", {}))
    dropped = {}

    # Top-level kwargs the installed class does not accept. These are
    # typically training-time settings rather than architecture, but each
    # is named so the distinction can be checked rather than assumed.
    top_ok = set(_ins.signature(ModelCls.__init__).parameters)
    for k in sorted(set(raw_hp) - top_ok):
        dropped[k] = raw_hp.pop(k)

    # Nested diffusion args. Only boltz2 checkpoints have carried keys the
    # installed AtomDiffusion rejects, and boltz1 constructs cleanly after
    # top-level filtering alone -- so this does not touch boltz1, where
    # mutating nested args against a v2 class signature would be wrong.
    if args.model == "boltz2":
        try:
            from boltz.model.modules.diffusionv2 import AtomDiffusion
            ad_params = set(
                _ins.signature(AtomDiffusion.__init__).parameters)
            dpa = dict(raw_hp.get("diffusion_process_args", {}))
            extra = sorted(set(dpa) - ad_params)
            if extra:
                for k in extra:
                    dropped[f"diffusion_process_args.{k}"] = dpa.pop(k)
                raw_hp["diffusion_process_args"] = dpa
        except Exception:  # noqa: BLE001
            pass

    report["dropped_hyperparameters"] = dropped
    if dropped:
        print("checkpoint carries hyperparameters this Boltz does not "
              "accept:")
        for k, v in dropped.items():
            print(f"    dropped {k} = {v}")
        print("  These are checkpoint hyperparameters the installed model "
              "constructor\n  does not accept. They are removed only so "
              "the architecture can be\n  built; exact state-dict "
              "agreement is required immediately afterward,\n  so a "
              "removal that did change the architecture would be caught "
              "there.")

    # Construct from the cleaned hyperparameters and load the state dict
    # explicitly. load_from_checkpoint is deliberately NOT used: it reads
    # the checkpoint's own hyper_parameters, so the filtering above would
    # not reliably take effect and the dropped keys could return.
    raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    sd = raw["state_dict"]
    t0 = time.time()
    try:
        model = ModelCls(**raw_hp)
    except Exception as e:  # noqa: BLE001
        report["load_error"] = f"{type(e).__name__}: {e}"
        out_path.write_text(json.dumps(report, indent=2))
        die(f"model construction failed after dropping {len(dropped)} "
            f"unsupported hyperparameter(s):\n  {type(e).__name__}: {e}")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"constructed and loaded in {time.time()-t0:.1f} s")

    report["load"] = {
        "model_keys": len(model.state_dict()),
        "checkpoint_keys": len(sd),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "missing_sample": list(missing)[:5],
        "unexpected_sample": list(unexpected)[:5],
    }
    print(f"state_dict: {len(model.state_dict())} model keys, {len(sd)} "
          f"checkpoint keys")
    print(f"            {len(missing)} missing, {len(unexpected)} unexpected")

    # FATAL on any disagreement. This is the check that matters: the boltz2
    # checkpoint is missing 128 Pairformer layer-norm parameters, which the
    # constructor would initialise randomly. The model would then run, the
    # hooks would fire, and every reported representation would be partly
    # noise while looking entirely plausible. A probe that only warned here
    # would certify exactly the failure it exists to prevent.
    if missing or unexpected:
        out_path.write_text(json.dumps(report, indent=2))
        print("\n  missing (model expects, checkpoint lacks):")
        for k in list(missing)[:5]:
            print(f"    {k}")
        print("  unexpected (checkpoint has, model lacks):")
        for k in list(unexpected)[:5]:
            print(f"    {k}")
        die(f"state_dict mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected.\nParameters the model expects "
            f"but the checkpoint lacks are randomly\ninitialised, so any "
            f"representation taken from this pairing would be\npartly "
            f"noise. Use a model and checkpoint that agree exactly.")
    del raw, sd

    model.eval()
    model.to(device)

    name, pf_mod, layers = find_pairformer(model)
    if layers is None:
        report["error"] = "pairformer not found"
        out_path.write_text(json.dumps(report, indent=2))
        die("could not locate a Pairformer ModuleList on the model")
    n_blocks = len(layers)
    print(f"pairformer: {name}  ({n_blocks} blocks)")
    report["pairformer_attr"] = name
    report["n_blocks"] = n_blocks

    # --- 3. data module ------------------------------------------------
    if args.model == "boltz1":
        from boltz.data.module.inference import (
            BoltzInferenceDataModule as DataModuleCls)
    else:
        from boltz.data.module.inferencev2 import (
            Boltz2InferenceDataModule as DataModuleCls)

    # Directories are taken from wherever the manifest actually landed.
    # boltz 2.2.1 nests a second "processed" level under out_dir, so
    # hard-coding proc_dir/"structures" finds nothing.
    dm_kwargs = {
        "manifest": manifest,
        "target_dir": struct_root / "structures",
        "msa_dir": struct_root / "msa",
        "num_workers": 0,
    }
    # The v2 data module takes a mol_dir; the v1 one does not.
    if "mol_dir" in _ins.signature(DataModuleCls.__init__).parameters:
        dm_kwargs["mol_dir"] = Path(args.cache) / "mols"
    dm = DataModuleCls(**dm_kwargs)
    dm.setup("predict")
    batch = next(iter(dm.predict_dataloader()))
    batch = {k: (v.to(device) if hasattr(v, "to") else v)
             for k, v in batch.items()}
    print(f"batch keys: {len(batch)}")

    # --- 4. the actual hook test ---------------------------------------
    print("\n" + "-" * 70)
    print("3. running forward with hooks live")
    print("-" * 70)

    captures: list[dict] = []

    class _TrunkDone(Exception):
        """Control-flow exception used to stop once the expected number of
        Pairformer captures has arrived.

        The forward pass continues past the trunk into structure sampling,
        which this study never uses and which fails here on a degenerate
        matrix at num_sampling_steps=1 (linalg.svd does not converge).
        Letting it run would discard good trunk captures because of an
        error in a component downstream of them.

        Note this stops on a COUNT, which is not the same as knowing the
        trunk finished -- that is what the per-layer count and order
        checks below establish. If those pass, the count was reached by
        the trunk executing as the recycling model predicts.
        """

    def make_hook(i, stop_after):
        def hook(module, inputs, output):
            rec = {"layer": i, "call": len(captures)}
            try:
                s, z = output[0], output[1]
                rec["s"] = summarize(s)
                rec["z"] = summarize(z)
            except Exception as e:  # noqa: BLE001
                rec["error"] = repr(e)
            # Append BEFORE stopping, including a failed unpack, so the
            # downstream "every capture carries s and z" check sees it and
            # rejects the run rather than the record being lost.
            captures.append(rec)
            if len(captures) >= stop_after:
                raise _TrunkDone
        return hook

    # Inspect the forward signature ONCE and build the call explicitly.
    # An earlier version wrapped the call in `except TypeError: model(batch)`
    # so that signature differences across Boltz versions would not break
    # the probe. That is precisely the wrong trade here: the clause would
    # also swallow a genuine TypeError raised deep inside the model, rerun
    # with default recycling, and report a confident recycling result that
    # measured something else entirely. A missing keyword is now a hard
    # failure, which is visible, rather than a silent substitution.
    import inspect as _inspect
    fwd_params = _inspect.signature(model.forward).parameters
    report["forward_signature"] = list(fwd_params)
    if "recycling_steps" not in fwd_params:
        report["error"] = "forward has no recycling_steps parameter"
        out_path.write_text(json.dumps(report, indent=2))
        die("model.forward does not accept recycling_steps, so the "
            "recycling audit\ncannot be performed safely. Inspect "
            "inspect.signature(model.forward)\nand adapt this probe "
            "deliberately rather than guessing.")

    def run(recycling_steps: int):
        captures.clear()
        # One capture per block per recycle; stop as soon as that many
        # have arrived.
        stop_after = n_blocks * (recycling_steps + 1)
        handles = [layers[i].register_forward_hook(make_hook(i, stop_after))
                   for i in range(n_blocks)]
        kwargs = {"recycling_steps": recycling_steps}
        # One diffusion sample, and at most one in flight. In an earlier
        # Boltz-1 run, leaving max_parallel_samples at its None default
        # produced "unsupported operand type(s) for %: 'int' and
        # 'NoneType'" downstream; setting it explicitly to 1 is consistent
        # with diffusion_samples=1. The exact internal expression was not
        # traced, so this records the observation rather than a diagnosis.
        for opt, val in (("num_sampling_steps", 1),
                         ("diffusion_samples", 1),
                         ("max_parallel_samples", 1)):
            if opt in fwd_params:
                kwargs[opt] = val
        print(f"  forward kwargs: {kwargs}")
        t = time.time()
        try:
            with torch.no_grad():
                model(batch, **kwargs)
        except _TrunkDone:
            pass  # expected: the trunk finished and we stopped on purpose
        finally:
            for h in handles:
                h.remove()
        return list(captures), time.time() - t

    try:
        caps0, secs0 = run(args.recycling_steps)
    except Exception as e:  # noqa: BLE001
        report["forward_error"] = f"{type(e).__name__}: {e}"
        out_path.write_text(json.dumps(report, indent=2))
        die(f"forward pass failed: {type(e).__name__}: {e}\n"
            f"Wrote partial report to {out_path}.")

    if not caps0:
        report["error"] = "no captures"
        out_path.write_text(json.dumps(report, indent=2))
        die("the forward pass ran but no hook fired. The Pairformer stack "
            "found is\nnot the one being executed; inspect "
            "model.named_modules().")

    errs = [c for c in caps0 if "error" in c]
    if errs:
        print(f"WARNING: {len(errs)} capture(s) could not be unpacked, "
              f"e.g. {errs[0]['error']}")

    # Require BOTH tensors, and require every capture to have them. The
    # hook writes s and z together, so a partial capture means a block
    # returned something unexpected -- and a probe meant to establish the
    # extraction contract should not report shapes and byte costs drawn
    # from the subset that happened to unpack.
    good = [c for c in caps0 if "s" in c and "z" in c]
    print(f"captures    : {len(caps0)}  ({len(good)} with both s and z)")
    # No `good and ...` guard here. If NOTHING unpacked, good is empty and
    # a truthiness guard would skip this check entirely -- walking past the
    # total-failure case on the way to reporting storage figures computed
    # from nothing. The contract is that every capture carries s and z.
    if len(good) != len(caps0):
        report["captures"] = len(caps0)
        report["captures_ok"] = len(good)
        report["partial_capture_sample"] = errs[0] if errs else None
        out_path.write_text(json.dumps(report, indent=2))
        die(f"{len(caps0) - len(good)} of {len(caps0)} captures lacked s "
            f"or z.\nEvery Pairformer block must return both for the "
            f"extraction contract to\nhold; a partial capture means some "
            f"block returns a different structure.\nInspect before "
            f"proceeding.")
    print(f"forward time: {secs0:.1f} s at recycling_steps="
          f"{args.recycling_steps}")
    if good:
        print(f"s shape     : {good[0]['s']['shape']}  "
              f"{good[0]['s']['dtype']}")
        print(f"z shape     : {good[0]['z']['shape']}  "
              f"{good[0]['z']['dtype']}")
    report["forward_seconds"] = round(secs0, 2)
    report["captures"] = len(caps0)
    report["captures_ok"] = len(good)
    report["first_capture"] = good[0] if good else None
    report["last_capture"] = good[-1] if good else None

    # --- 5. recycling ---------------------------------------------------
    if not args.skip_recycle_test:
        print("\n" + "-" * 70)
        print("4. recycling check")
        print("-" * 70)
        r2 = args.recycling_steps + 1
        try:
            caps1, secs1 = run(r2)
        except Exception as e:  # noqa: BLE001
            # Fail closed. The probe exists to certify the extraction
            # contract, and a recycling audit that could not be completed
            # certifies nothing. Continuing to a success exit would invite
            # building the extractor on an unverified assumption.
            report["recycling"] = {"error": f"{type(e).__name__}: {e}",
                                   "matches_model": False}
            out_path.write_text(json.dumps(report, indent=2))
            die(f"recycling audit failed on the second pass: "
                f"{type(e).__name__}: {e}\nThe first pass captured "
                f"{len(caps0)} tensors, but without the second pass the\n"
                f"recycling behaviour is unverified. Pass "
                f"--skip-recycle-test only if you\nintend to verify it some "
                f"other way.")

        # Zero captures is its own failure, and needs its own branch: a
        # `if caps1:` guard would skip every downstream check and let the
        # probe exit successfully on the strength of the first pass alone,
        # having tested nothing about recycling.
        if not caps1:
            report["recycling"] = {
                "error": "second pass produced zero Pairformer captures",
                "matches_model": False}
            out_path.write_text(json.dumps(report, indent=2))
            die("the second recycling pass completed but produced zero "
                "Pairformer\ncaptures. Recycling behaviour therefore "
                "cannot be certified.")

        # The second pass must satisfy the same contract as the first.
        # Counting executions alone would let "96 hooks fired, 96
        # predicted, 0 unpacked" pass the recycling check, certifying a
        # model whose blocks return something other than (s, z).
        good1 = [c for c in caps1 if "s" in c and "z" in c]
        if len(good1) != len(caps1):
            report["recycling_capture_errors"] = len(caps1) - len(good1)
            out_path.write_text(json.dumps(report, indent=2))
            die(f"the second recycling pass produced {len(caps1)} captures "
                f"but only\n{len(good1)} unpacked as (s, z). The capture "
                f"count alone cannot certify\nthe extraction contract when "
                f"the outputs themselves are wrong.")

        # PER-LAYER counts, not just totals. A total of 48 is also
        # consistent with one layer firing twice and another not at all,
        # which would mean the extractor's "keep the last capture per
        # layer" rule silently drops a block. Every capture already
        # records its layer index, so this costs nothing.
        from collections import Counter
        counts0 = Counter(c["layer"] for c in caps0)
        counts1 = Counter(c["layer"] for c in caps1)
        exp0, exp1 = args.recycling_steps + 1, r2 + 1
        bad0 = {i: counts0[i] for i in range(n_blocks)
                if counts0[i] != exp0}
        bad1 = {i: counts1[i] for i in range(n_blocks)
                if counts1[i] != exp1}
        if bad0 or bad1:
            report["recycling"] = {"per_layer_bad_pass1": bad0,
                                   "per_layer_bad_pass2": bad1,
                                   "matches_model": False}
            out_path.write_text(json.dumps(report, indent=2))
            die(f"per-layer firing counts disagree with expectation "
                f"({exp0} and {exp1}\nper layer).\n  pass 1 bad layers: "
                f"{bad0}\n  pass 2 bad layers: {bad1}\n\nThe extractor "
                f"selects one capture per layer per recycle; a layer that "
                f"fires\nthe wrong number of times breaks that rule.")

        # Call ORDER within each recycle. The extractor keeps the final
        # recycle, which is only well defined if each pass runs the blocks
        # in index order 0..n-1.
        order_ok = True
        for pass_caps, nrec in ((caps0, exp0), (caps1, exp1)):
            for r in range(nrec):
                chunk = [c["layer"] for c in pass_caps
                         [r * n_blocks:(r + 1) * n_blocks]]
                if chunk != list(range(n_blocks)):
                    order_ok = False
        report["recycling_order_sequential"] = order_ok
        print(f"per-layer counts: exactly {exp0} and {exp1} per block")
        print(f"call order      : "
              f"{'sequential 0..' + str(n_blocks - 1) + ' per recycle' if order_ok else 'NOT sequential'}")
        if not order_ok:
            print("  Captures do not arrive in block order within a "
                  "recycle. Grouping by\n  position would then mis-assign "
                  "layers; the extractor must group by the\n  recorded "
                  "layer index instead.")

        if caps1:
            obs0, obs1 = len(caps0), len(caps1)
            pred0 = n_blocks * (args.recycling_steps + 1)
            pred1 = n_blocks * (r2 + 1)
            print(f"recycling_steps={args.recycling_steps}: "
                  f"{obs0} captures (predicted {pred0})")
            print(f"recycling_steps={r2}: "
                  f"{obs1} captures (predicted {pred1})")
            ok = (obs0 == pred0 and obs1 == pred1)
            report["recycling"] = {
                "observed": {str(args.recycling_steps): obs0, str(r2): obs1},
                "predicted": {str(args.recycling_steps): pred0,
                              str(r2): pred1},
                "matches_model": ok,
                "seconds": {str(args.recycling_steps): round(secs0, 2),
                            str(r2): round(secs1, 2)},
            }
            if ok:
                print("\n  Capture counts scale as n_blocks x (recycles+1), "
                      "so the extractor\n  must tag each capture with its "
                      "recycle and keep only the last.")
            else:
                out_path.write_text(json.dumps(report, indent=2))
                die("Pairformer capture counts do not match n_blocks x "
                    "(recycling_steps + 1).\n"
                    f"  observed  {obs0} and {obs1}\n"
                    f"  predicted {pred0} and {pred1}\n\n"
                    "Do not build the extractor on the assumed recycling "
                    "model. Keeping the\nwrong capture per layer would "
                    "yield representations from an early\nrecycle, which "
                    "look entirely plausible and are wrong. Understand the\n"
                    "discrepancy first.")

    # --- 6. storage, from measurements ----------------------------------
    if good:
        print("\n" + "-" * 70)
        print("5. storage, measured")
        print("-" * 70)
        s_b = good[0]["s"]["bytes"]
        z_b = good[0]["z"]["bytes"]
        native = good[0]["s"]["dtype"]
        native_z = good[0]["z"]["dtype"]
        # fp16 sizes are computed from ELEMENT COUNTS, not by rescaling the
        # native byte totals. Rescaling needs a per-tensor element size, and
        # inferring one from s and applying it to z would misreport z
        # whenever the two tensors have different dtypes. Element count x 2
        # is correct regardless.
        import math as _math
        s_numel = _math.prod(good[0]["s"]["shape"])
        z_numel = _math.prod(good[0]["z"]["shape"])
        half = {"s_all": s_numel * 2 * n_blocks,
                "z_all": z_numel * 2 * n_blocks}
        report["storage_measured"] = {
            "native_dtype_s": native,
            "native_dtype_z": native_z,
            "s_per_layer_bytes": s_b,
            "z_per_layer_bytes": z_b,
            "s_all_layers_bytes": s_b * n_blocks,
            "z_all_layers_bytes": z_b * n_blocks,
            "s_all_layers_fp16_bytes": int(half["s_all"]),
            "z_all_layers_fp16_bytes": int(half["z_all"]),
        }
        print(f"native dtype: s {native}, z {native_z}")
        print(f"  s: {s_b/1e6:.2f} MB/layer, "
              f"{s_b*n_blocks/1e6:.1f} MB across {n_blocks} layers")
        print(f"  z: {z_b/1e6:.2f} MB/layer, "
              f"{z_b*n_blocks/1e6:.1f} MB across {n_blocks} layers")
        print(f"  if saved as fp16: s {half['s_all']/1e6:.1f} MB, "
              f"z {half['z_all']/1e6:.1f} MB")
        print(f"\n  z grows with the square of length. This protein is "
              f"{len(seq)} residues;\n  a 400-residue target costs roughly "
              f"{(400/len(seq))**2:.0f}x as much for z.")

    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")
    print(f"Work directory kept at {work} (delete it yourself).")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
