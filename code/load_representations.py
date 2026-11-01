#!/usr/bin/env python3
"""
Read representations written by extract_representations.py.

Use this rather than np.load directly. The arrays are stored as uint16
holding bfloat16 BITS, so reading them without the reinterpretation gives
integers in the tens of thousands that look like plausible activations
and are not. The `dtype` field in each file records the encoding; this
loader checks it rather than assuming.

bfloat16 is used because fp16 came within a factor of four of overflowing
in the five-system smoke test, with the maxima at the final block in
every system and not scaling with protein length -- so neither depth nor
size gave a bound to reason from. bfloat16 keeps the same two bytes and
float32's exponent range, at the cost of relative precision that does not
survive the standardisation applied before probing.

USAGE
-----
    from load_representations import load_system, iter_systems

    rep = load_system("reps/systems/7t9z__2__1.B__1.E.npz")
    rep["s"]       # (48, L, 384) float32
    rep["z_pool"]  # (48, L, 384) float32
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

STORAGE_DTYPE = "bfloat16"
_POOL_PARTS = ("row_mean", "col_mean", "self_pair_diagonal")

# The representation contract, fixed by the trunk and the pooling
# definition. Stated here so a wrong trunk or a changed pooling scheme
# fails at read time rather than producing arrays of the wrong width
# that every downstream step accepts.
S_WIDTH = 384          # Boltz-1 per-residue track
Z_CHANNEL = 128        # pair-feature width
POOL_WIDTH = 3 * Z_CHANNEL   # row mean | column mean | self-pair diagonal


def _to_float32(bits: np.ndarray) -> np.ndarray:
    """uint16 bfloat16 bits -> float32.

    bfloat16 is the top 16 bits of a float32, so the conversion is a
    left shift into the high half rather than a numeric cast. Casting
    instead -- `bits.astype(np.float32)` -- silently yields values in the
    tens of thousands, which is why this function exists and why callers
    should not open these files by hand.
    """
    if bits.dtype != np.uint16:
        raise TypeError(f"expected uint16 bits, got {bits.dtype}")
    return (bits.astype(np.uint32) << 16).view(np.float32)


def load_system(path, arrays=("s", "z_pool")) -> dict:
    """Load one system's representations, decoded to float32."""
    path = Path(path)
    out = {}
    with np.load(path, allow_pickle=False) as d:
        if "dtype" not in d.files:
            raise ValueError(
                f"{path.name} has no dtype field. It predates the "
                f"bfloat16 change and its arrays are float16; it must be "
                f"regenerated rather than read with this loader.")
        stored = str(np.asarray(d["dtype"]).reshape(-1)[0])
        if stored != STORAGE_DTYPE:
            raise ValueError(
                f"{path.name} was written as {stored}, this loader reads "
                f"{STORAGE_DTYPE}")

        out["length"] = int(np.asarray(d["length"]).reshape(-1)[0])
        out["layers"] = np.asarray(d["layers"])
        if "chain_order" in d.files:
            out["chain_order"] = [str(c) for c in d["chain_order"]]

        for name in arrays:
            # A requested array that is absent is an error, not a
            # silently shorter result: a caller asking for z_pool and
            # receiving only s would proceed on the wrong features.
            if name not in d.files:
                raise KeyError(
                    f"{path.name}: requested array '{name}' is not in "
                    f"this file (has {sorted(d.files)})")
            out[name] = _to_float32(d[name])

    # Shape agreement. The probe is per-residue and per-layer, so a file
    # whose arrays disagree about either would misalign features against
    # labels while looking entirely well-formed.
    n_layers = len(out["layers"])
    L = out["length"]
    for name, width in (("s", S_WIDTH), ("z_pool", POOL_WIDTH)):
        if name not in out:
            continue
        a = out[name]
        if a.ndim != 3:
            raise ValueError(f"{path.name}: {name} has {a.ndim} "
                             f"dimensions, expected 3")
        if a.shape[0] != n_layers:
            raise ValueError(f"{path.name}: {name} has {a.shape[0]} "
                             f"layers, but 'layers' records {n_layers}")
        if a.shape[1] != L:
            raise ValueError(f"{path.name}: {name} has {a.shape[1]} "
                             f"residues, but 'length' records {L}")
        if a.shape[2] != width:
            raise ValueError(f"{path.name}: {name} width {a.shape[2]}, "
                             f"expected {width}. A different trunk or a "
                             f"changed pooling scheme would do this.")
    if "s" in out and "z_pool" in out:
        if out["s"].shape[:2] != out["z_pool"].shape[:2]:
            raise ValueError(f"{path.name}: s and z_pool disagree on "
                             f"layers or residues")
    if "z_final" in out:
        zf = out["z_final"]
        if zf.shape != (L, L, Z_CHANNEL):
            raise ValueError(f"{path.name}: z_final shape {zf.shape}, "
                             f"expected ({L}, {L}, {Z_CHANNEL})")

    # Cheap sanity check: a decoding mistake shows up here as values that
    # are enormous or non-finite, before they reach a probe.
    for name in ("s", "z_pool", "z_final"):
        if name in out:
            a = out[name]
            if not np.isfinite(a).all():
                # Not necessarily an overflow: bfloat16 carries float32's
                # exponent range, so a non-finite value more likely means
                # the stored representation is invalid.
                raise ValueError(
                    f"{path.name}: {name} contains non-finite values "
                    f"after decoding; the stored representation is "
                    f"invalid")
    return out


def split_pool(z_pool: np.ndarray) -> dict:
    """Split the pooled pair summary into its three channel groups.

    z_pool is the concatenation of the row mean, the column mean and the
    self-pair diagonal, in that order. Keeping them separable matters
    because they are not interchangeable: the diagonal is not recoverable
    from either marginal.
    """
    if z_pool.ndim < 1:
        raise ValueError("z_pool must have at least one dimension")
    w = z_pool.shape[-1]
    # Without this check a width not divisible by three returns three
    # arrays and silently drops the remainder -- a (2, 5) input yields
    # three (2, 1) arrays, discarding two channels, with no error.
    if w != POOL_WIDTH:
        raise ValueError(
            f"pooled z width {w}, expected {POOL_WIDTH} "
            f"({Z_CHANNEL} row mean + {Z_CHANNEL} column mean + "
            f"{Z_CHANNEL} self-pair diagonal)")
    c = w // 3
    return {name: z_pool[..., i * c:(i + 1) * c]
            for i, name in enumerate(_POOL_PARTS)}


def _check_layer_tensor(path, name, a, layer, expected_width, n_layers,
                        L):
    """The same contract load_system enforces, applied per layer.

    load_layer is what the probe and null scripts call, so without this
    they would bypass every check load_system performs and a malformed
    file would reach a probe with its shape unexamined.
    """
    if a.ndim != 3:
        raise ValueError(f"{path.name}: {name} has {a.ndim} dimensions, "
                         f"expected 3")
    if not (0 <= layer < a.shape[0]):
        raise IndexError(f"{path.name}: layer {layer} outside "
                         f"0..{a.shape[0] - 1}")
    if a.shape[0] != n_layers:
        raise ValueError(f"{path.name}: {name} has {a.shape[0]} layers, "
                         f"but 'layers' records {n_layers}")
    if a.shape[1] != L:
        raise ValueError(f"{path.name}: {name} has {a.shape[1]} "
                         f"residues, but 'length' records {L}")
    if a.shape[2] != expected_width:
        raise ValueError(f"{path.name}: {name} width {a.shape[2]}, "
                         f"expected {expected_width}")


def load_layer(path, layer: int, which: str) -> np.ndarray:
    """One layer's features for one system, decoded to float32.

    The canonical per-layer reader. The probe and null scripts import
    this rather than carrying their own copy of the decode expression:
    one definition of how a stored representation becomes a number means
    a correction here reaches every analysis, which a duplicated
    expression would not.

    `which` is "s", "zpool", or "both" (the two concatenated along the
    feature axis, s first).
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as d:
        if "dtype" not in d.files:
            raise ValueError(
                f"{path.name} has no dtype field. It predates the "
                f"bfloat16 change and must be regenerated.")
        stored = str(np.asarray(d["dtype"]).reshape(-1)[0])
        if stored != STORAGE_DTYPE:
            raise ValueError(f"{path.name} was written as {stored}, "
                             f"this reader expects {STORAGE_DTYPE}")
        L = int(np.asarray(d["length"]).reshape(-1)[0])
        n_layers = len(np.asarray(d["layers"]))

        if which == "s":
            a = d["s"]
            _check_layer_tensor(path, "s", a, layer, S_WIDTH, n_layers, L)
            x = _to_float32(a[layer])
        elif which == "zpool":
            a = d["z_pool"]
            _check_layer_tensor(path, "z_pool", a, layer, POOL_WIDTH,
                                n_layers, L)
            x = _to_float32(a[layer])
        elif which == "both":
            sa, za = d["s"], d["z_pool"]
            _check_layer_tensor(path, "s", sa, layer, S_WIDTH, n_layers, L)
            _check_layer_tensor(path, "z_pool", za, layer, POOL_WIDTH,
                                n_layers, L)
            if sa.shape[:2] != za.shape[:2]:
                raise ValueError(f"{path.name}: s and z_pool disagree on "
                                 f"layers or residues")
            x = np.concatenate([_to_float32(sa[layer]),
                                _to_float32(za[layer])], axis=1)
        else:
            raise ValueError(f"unknown feature set '{which}'; expected "
                             f"'s', 'zpool' or 'both'")

    if not np.isfinite(x).all():
        raise ValueError(f"{path.name}: {which} layer {layer} contains "
                         f"non-finite values after decoding")
    return x


def iter_systems(repdir, arrays=("s", "z_pool")):
    """Yield (system_id, representations) for every file in a directory."""
    for f in sorted(Path(repdir).glob("*.npz")):
        yield f.stem, load_system(f, arrays=arrays)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit("usage: load_representations.py <file.npz>")
    rep = load_system(sys.argv[1])
    print(f"length: {rep['length']}  layers: {len(rep['layers'])}")
    for k in ("s", "z_pool", "z_final"):
        if k in rep:
            a = rep[k]
            print(f"  {k:8} {a.shape} {a.dtype}  "
                  f"range [{a.min():.3g}, {a.max():.3g}]")
    if "z_pool" in rep:
        for name, part in split_pool(rep["z_pool"]).items():
            print(f"    {name:20} {part.shape}  "
                  f"|max| {np.abs(part).max():.4g}")
