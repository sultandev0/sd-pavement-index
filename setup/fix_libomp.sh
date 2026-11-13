#!/usr/bin/env bash
# setup/fix_libomp.sh — make torch use the conda env's OpenMP runtime (one libomp.dylib for numpy, sklearn, torch).
#
# Why: the pip torch wheel ships its own torch/lib/libomp.dylib, while conda-forge numpy (OpenBLAS, openmp
# build) and scikit-learn load $CONDA_PREFIX/lib/libomp.dylib. Two runtimes in one process abort with
# "OMP: Error #15". Fix: keep torch's copy as a backup and point torch at the env's copy instead.
#
# Run from the activated `pavement` env:   bash setup/fix_libomp.sh
# Re-run after any `pip install/upgrade torch` (pip restores torch's own copy). Safe to run twice.
# Touches only files inside the pavement env (no sudo, no system files, never conda base).
#
# Undo (run inside torch/lib, path printed below):
#   rm libomp.dylib && mv libomp.dylib.torch-bundled.bak libomp.dylib
set -euo pipefail

# 1) refuse to run anywhere but the pavement env
if [ "${CONDA_DEFAULT_ENV:-}" != "pavement" ] || [ -z "${CONDA_PREFIX:-}" ]; then
    echo "STOP: activate the env first (conda activate pavement). Current env: '${CONDA_DEFAULT_ENV:-none}'" >&2
    exit 1
fi

ENV_OMP="$CONDA_PREFIX/lib/libomp.dylib"
# locate torch WITHOUT importing it (importing torch is what crashes before the fix)
TORCH_LIB="$(python -c 'import importlib.util, os; s = importlib.util.find_spec("torch"); print(os.path.join(os.path.dirname(s.origin), "lib") if s else "")')"
if [ -z "$TORCH_LIB" ] || [ ! -d "$TORCH_LIB" ]; then
    echo "STOP: torch is not installed in this env." >&2; exit 1
fi
if [ ! -e "$ENV_OMP" ]; then
    echo "STOP: $ENV_OMP not found (expected from conda-forge llvm-openmp)." >&2; exit 1
fi
TORCH_OMP="$TORCH_LIB/libomp.dylib"
BAK="$TORCH_LIB/libomp.dylib.torch-bundled.bak"
echo "env OpenMP : $ENV_OMP"
echo "torch/lib  : $TORCH_LIB"

# 2) apply the fix (or report it is already applied)
if [ -L "$TORCH_OMP" ]; then
    # compare fully resolved paths, so an absolute or relative link to the env copy both count
    REAL() { python -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"; }
    if [ "$(REAL "$TORCH_OMP")" = "$(REAL "$ENV_OMP")" ]; then
        echo "already applied: torch/lib/libomp.dylib -> env copy (no change made)"
    else
        echo "STOP: torch/lib/libomp.dylib is a link to something unexpected: $(readlink "$TORCH_OMP")" >&2
        exit 1
    fi
elif [ -f "$TORCH_OMP" ]; then
    # a real file = torch's bundled copy (fresh install or after a pip upgrade): back it up, then link
    mv -f "$TORCH_OMP" "$BAK"
    ln -s "$ENV_OMP" "$TORCH_OMP"
    echo "applied: torch's copy saved as $(basename "$BAK"); torch/lib/libomp.dylib -> env copy"
elif [ -f "$BAK" ]; then
    ln -s "$ENV_OMP" "$TORCH_OMP"
    echo "applied: link re-created (backup already present)"
else
    echo "STOP: neither torch/lib/libomp.dylib nor its backup exists; check the torch install." >&2
    exit 1
fi

# 3) verify: all three libraries import together, and report mps
python -c "import numpy, sklearn, torch; print('verify OK | numpy', numpy.__version__, '| torch', torch.__version__, '| mps:', torch.backends.mps.is_available())"
