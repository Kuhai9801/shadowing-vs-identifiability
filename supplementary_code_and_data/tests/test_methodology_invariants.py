import ast
from pathlib import Path


def _collect_calls(tree: ast.AST):
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                calls.append(fn.id)
            elif isinstance(fn, ast.Attribute):
                # e.g., torch.atan2 -> "torch.atan2"
                parts = []
                cur = fn
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                calls.append(".".join(reversed(parts)))
    return calls


def test_neural_loss_contains_no_wrap_or_inverse_maps():
    """Static check: the neural solver must not call frac/mod/atan2-like inverses.

    This test is intentionally conservative: it checks for the presence of disallowed
    function calls in solvers/neural_4dvar.py. It does *not* attempt to reason about
    arithmetic modulo used for RNG seeds.
    """
    path = Path(__file__).resolve().parents[1] / "solvers" / "neural_4dvar.py"
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = set(_collect_calls(tree))

    # Disallowed: inverse angle maps (introduce discontinuities) and torus projection.
    banned = {
        "atan2",
        "math.atan2",
        "np.arctan2",
        "numpy.arctan2",
        "torch.atan2",
        "arctan2",
        "frac",
        "torch.frac",
        "torch.remainder",
        "torch.fmod",
    }
    assert calls.isdisjoint(banned), f"Disallowed calls found in neural solver: {sorted(calls & banned)}"


def test_neural_solver_uses_embedding_maps():
    """Static check: E and E_x must be present as calls in the neural solver."""
    path = Path(__file__).resolve().parents[1] / "solvers" / "neural_4dvar.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = set(_collect_calls(tree))
    assert "E" in calls
    assert "E_x" in calls
