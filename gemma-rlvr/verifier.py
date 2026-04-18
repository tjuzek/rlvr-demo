"""
Math answer verifier for RLVR on GSM8K.

Extracts the final numeric answer from a completion and compares it
to the ground-truth number. Returns a binary reward.

This is the 'V' in RLVR for math: no code execution, just regex + float
comparison. Much faster than the code verifier (no subprocess).

Extraction strategy (in order):
  1. Canonical GSM8K marker:  "#### 42"
  2. CoT-style tail:           "So the answer is 42."
  3. LaTeX boxed form:         "\\boxed{42}"
  4. Fallback:                 last number in the completion

All captured numbers are comma-stripped and compared as floats with a
small tolerance so "72" and "72.0" match.
"""

import re

_GSM_MARKER = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)")
_ANSWER_IS = re.compile(r"answer\s+is\s*:?\s*\$?([-+]?\d[\d,]*\.?\d*)", re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\{\s*([-+]?\d[\d,]*\.?\d*)\s*\}")
_ANY_NUMBER = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def extract_answer(text: str) -> str | None:
    """Return the first extractable answer as a comma-stripped string, or None."""
    if not text:
        return None

    for pattern in (_GSM_MARKER, _ANSWER_IS, _BOXED):
        m = pattern.search(text)
        if m:
            return m.group(1).replace(",", "").rstrip(".")

    matches = _ANY_NUMBER.findall(text)
    if matches:
        return matches[-1].replace(",", "").rstrip(".")

    return None


def _as_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def verify_answer(completion: str, ground_truth: str, tol: float = 1e-4) -> dict:
    """Score a single model completion against ground truth.

    Returns:
        dict with keys:
          - passed: bool
          - reward: float (1.0 or 0.0)
          - predicted: str | None
          - expected: str
          - error: str | None (short description of the mismatch)
    """
    predicted = extract_answer(completion)
    expected_str = str(ground_truth).replace(",", "").strip()

    if predicted is None:
        return {
            "passed": False,
            "reward": 0.0,
            "predicted": None,
            "expected": expected_str,
            "error": "no_number_found",
        }

    p = _as_float(predicted)
    e = _as_float(expected_str)

    if p is None or e is None:
        passed = predicted == expected_str
    else:
        passed = abs(p - e) < tol

    return {
        "passed": passed,
        "reward": 1.0 if passed else 0.0,
        "predicted": predicted,
        "expected": expected_str,
        "error": None if passed else f"got {predicted!r}, expected {expected_str!r}",
    }


if __name__ == "__main__":
    print("=== Math Verifier Self-Test ===\n")

    cases = [
        ("Natalia sold 48/2 = 24 clips in May. 48+24 = 72. So the answer is 72.", "72", True),
        ("In total: 48 + 24 = **72 clips**.", "72", True),
        ("Step 1: 5+3=8. Step 2: 8*2=16.\n#### 16", "16", True),
        ("The answer is 1,234,567.", "1234567", True),
        ("answer is 42.0", "42", True),
        ("\\boxed{17}", "17", True),
        ("So the answer is 100.", "72", False),
        ("I don't know.", "42", False),
        ("The price rose from $5 to $10. The answer is $10.", "10", True),
    ]

    failures = 0
    for text, truth, should_pass in cases:
        r = verify_answer(text, truth)
        ok = r["passed"] == should_pass
        tag = "OK " if ok else "XX "
        print(f"  {tag} expect passed={should_pass:5}  got={r['passed']:5}  "
              f"pred={r['predicted']!r:>10}  truth={truth!r}")
        if not ok:
            failures += 1
            print(f"      full result: {r}")

    print()
    if failures:
        print(f"FAILED: {failures} case(s)")
        raise SystemExit(1)
    print("All verifier tests passed.")
