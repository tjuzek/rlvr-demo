"""
Create synthetically corrupted code examples for RLVR presentation.

Takes correct MBPP solutions and applies systematic, deterministic
corruptions — each producing broken code that fails the verifier.

This demonstrates:
  1. What "bad" code looks like (before RLVR training)
  2. Why verifiable rewards matter (the verifier catches these bugs)
  3. The gap RLVR aims to close

Corruption types:
  - off_by_one:     Change +1 to +2, -1 to -2, etc.
  - wrong_operator: Swap + for -, * for /, == for !=, etc.
  - wrong_return:   Return wrong value (None, 0, empty, swapped args)
  - missing_edge:   Remove if/elif clauses that handle edge cases
  - wrong_variable: Swap variable names in expressions
  - wrong_index:    Change list/string indices (0 -> 1, -1 -> -2, etc.)

Usage:
    python create_corruptions.py
"""

import copy
import json
import random
import re
from pathlib import Path

SEED = 42
OUTPUT_DIR = Path(__file__).parent / "data"


# ---- Corruption functions ----
# Each takes source code and returns (corrupted_code, description) or None

def corrupt_off_by_one(code: str) -> tuple[str, str] | None:
    """Change numeric literals by +/- 1."""
    # Find patterns like + 1, - 1, range(n), [:n], [n:]
    patterns = [
        (r'(\+ )(\d+)', lambda m: f"{m.group(1)}{int(m.group(2)) + 1}"),
        (r'(\- )(\d+)', lambda m: f"{m.group(1)}{int(m.group(2)) + 1}"),
        (r'(range\()(\d+)', lambda m: f"{m.group(1)}{int(m.group(2)) - 1}"),
        (r'(\[:)(\d+)', lambda m: f"{m.group(1)}{int(m.group(2)) + 1}"),
        (r'(\[)(\d+)(:])', lambda m: f"{m.group(1)}{int(m.group(2)) + 1}{m.group(3)}"),
    ]
    for pattern, replacement in patterns:
        if re.search(pattern, code):
            new_code = re.sub(pattern, replacement, code, count=1)
            if new_code != code:
                return new_code, "off_by_one"
    return None


def corrupt_wrong_operator(code: str) -> tuple[str, str] | None:
    """Swap arithmetic or comparison operators."""
    swaps = [
        (' + ', ' - '),
        (' - ', ' + '),
        (' * ', ' + '),
        (' >= ', ' > '),
        (' <= ', ' < '),
        (' == ', ' != '),
        (' and ', ' or '),
        (' or ', ' and '),
    ]
    for old, new in swaps:
        if old in code:
            # Only swap first occurrence
            new_code = code.replace(old, new, 1)
            if new_code != code:
                return new_code, "wrong_operator"
    return None


def corrupt_wrong_return(code: str) -> tuple[str, str] | None:
    """Replace return value with a wrong one."""
    lines = code.split('\n')
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('return ') and len(stripped) > 7:
            indent = line[:len(line) - len(line.lstrip())]
            return_val = stripped[7:]

            # Choose a corruption based on what's being returned
            if return_val in ('True', 'False'):
                new_val = 'False' if return_val == 'True' else 'True'
            elif return_val.isdigit():
                new_val = str(int(return_val) + 1)
            elif return_val.startswith('['):
                new_val = '[]'
            elif return_val.startswith('"') or return_val.startswith("'"):
                new_val = '""'
            else:
                new_val = 'None'

            lines[i] = f"{indent}return {new_val}"
            return '\n'.join(lines), "wrong_return"
    return None


def corrupt_missing_edge_case(code: str) -> tuple[str, str] | None:
    """Remove an if/elif block that handles an edge case."""
    lines = code.split('\n')

    # Find if blocks that look like edge case handlers
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith('if ') and stripped.endswith(':')
                and any(kw in stripped for kw in ['== 0', '== 1', 'is None',
                        '== []', '== ""', "== ''", 'not ', 'len('])):
            # Remove this if block and its body
            indent_level = len(line) - len(line.lstrip())
            end = i + 1
            while end < len(lines):
                next_line = lines[end]
                if next_line.strip() == '':
                    end += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent > indent_level:
                    end += 1
                else:
                    break

            if end > i + 1:
                new_lines = lines[:i] + lines[end:]
                new_code = '\n'.join(new_lines)
                if new_code.strip():
                    return new_code, "missing_edge_case"
    return None


def corrupt_wrong_index(code: str) -> tuple[str, str] | None:
    """Change list/string indices."""
    patterns = [
        (r'\[0\]', '[1]'),
        (r'\[-1\]', '[-2]'),
        (r'\[1\]', '[0]'),
        (r'\[i\]', '[i+1]'),
    ]
    for pattern, replacement in patterns:
        if re.search(pattern, code):
            new_code = re.sub(pattern, replacement, code, count=1)
            if new_code != code:
                return new_code, "wrong_index"
    return None


ALL_CORRUPTIONS = [
    corrupt_off_by_one,
    corrupt_wrong_operator,
    corrupt_wrong_return,
    corrupt_missing_edge_case,
    corrupt_wrong_index,
]


def corrupt_example(example: dict, rng: random.Random) -> dict | None:
    """Apply a random corruption to an example's reference code.

    Returns the corrupted example or None if no corruption worked.
    """
    code = example["reference_code"]

    # Shuffle corruption order and try each one
    corruptions = list(ALL_CORRUPTIONS)
    rng.shuffle(corruptions)

    for corrupt_fn in corruptions:
        result = corrupt_fn(code)
        if result is not None:
            corrupted_code, corruption_type = result
            return {
                **example,
                "corrupted_code": corrupted_code,
                "corruption_type": corruption_type,
                "original_code": code,
            }

    return None


def main():
    rng = random.Random(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load training data
    train_path = OUTPUT_DIR / "code_rlvr_train.json"
    data = json.load(open(train_path))
    print(f"Loaded {len(data)} training examples")

    # Apply corruptions
    corrupted = []
    corruption_counts = {}
    for ex in data:
        result = corrupt_example(ex, rng)
        if result is not None:
            corrupted.append(result)
            ct = result["corruption_type"]
            corruption_counts[ct] = corruption_counts.get(ct, 0) + 1

    print(f"\nCorrupted {len(corrupted)}/{len(data)} examples")
    print("\nCorruption distribution:")
    for ct, count in sorted(corruption_counts.items(), key=lambda x: -x[1]):
        print(f"  {ct:25s}: {count}")

    # Verify that corruptions actually break the code
    from verifier import verify_code

    actually_broken = 0
    still_passes = 0
    for ex in corrupted:
        r = verify_code(
            ex["corrupted_code"],
            ex["ground_truth"],
            ex.get("test_imports"),
            ex.get("test_setup_code", ""),
        )
        if not r["passed"]:
            actually_broken += 1
        else:
            still_passes += 1

    print(f"\nVerification:")
    print(f"  Actually broken: {actually_broken} ({actually_broken/len(corrupted)*100:.0f}%)")
    print(f"  Still passes:    {still_passes} ({still_passes/len(corrupted)*100:.0f}%)")

    # Only keep corruptions that actually break the tests
    final = [ex for ex in corrupted
             if not verify_code(
                 ex["corrupted_code"],
                 ex["ground_truth"],
                 ex.get("test_imports"),
                 ex.get("test_setup_code", ""),
             )["passed"]]

    print(f"  Kept (verified broken): {len(final)}")

    # Save
    out_path = OUTPUT_DIR / "code_rlvr_corrupted.json"
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out_path}")

    # Show a few examples
    print("\n" + "=" * 70)
    print("EXAMPLE CORRUPTIONS")
    print("=" * 70)
    for ex in final[:5]:
        print(f"\n--- Task {ex['task_id']} ({ex['corruption_type']}) ---")
        print(f"Problem: {ex['messages'][0]['content'][:100]}...")
        print(f"\nOriginal (correct):")
        for line in ex["original_code"].split("\n")[:8]:
            print(f"  {line}")
        print(f"\nCorrupted (broken):")
        for line in ex["corrupted_code"].split("\n")[:8]:
            print(f"  {line}")
        print()


if __name__ == "__main__":
    main()
