"""
Code execution verifier for RLVR.

Takes generated code and test assertions, executes them in an isolated
subprocess with a timeout, and returns a binary reward signal.

This is the verifiable reward function — the 'V' in RLVR.
"""

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


TIMEOUT_SECONDS = 10


def verify_code(
    generated_code: str,
    test_assertions: list[str],
    test_imports: list[str] | None = None,
    test_setup_code: str = "",
    timeout: int = TIMEOUT_SECONDS,
) -> dict:
    """Execute generated code against test assertions.

    Returns:
        dict with keys:
            - passed: bool — did all tests pass?
            - reward: float — 1.0 if passed, 0.0 if not
            - error: str | None — error message if failed
            - num_tests: int — total number of test assertions
            - num_passed: int — how many passed before failure
    """
    # Build the test script
    parts = []

    # Imports needed by tests
    if test_imports:
        for imp in test_imports:
            parts.append(imp)

    # Setup code (e.g., helper class definitions needed by tests)
    if test_setup_code:
        parts.append(test_setup_code)

    # The generated code itself
    parts.append(generated_code)

    # Run each assertion, counting successes
    parts.append("")
    parts.append("_passed = 0")
    parts.append(f"_total = {len(test_assertions)}")
    for i, assertion in enumerate(test_assertions):
        # Wrap each assertion to count passes
        parts.append("try:")
        parts.append(f"    {assertion}")
        parts.append("    _passed += 1")
        parts.append("except Exception as _e:")
        parts.append(f'    print("FAIL test {i}:", repr(_e))')
    parts.append('print(f"RESULT: {_passed}/{_total}")')

    script = "\n".join(parts)

    # Write to temp file and execute in subprocess
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        # Parse result
        if result.returncode != 0:
            return {
                "passed": False,
                "reward": 0.0,
                "error": stderr[-500:] if stderr else f"Exit code {result.returncode}",
                "num_tests": len(test_assertions),
                "num_passed": 0,
            }

        # Find RESULT line
        for line in stdout.split("\n"):
            if line.startswith("RESULT:"):
                parts = line.split(": ")[1].split("/")
                num_passed = int(parts[0])
                num_total = int(parts[1])
                return {
                    "passed": num_passed == num_total,
                    "reward": 1.0 if num_passed == num_total else 0.0,
                    "error": None if num_passed == num_total else stdout,
                    "num_tests": num_total,
                    "num_passed": num_passed,
                }

        return {
            "passed": False,
            "reward": 0.0,
            "error": f"No RESULT line in output: {stdout[:200]}",
            "num_tests": len(test_assertions),
            "num_passed": 0,
        }

    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "reward": 0.0,
            "error": f"Timeout after {timeout}s",
            "num_tests": len(test_assertions),
            "num_passed": 0,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def extract_code_from_response(response: str) -> str:
    """Extract Python code from a model response.

    Handles common patterns:
    - Raw code (function definition directly)
    - Code in markdown fences (```python ... ```)
    - Code with extra explanation text around it
    """
    # Try to extract from markdown code block
    if "```" in response:
        blocks = response.split("```")
        for i in range(1, len(blocks), 2):
            block = blocks[i]
            # Strip language identifier
            if block.startswith("python"):
                block = block[len("python"):]
            elif block.startswith("Python"):
                block = block[len("Python"):]
            block = block.strip()
            if block:
                return block

    # If response starts with def/class/import, assume it's raw code
    stripped = response.strip()
    for prefix in ("def ", "class ", "import ", "from "):
        if stripped.startswith(prefix):
            return stripped

    # Last resort: try to find a function definition
    lines = stripped.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith("def ") or line.strip().startswith("class "):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines)

    # Give up, return the whole thing
    return stripped


# ---------- Self-test ----------

if __name__ == "__main__":
    import json

    print("=== Code Verifier Self-Test ===\n")

    # Test 1: Correct code
    code = textwrap.dedent("""\
        def add(a, b):
            return a + b
    """)
    tests = ['assert add(1, 2) == 3', 'assert add(0, 0) == 0', 'assert add(-1, 1) == 0']
    result = verify_code(code, tests)
    print(f"Test 1 (correct code): {result}")
    assert result["passed"] is True
    assert result["reward"] == 1.0

    # Test 2: Wrong code
    code = textwrap.dedent("""\
        def add(a, b):
            return a - b
    """)
    result = verify_code(code, tests)
    print(f"Test 2 (wrong code):   {result}")
    assert result["passed"] is False
    assert result["reward"] == 0.0

    # Test 3: Syntax error
    code = "def add(a, b)\n    return a + b"
    result = verify_code(code, tests)
    print(f"Test 3 (syntax error): {result}")
    assert result["passed"] is False

    # Test 4: Infinite loop (timeout)
    code = textwrap.dedent("""\
        def add(a, b):
            while True:
                pass
            return a + b
    """)
    result = verify_code(code, tests, timeout=2)
    print(f"Test 4 (timeout):      {result}")
    assert result["passed"] is False
    assert "Timeout" in result["error"]

    # Test 5: Extract code from markdown response
    response = '''Here's the solution:

```python
def add(a, b):
    return a + b
```

This function adds two numbers.'''
    extracted = extract_code_from_response(response)
    print(f"\nTest 5 (extract from markdown):")
    print(f"  Extracted: {extracted}")
    assert "def add" in extracted

    # Test 6: Verify against actual MBPP data
    data_path = Path(__file__).parent / "data" / "code_rlvr_test.json"
    if data_path.exists():
        data = json.load(open(data_path))
        ex = data[0]
        print(f"\nTest 6 (MBPP reference solution):")
        print(f"  Task: {ex['task_id']}")
        print(f"  Tests: {ex['ground_truth']}")
        result = verify_code(
            ex["reference_code"],
            ex["ground_truth"],
            ex.get("test_imports"),
            ex.get("test_setup_code", ""),
        )
        print(f"  Result: {result}")
        assert result["passed"] is True, f"Reference solution should pass: {result}"

    print("\n=== All verifier tests passed ===")
