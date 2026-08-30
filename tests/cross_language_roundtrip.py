#!/usr/bin/env python3
"""E3.S2.T4 — the golden-file format, verified across the language boundary.

    python3 tests/cross_language_roundtrip.py <path-to-tensor_file_tool>

Both directions matter. C++ writing and C++ reading proves only that it is self-consistent; a
misread of the format description would round-trip perfectly and still produce files no reference
dumper could write.
"""

import os
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from veda_tensor import read_tensor, write_tensor  # noqa: E402

# The same fixture the C++ tool holds.
FIXTURE_SHAPE = (2, 3)
FIXTURE_VALUES = [1.5, -2.25, 0.0, 3.125, -0.1, 100000.0]

# Python floats are doubles; the file holds float32. -0.1 as a double is not the number a float32
# -0.1 widens back to, so the fixture is rounded through fp32 before anything is compared. This is
# not a tolerance — it is the same value, spelled in the precision the file actually stores.
FIXTURE_F32 = list(struct.unpack(f"<{len(FIXTURE_VALUES)}f",
                                 struct.pack(f"<{len(FIXTURE_VALUES)}f", *FIXTURE_VALUES)))

checks_run = 0
failures = []


def check(condition, description):
    global checks_run
    checks_run += 1
    if condition:
        print(f"  ok   {description}")
    else:
        print(f"  FAIL {description}")
        failures.append(description)


def main():
    if len(sys.argv) != 2:
        print("usage: cross_language_roundtrip.py <path-to-tensor_file_tool>")
        return 2
    tool = sys.argv[1]

    with tempfile.TemporaryDirectory() as directory:
        from_cpp = os.path.join(directory, "from_cpp.bin")
        from_python = os.path.join(directory, "from_python.bin")

        # C++ writes, Python reads
        subprocess.run([tool, "write", from_cpp], check=True)
        shape, values = read_tensor(from_cpp)
        check(shape == FIXTURE_SHAPE, f"python reads the shape written by c++: {shape}")
        check(values == FIXTURE_F32, "python reads every value written by c++, exactly")
        check(values != FIXTURE_VALUES, "and -0.1 as float32 is not -0.1 as a double")

        # the byte length the format description predicts
        expected_size = 8 + 4 + 4 + 8 * len(FIXTURE_SHAPE) + 4 * len(FIXTURE_VALUES)
        check(os.path.getsize(from_cpp) == expected_size,
              f"the file is {expected_size} bytes, as the format says")

        # Python writes, C++ reads
        write_tensor(from_python, FIXTURE_SHAPE, FIXTURE_VALUES)
        check(os.path.getsize(from_python) == expected_size, "python writes the same byte length")
        with open(from_cpp, "rb") as a, open(from_python, "rb") as b:
            check(a.read() == b.read(), "the two files are byte-identical")
        result = subprocess.run([tool, "verify", from_python])
        check(result.returncode == 0, "c++ reads every value written by python, exactly")

        # a rank-0 tensor crosses too
        scalar = os.path.join(directory, "scalar.bin")
        write_tensor(scalar, (), [7.5])
        check(read_tensor(scalar) == ((), [7.5]), "a rank-0 tensor round-trips in python")
        check(os.path.getsize(scalar) == 20, "a rank-0 file is 20 bytes")

        # corruption is rejected on this side too
        broken = os.path.join(directory, "broken.bin")
        with open(from_cpp, "rb") as source:
            blob = source.read()
        with open(broken, "wb") as out:
            out.write(blob[:-4])
        try:
            read_tensor(broken)
            check(False, "python rejects a truncated file")
        except ValueError as error:
            check("body is" in str(error), f"python rejects a truncated file: {error}")

    # the dumper must stay out of the build (an epic acceptance criterion)
    root = os.path.join(os.path.dirname(__file__), "..")
    with open(os.path.join(root, "CMakeLists.txt"), encoding="utf-8") as cmake:
        contents = cmake.read()
    check("dump_reference.py" not in contents, "no cmake target references dump_reference.py")
    check("veda_tensor.py" not in contents, "no cmake target references veda_tensor.py")

    print(f"\nCrossLanguageRoundTrip: {checks_run - len(failures)}/{checks_run} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
