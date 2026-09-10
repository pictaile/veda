"""The veda golden-file format, in pure Python.

One fp32 tensor per file, little-endian throughout:

    offset  size        contents
    0       8 bytes     magic b"VEDATNSR"
    8       4 bytes     uint32   dtype code (0 = F32)
    12      4 bytes     uint32   rank
    16      8*rank      uint64   dimensions, outermost first
    ...     4*numel     float32  data, contiguous row-major

Deliberately dependency-free: no numpy, no torch. This module exists so the C++ reader can be
verified against an independent implementation of the same description — if the two disagree, the
bug is in the description, and finding that out now is much cheaper than after 600 MB of dumps.

Development tooling. Nothing in src/ imports it, and Veda neither builds nor runs with Python.
"""

import struct

MAGIC = b"VEDATNSR"
DTYPE_F32 = 0


def write_tensor(path, shape, values):
    """Writes a flat sequence of floats under the given shape.

    `values` is iterated in row-major index order and must hold exactly prod(shape) elements.
    """
    values = list(values)
    expected = 1
    for dimension in shape:
        expected *= dimension
    if len(values) != expected:
        raise ValueError(f"shape {tuple(shape)} needs {expected} values, got {len(values)}")

    with open(path, "wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<II", DTYPE_F32, len(shape)))
        for dimension in shape:
            out.write(struct.pack("<Q", dimension))
        out.write(struct.pack(f"<{len(values)}f", *values))


def read_tensor(path):
    """Returns (shape, values). Raises ValueError with the path on any corruption."""
    with open(path, "rb") as source:
        blob = source.read()

    if len(blob) < 16 or blob[:8] != MAGIC:
        raise ValueError(f'tensor file "{path}": is not a veda tensor file (bad magic)')

    dtype, rank = struct.unpack_from("<II", blob, 8)
    if dtype != DTYPE_F32:
        raise ValueError(f'tensor file "{path}": unsupported dtype code {dtype}')

    header_end = 16 + 8 * rank
    if len(blob) < header_end:
        raise ValueError(f'tensor file "{path}": truncated header')

    shape = struct.unpack_from(f"<{rank}Q", blob, 16) if rank else ()
    count = 1
    for dimension in shape:
        count *= dimension

    body = len(blob) - header_end
    if body != count * 4:
        raise ValueError(
            f'tensor file "{path}": body is {body} bytes, header implies {count * 4} '
            f"for shape {tuple(shape)}"
        )

    values = list(struct.unpack_from(f"<{count}f", blob, header_end)) if count else []
    return tuple(shape), values


def write_torch(path, tensor):
    """Writes a torch tensor. Kept here so the dumper stays about hooks, not about bytes."""
    detached = tensor.detach().to("cpu").float().contiguous()
    write_tensor(path, tuple(detached.shape), detached.reshape(-1).tolist())
