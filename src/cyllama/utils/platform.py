"""Platform-specific runtime setup for native extension loading."""

import sys

_initialized = False


def ensure_native_deps() -> None:
    """Ensure platform-specific shared libraries are discoverable.

    Idempotent. On Windows, **always** registers NVIDIA CUDA toolkit ``bin`` /
    ``bin\\x64`` directories via :func:`os.add_dll_directory` so extensions
    linked with CUDA (e.g. ``llama_cpp``) can resolve ``cudart``, ``cublas``,
    etc.  We do not gate on ``build_config.json``: that file is often missing
    from wheels (gitignored; uv/scikit-build exclude ignored files), and
    isolated builds may record ``cuda: false`` even when CMake built with CUDA.

    No-op on other platforms. Safe for CPU-only wheels: only search paths are
    added; CUDA DLLs are not loaded until the extension needs them.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    if sys.platform != "win32":
        return

    _setup_cuda_dll_paths()


def _setup_cuda_dll_paths() -> None:
    """Register CUDA toolkit DLL directories on Windows."""
    import glob
    import os
    import re
    import shutil

    if not hasattr(os, "add_dll_directory"):
        return

    seen: set[str] = set()
    ordered_dirs: list[str] = []

    def add_bin(path: str) -> None:
        if path in seen or not os.path.isdir(path):
            return
        seen.add(path)
        ordered_dirs.append(path)
        try:
            os.add_dll_directory(path)  # type: ignore[attr-defined]
        except OSError:
            pass

    # 1. Explicit env vars (highest priority)
    for key in ("CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(key)
        if root:
            add_bin(os.path.join(root, "bin"))
            # CUDA 12+ Windows toolkits often ship cuBLAS in bin\x64 only
            add_bin(os.path.join(root, "bin", "x64"))

    # 2. nvcc on PATH
    nvcc = shutil.which("nvcc")
    if nvcc:
        add_bin(os.path.dirname(os.path.abspath(nvcc)))

    # 2b. Conda / mamba env (zlib, cudnn, etc. often only here)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        add_bin(os.path.join(conda_prefix, "Library", "bin"))

    # 3. Standard install location (newest version first)
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    cuda_root = os.path.join(pf, "NVIDIA GPU Computing Toolkit", "CUDA")
    if os.path.isdir(cuda_root):

        def ver_key(d: str) -> tuple[int, ...]:
            m = re.search(r"v(\d+)\.(\d+)", d)
            return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

        for vdir in sorted(
            glob.glob(os.path.join(cuda_root, "v*")),
            key=ver_key,
            reverse=True,
        ):
            add_bin(os.path.join(vdir, "bin"))
            add_bin(os.path.join(vdir, "bin", "x64"))

    # ``add_dll_directory`` is not always enough for every transitive DLL;
    # prepending the same dirs to PATH matches what many GPU stacks expect.
    if ordered_dirs:
        os.environ["PATH"] = os.pathsep.join(ordered_dirs) + os.pathsep + os.environ.get(
            "PATH", ""
        )
