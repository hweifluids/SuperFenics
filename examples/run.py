"""Run a SuperFenics example on Windows and keep artifacts under .tests."""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
from ctypes import wintypes
from pathlib import Path


def bind_cpu(cpu):
    """Keep this launcher and its children on one logical CPU in Windows group 0."""

    class GroupAffinity(ctypes.Structure):
        _fields_ = [
            ("mask", ctypes.c_size_t),
            ("group", wintypes.WORD),
            ("reserved", wintypes.WORD * 3),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD
    ]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.GetProcessGroupAffinity.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(wintypes.WORD), ctypes.POINTER(wintypes.WORD)
    ]
    kernel.GetProcessAffinityMask.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)
    ]

    process = kernel.GetCurrentProcess()
    affinity = GroupAffinity(mask=1 << cpu, group=0)
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        # JobObjectGroupInformationEx takes an array of GROUP_AFFINITY entries.
        if not kernel.SetInformationJobObject(
            job, 14, ctypes.byref(affinity), ctypes.sizeof(affinity)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.AssignProcessToJobObject(job, process):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(job)

    groups = (wintypes.WORD * 64)()
    count = wintypes.WORD(len(groups))
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessGroupAffinity(process, ctypes.byref(count), groups):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.GetProcessAffinityMask(
        process, ctypes.byref(process_mask), ctypes.byref(system_mask)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if list(groups[:count.value]) != [0] or process_mask.value != affinity.mask:
        raise RuntimeError(
            f"Unexpected CPU affinity: groups {list(groups[:count.value])}, "
            f"mask {process_mask.value:#x}"
        )
    return f"Windows processor group 0, logical CPU {cpu}, mask {process_mask.value:#x}"


def run(command, out, env, log):
    print(subprocess.list2cmdline([str(arg) for arg in command]), file=log, flush=True)
    output = []
    with subprocess.Popen(
        command, cwd=out, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, errors="replace",
    ) as child:
        for line in child.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            output.append(line)
        if child.wait():
            raise subprocess.CalledProcessError(child.returncode, command)
    return "".join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=("cpu-assembly", "gpu-cell-kernel"))
    parser.add_argument("--cpu", type=int, default=0, help="Logical CPU in group 0 (0–63)")
    parser.add_argument("--gpu", type=int, help="CUDA device index for gpu-cell-kernel")
    args = parser.parse_args()
    if sys.platform != "win32" or not 0 <= args.cpu <= 63:
        parser.error("Choose a logical CPU between 0 and 63 on Windows.")
    if args.example == "gpu-cell-kernel" and (args.gpu is None or args.gpu < 0):
        parser.error("Provide the CUDA device index using --gpu.")

    affinity = bind_cpu(args.cpu)
    examples = Path(__file__).resolve().parent
    source = examples / args.example
    runs = examples.parent / ".tests" / "examples" / args.example
    runs.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="run-", dir=runs))
    print(f"{affinity}\nOutput: {out}", flush=True)

    env = os.environ.copy()
    library_bin = Path(sys.prefix) / "Library" / "bin"
    if library_bin.is_dir():
        env["PATH"] = f"{library_bin}{os.pathsep}{env.get('PATH', '')}"
    env.update(
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1", TEMP=str(out), TMP=str(out),
    )
    python = [sys.executable, "-B"]
    with (out / "console.log").open("w", encoding="utf-8") as log:
        print(affinity, file=log, flush=True)
        if args.example == "cpu-assembly":
            run([*python, source / "benchmark_cpu.py", "--out", out], out, env, log)
        else:
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            run([*python, source / "generate_gpu_probe.py", "--out", out], out, env, log)
            exe = out / "probe.exe"
            run(
                ["nvcc", "-std=c++20", "-arch=sm_75", "-O3", "--fmad=false", "-lineinfo",
                 "-Xcompiler", "/O2,/fp:precise", "--ptxas-options=-v",
                 "-I", out, "-o", exe, source / "probe.cu"],
                out, env, log,
            )
            result = run([exe, "65536", str(args.gpu), "7"], out, env, log)
            (out / "gpu_probe_result.json").write_text(
                json.dumps(json.loads(result), indent=2) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
