"""
Setup script for WaveFactor.
Uses CMakeExtension to delegate all C++ compilation, source management,
and optimization flags to CMakeLists.txt (the single source of truth).
Based on the canonical pybind11/cmake_example architecture.
"""

import os
import sys
import shlex
import shutil
import sysconfig
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


def _should_use_mingw() -> bool:
    """Detect whether MinGW Makefiles should be used on Windows."""
    if sys.platform != "win32":
        return False
    if os.environ.get("CMAKE_GENERATOR"):
        return "MinGW" in os.environ["CMAKE_GENERATOR"]
    if os.environ.get("USE_MINGW"):
        return True
    if os.environ.get("CIBUILDWHEEL") or shutil.which("cl"):
        return False

    # Prefer MSVC when Visual Studio C++ build tools are installed
    prog_files = os.environ.get("ProgramFiles(x86)") or os.environ.get("ProgramFiles", "")
    vswhere = os.path.join(prog_files, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if os.path.isfile(vswhere):
        res = subprocess.run(
            [
                vswhere,
                "-latest",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return False

    # Fall back to MinGW Makefiles on local Windows machines that only have MinGW installed
    return shutil.which("mingw32-make") is not None


def _split_cmake_args(raw_args: str):
    """Split CMAKE_ARGS safely across POSIX and Windows without stripping Windows backslashes."""
    tokens = shlex.split(raw_args, posix=(os.name != "nt"))
    if os.name == "nt":
        tokens = [
            t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in ('"', "'") else t
            for t in tokens
        ]
    return tokens


class CMakeExtension(Extension):
    """A setuptools Extension that delegates compilation to CMake."""
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Build extension that invokes CMake in an isolated temporary build directory."""
    def build_extension(self, ext):
        ext_fullpath = os.path.abspath(self.get_ext_fullpath(ext.name))
        extdir = os.path.dirname(ext_fullpath)
        os.makedirs(extdir, exist_ok=True)

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        build_temp = os.path.abspath(os.path.join(self.build_temp, ext.name))
        os.makedirs(build_temp, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={build_temp}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DBUILD_PYTHON_MODULE=ON",
            "-DBUILD_CPP_TESTS=OFF",
        ]

        build_args = ["--target", ext.name]

        use_mingw = _should_use_mingw()
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")
        if use_mingw and not cmake_generator:
            cmake_args.extend(["-G", "MinGW Makefiles"])
            is_multi_config = False
        elif sys.platform == "win32":
            is_multi_config = (
                (not cmake_generator)
                or ("Visual Studio" in cmake_generator)
                or ("Multi-Config" in cmake_generator)
            )
        else:
            is_multi_config = "Multi-Config" in cmake_generator

        if is_multi_config:
            cfg_upper = cfg.upper()
            cmake_args.extend([
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg_upper}={extdir}",
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{cfg_upper}={extdir}",
                f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{cfg_upper}={build_temp}",
            ])
            build_args.extend(["--config", cfg])
        else:
            cmake_args.append(f"-DCMAKE_BUILD_TYPE={cfg}")

        if "CMAKE_ARGS" in os.environ:
            cmake_args.extend(_split_cmake_args(os.environ["CMAKE_ARGS"]))

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args.append("--parallel")

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)
        self._copy_built_binary(build_temp, extdir, ext.name)

        if not os.path.isfile(ext_fullpath):
            raise RuntimeError(
                f"CMake build completed, but expected compiled extension binary was not found at: {ext_fullpath}"
            )

    def _copy_built_binary(self, src_dir, dest_dir, target_name):
        """Copies the compiled .pyd / .so extension matching the active Python ABI if needed."""
        src_dir = os.path.abspath(src_dir)
        dest_dir = os.path.abspath(dest_dir)
        if src_dir == dest_dir:
            return

        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
        expected_exact = f"{target_name}{ext_suffix}" if ext_suffix else None

        fallback_match = None
        for root, _, files in os.walk(src_dir):
            for f in files:
                if expected_exact and f == expected_exact:
                    self._copy_if_newer(os.path.join(root, f), os.path.join(dest_dir, f))
                    return
                if fallback_match is None and f.startswith(target_name) and f.endswith((".pyd", ".so", ".dylib")):
                    fallback_match = (os.path.join(root, f), os.path.join(dest_dir, f))

        if fallback_match is not None:
            self._copy_if_newer(*fallback_match)

    @staticmethod
    def _copy_if_newer(src_file, dest_file):
        if os.path.abspath(src_file) == os.path.abspath(dest_file):
            return
        if not os.path.exists(dest_file) or os.path.getmtime(src_file) > os.path.getmtime(dest_file):
            shutil.copy2(src_file, dest_file)


setup(
    ext_modules=[CMakeExtension("WaveFactor")],
    cmdclass={"build_ext": CMakeBuild},
)
