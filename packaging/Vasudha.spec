# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Vasudha desktop app.

One-folder, not one-file. A one-file build unpacks the whole bundle to a temp
directory on every launch, and llama-cpp-python's native libraries are large
enough that this adds seconds to every start — on top of the model load. The
folder is what gets zipped or handed to an installer.

Deliberately NOT bundled: torch, transformers, datasets, unsloth and friends.
They are what the training pipeline needs, not the app: this process only talks
HTTP to ollama or runs llama-cpp in-process. Leaving them in adds gigabytes for
code that never executes here.
"""
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

ROOT = Path(SPECPATH).parent

block_cipher = None

# --- data ------------------------------------------------------------------
# The UI goes at the BUNDLE ROOT as "ui", not "app/ui".
#
# pywebview's get_app_root() returns sys._MEIPASS in a frozen build and serves
# its local HTTP root from there, building the page URL relative to it. With
# the files at _MEIPASS/app/ui it requested http://127.0.0.1:PORT/ui/index.html
# and 404'd — one directory level too deep. Putting them at _MEIPASS/ui makes
# the path pywebview asks for the path that exists.
datas = [
    (str(ROOT / "app" / "ui"), "ui"),
    # Persona presets ship as data and are copied to %LOCALAPPDATA%\Vasudha\
    # personas on first run, so voice can be tuned by editing a JSON file
    # instead of rebuilding a 212 MB bundle.
    (str(ROOT / "personas"), "personas"),
    # Download link + checksum, resolvable at runtime so a moved URL can be
    # fixed by editing a file next to the exe instead of rebuilding 212 MB.
    (str(ROOT / "model_source.json"), "."),
]

# pywebview ships the WebView2 interop assemblies as package data; without them
# the window silently fails to create a browser control.
datas += collect_data_files("webview")

# llama-cpp-python carries its own compiled backend. With the Vulkan wheel that
# is several shared libraries, not one: llama.cpp splits its GPU backends into
# separate ggml-vulkan / ggml-cpu modules and loads them by name at runtime, so
# collect_dynamic_libs (which walks the package directory) is what keeps them
# together. Missing one does not fail the build — it fails at first launch, as a
# silent fallback to CPU.
datas += collect_data_files("llama_cpp")
binaries = collect_dynamic_libs("llama_cpp")

# pip, copied in as ORDINARY FILES rather than frozen into the archive.
#
# Freezing it does not work. pip vendors distlib, and distlib resolves its own
# resources through a finder registry that understands real directories and
# zipimports and not PyInstaller's loader, so `pip install` dies with
#
#     DistlibException: Unable to locate finder for 'pip._vendor.distlib'
#
# Verified by building it that way first. Shipped as a plain directory tree
# that app/runtime.py puts on sys.path, pip sees the layout it expects and
# works — including its vendored CA bundle, which it needs to verify PyPI.
import pip as _pip_pkg
datas += [(str(Path(_pip_pkg.__file__).parent), "pip_runtime/pip")]

# --- imports ---------------------------------------------------------------
hiddenimports = []
# The platform backend is chosen at runtime by string, so PyInstaller's static
# analysis never sees these imports.
hiddenimports += collect_submodules("webview.platforms")
hiddenimports += [
    "clr",                    # pythonnet, required by the WinForms/EdgeChromium backend
    "llama_cpp",
    "bs4",
    "ddgs",
    "requests",
    # jinja2 renders the GGUF's own chat template, which is what lets this app
    # load a model other than the one it ships with. Imported by name inside
    # app/backends.py, so PyInstaller does not see it statically.
    "jinja2",
    # Physical-core detection for n_threads. Optional at runtime (there is a
    # fallback), bundled because the fallback is measurably worse.
    "psutil",
    # Available to sandboxed python_tool runs, since a frozen build only has
    # what is bundled. Stdlib maths is the common case for this product.
    "decimal",
    "fractions",
    "statistics",
]

# Standard-library modules pip imports that nothing else in this app does, so
# PyInstaller's analysis never sees them. Missing one is not a build error — it
# is a ModuleNotFoundError partway through an install, which reads like a
# broken package rather than a packaging gap. Found by building and running the
# real thing: logging.config was the first, and there is no reason to discover
# the rest one rebuild at a time.
hiddenimports += [
    "logging.config", "logging.handlers",
    "configparser", "sysconfig", "platform", "netrc", "getpass",
    "http.cookiejar", "http.client", "email", "email.parser",
    "xml.etree.ElementTree", "unicodedata", "csv", "base64", "binascii",
    "zipfile", "tarfile", "gzip", "bz2", "lzma", "shutil", "tempfile",
    "hashlib", "ssl", "socket", "select", "queue",
    "importlib.metadata", "importlib.resources", "pkgutil", "sqlite3",
    "compileall", "py_compile", "filecmp", "difflib", "pprint",
    "textwrap", "argparse", "optparse", "webbrowser", "ctypes.util",
]

# Present for the model backend; also makes numeric work in python_tool usable.
try:
    import numpy  # noqa: F401
    hiddenimports.append("numpy")
except ImportError:
    pass

excludes = [
    "torch", "torchvision", "torchaudio",
    "transformers", "datasets", "tokenizers", "safetensors",
    "peft", "trl", "unsloth", "accelerate", "bitsandbytes",
    "triton", "tensorboard", "wandb",
    "matplotlib", "IPython", "notebook", "jupyter",
    "scipy", "pandas", "sklearn",
    "flask",          # the desktop app uses the js_api bridge, not HTTP
    "PyQt5", "PySide2", "PySide6", "PyQt6",   # other pywebview backends
    "tkinter",
]

a = Analysis(
    [str(ROOT / "app" / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Windows wants .ico, macOS wants .icns, and handing either the wrong format
# is a hard error rather than a fallback.
if sys.platform == "darwin":
    _icon = ROOT / "packaging" / "vasudha.icns"
elif sys.platform == "win32":
    _icon = ROOT / "packaging" / "vasudha.ico"
else:
    _icon = ROOT / "packaging" / "nonexistent"

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Vasudha",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # No console window. Safe only because app/runtime.py rebinds sys.stdout and
    # sys.stderr to fds 1/2 before running a sandboxed script — a windowed build
    # otherwise leaves them unwired, capture_output returns empty, and every
    # python_tool call silently produces no result.
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(_icon) if _icon.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Vasudha",
)

# PyInstaller leaves an intermediate Vasudha.exe in build/Vasudha/. It is not
# runnable — it has no bundled runtime beside it — and double-clicking it fails
# with "Failed to load Python DLL ... python314.dll". It looks exactly like the
# real thing in Explorer, so remove it and leave only dist/Vasudha/Vasudha.exe.
_stray = (Path(DISTPATH).parent / "build" / "Vasudha" /
          ("Vasudha.exe" if sys.platform == "win32" else "Vasudha"))
try:
    if _stray.exists():
        _stray.unlink()
except OSError:
    pass
