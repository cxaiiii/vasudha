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

# llama-cpp-python carries its own compiled backend.
datas += collect_data_files("llama_cpp")
binaries = collect_dynamic_libs("llama_cpp")

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
    # Available to sandboxed python_tool runs, since a frozen build only has
    # what is bundled. Stdlib maths is the common case for this product.
    "decimal",
    "fractions",
    "statistics",
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

_icon = ROOT / "packaging" / "vasudha.ico"

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
_stray = Path(DISTPATH).parent / "build" / "Vasudha" / "Vasudha.exe"
try:
    if _stray.exists():
        _stray.unlink()
except OSError:
    pass
