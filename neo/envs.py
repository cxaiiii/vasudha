"""Tool environments for RL, and the matching tool schemas for evaluation.

The model should work with whatever harness it is dropped into, so the one
executable tool — a Python interpreter — is offered under several names and
descriptions that real harnesses use (`python`, `code_interpreter`,
`run_python`, `python_tool`), always with a single `code` argument. Nothing
here imitates a particular application's strings.

`TextEnv` has no tools. It carries the tasks whose tools are declared in the
prompt itself (function-calling tasks: the calls are graded, not executed) and
every task meant to be solved with no tools at all.

TRL's GRPOTrainer instantiates these through `environment_factory`: one
instance per rollout, `reset(**row)` before generation, public methods exposed
as tools (their docstrings become the schemas).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from neo.sandbox import default_pool

PYTHON_DOCS = {
    "python": (
        "Execute Python code in a fresh sandbox and return what it prints (stdout, then any error). "
        "Use print() to show results. numpy, scipy and sympy are available.",
        "The Python code to run.",
    ),
    "code_interpreter": (
        "Run Python 3 code and return its output. Use it for calculations, unit conversions, data "
        "processing and checking your work; print every value you need.",
        "Python source code.",
    ),
    "run_python": (
        "Run a Python script and get its standard output and errors. Each call starts fresh, so "
        "define everything the script needs.",
        "A complete Python script.",
    ),
    "python_tool": (
        "Compute a numeric or algorithmic result in a Python sandbox and return its stdout.",
        "Python source to execute. Print the final values.",
    ),
}


def _doc(name: str) -> str:
    desc, arg = PYTHON_DOCS[name]
    return f"""
{desc}

Args:
    code: {arg}
"""


@dataclass
class CallRecord:
    name: str
    code: str
    result: str
    ok: bool


def _failed(result: str) -> bool:
    return ("Traceback (most recent call last)" in result or result.startswith("TimeoutError")
            or "TimeoutError: execution exceeded" in result or result.startswith("SystemError"))


class _Base:
    tool_name: Optional[str] = None

    def __init__(self) -> None:
        self.calls: list[CallRecord] = []
        self.task: dict[str, Any] = {}

    def reset(self, **kwargs) -> None:
        self.calls = []
        self.task = kwargs
        return None

    def _run(self, name: str, code: str) -> str:
        result = default_pool().run(code or "")
        self.calls.append(CallRecord(name, code or "", result, not _failed(result)))
        return result


class TextEnv(_Base):
    """No tools."""


class PythonEnv(_Base):
    tool_name = "python"

    def python(self, code: str) -> str:
        return self._run("python", code)


class CodeInterpreterEnv(_Base):
    tool_name = "code_interpreter"

    def code_interpreter(self, code: str) -> str:
        return self._run("code_interpreter", code)


class RunPythonEnv(_Base):
    tool_name = "run_python"

    def run_python(self, code: str) -> str:
        return self._run("run_python", code)


class PythonToolEnv(_Base):
    tool_name = "python_tool"

    def python_tool(self, code: str) -> str:
        return self._run("python_tool", code)


PythonEnv.python.__doc__ = _doc("python")
CodeInterpreterEnv.code_interpreter.__doc__ = _doc("code_interpreter")
RunPythonEnv.run_python.__doc__ = _doc("run_python")
PythonToolEnv.python_tool.__doc__ = _doc("python_tool")

ENVIRONMENTS = {
    "text": TextEnv,
    "python": PythonEnv,
    "code_interpreter": CodeInterpreterEnv,
    "run_python": RunPythonEnv,
    "python_tool": PythonToolEnv,
}
PYTHON_ENVS = ("python", "code_interpreter", "run_python", "python_tool")


def python_tool_schema(name: str) -> dict:
    """JSON schema of a python-tool variant, exactly as TRL renders it."""
    desc, arg = PYTHON_DOCS[name]
    return {"type": "function", "function": {
        "name": name,
        "description": desc,
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": arg}},
                       "required": ["code"]},
        "return": {"type": "string"},
    }}
