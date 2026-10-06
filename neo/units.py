"""Units: parse "16 mm", "1.2e8 Pa", "4.7 × 10⁻³ F", "₹77,165" out of free text.

This is the part of the reward that decides whether an answer is right, so it
has to be stricter than a regex over the last number. The failures it exists
to catch are the ones this project has already seen in production:

  * the tool prints 0.016 (metres) and the reply says "0.016 mm";
  * 1.2e8 Pa computed correctly, then reported as "120,000 MPa";
  * the right magnitude with no unit at all.

So every quantity is parsed together with its unit, converted to SI, and only
then compared. A value labelled with the wrong unit is a wrong value.

Case matters for units and is kept: MPa vs mPa and MW vs mW differ by 10^9.
The only case-insensitive fallbacks are spellings that are unambiguous in an
engineering answer ("mpa" can only sensibly mean MPa there).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

# --------------------------------------------------------------------------
# Unit table: canonical name -> (dimension, factor to SI base)
# Temperature is affine and handled separately (dimension "temperature").
# --------------------------------------------------------------------------

_UNITS: dict[str, tuple[str, float]] = {}
_ALIASES: dict[str, str] = {}


def _add(canonical: str, dim: str, factor: float, *aliases: str) -> None:
    _UNITS[canonical] = (dim, factor)
    _ALIASES[_squash(canonical)] = canonical
    for alias in aliases:
        _ALIASES[_squash(alias)] = canonical


def _squash(unit: str) -> str:
    """Canonical lookup key: unicode folded, separators unified, no spaces."""
    u = unit.strip()
    u = u.replace("µ", "u").replace("μ", "u")
    u = u.replace("Ω", "ohm").replace("Ω", "ohm")
    for sup, plain in (("²", "^2"), ("³", "^3"), ("⁴", "^4"), ("⁻¹", "^-1"), ("⁻²", "^-2")):
        u = u.replace(sup, plain)
    u = re.sub(r"[·⋅×∙•]", "*", u)
    u = u.replace("(", "").replace(")", "").replace(" ", "")
    # mm2 -> mm^2, m3 -> m^3 (a letter immediately followed by 2/3/4)
    u = re.sub(r"(?<=[A-Za-z])([234])(?![0-9])", r"^\1", u)
    u = u.replace("**", "^")
    return u


# length
_add("m", "length", 1.0, "metre", "metres", "meter", "meters")
_add("mm", "length", 1e-3, "millimetre", "millimetres", "millimeter", "millimeters")
_add("cm", "length", 1e-2, "centimetre", "centimetres", "centimeter", "centimeters")
_add("km", "length", 1e3, "kilometre", "kilometres", "kilometer", "kilometers")
_add("um", "length", 1e-6, "micron", "microns", "micrometre", "micrometres", "micrometer", "micrometers")
_add("nm", "length", 1e-9, "nanometre", "nanometres", "nanometer", "nanometers")
_add("inch", "length", 0.0254, "inches", "\"")
_add("ft", "length", 0.3048, "foot", "feet")
# area
_add("m^2", "area", 1.0, "sq m", "square metres", "square meters")
_add("mm^2", "area", 1e-6, "sq mm", "square millimetres", "square millimeters")
_add("cm^2", "area", 1e-4, "sq cm", "square centimetres", "square centimeters")
_add("in^2", "area", 0.0254 ** 2, "sq in", "square inches")
_add("ft^2", "area", 0.3048 ** 2, "sq ft", "square feet")
# volume
_add("m^3", "volume", 1.0, "cubic metres", "cubic meters")
_add("L", "volume", 1e-3, "l", "litre", "litres", "liter", "liters")
_add("mL", "volume", 1e-6, "ml", "millilitre", "millilitres", "milliliter", "milliliters")
_add("cm^3", "volume", 1e-6, "cc", "cubic centimetres", "cubic centimeters")
_add("mm^3", "volume", 1e-9)
_add("gal", "volume", 3.785411784e-3, "gallon", "gallons")
# second moment of area
_add("m^4", "length4", 1.0)
_add("mm^4", "length4", 1e-12)
_add("cm^4", "length4", 1e-8)
_add("in^4", "length4", 0.0254 ** 4)
# mass
_add("kg", "mass", 1.0, "kilogram", "kilograms", "kilogramme", "kilogrammes")
_add("g", "mass", 1e-3, "gram", "grams", "gramme", "grammes")
_add("mg", "mass", 1e-6, "milligram", "milligrams")
_add("t", "mass", 1e3, "tonne", "tonnes", "metric ton", "metric tons")
_add("lb", "mass", 0.45359237, "lbs", "pound", "pounds")
# time
_add("s", "time", 1.0, "sec", "secs", "second", "seconds")
_add("ms", "time", 1e-3, "millisecond", "milliseconds")
_add("us", "time", 1e-6, "microsecond", "microseconds")
_add("min", "time", 60.0, "mins", "minute", "minutes")
_add("h", "time", 3600.0, "hr", "hrs", "hour", "hours")
_add("day", "time", 86400.0, "days")
_add("year", "time", 365.0 * 86400.0, "years", "yr", "yrs")
# velocity / acceleration
_add("m/s", "velocity", 1.0, "m*s^-1", "metres per second", "meters per second")
_add("km/h", "velocity", 1 / 3.6, "kmh", "kph", "km/hr", "kilometres per hour", "kilometers per hour")
_add("mm/s", "velocity", 1e-3)
_add("cm/s", "velocity", 1e-2)
_add("ft/s", "velocity", 0.3048)
_add("mph", "velocity", 0.44704, "mi/h", "mi/hr", "miles per hour")
_add("mi", "length", 1609.344, "mile", "miles")
_add("T", "magnetic_field", 1.0, "tesla", "teslas")
_add("mT", "magnetic_field", 1e-3)
_add("uT", "magnetic_field", 1e-6)
_add("N/C", "electric_field", 1.0, "V/m")
_add("kN/C", "electric_field", 1e3, "kV/m")
_add("J/K", "entropy", 1.0)
_add("kg*m^2", "mass_moment", 1.0, "kg m^2")
_add("W/m^2", "flux", 1.0)
_add("m/s^2", "acceleration", 1.0, "m*s^-2")
# force
_add("N", "force", 1.0, "newton", "newtons")
_add("kN", "force", 1e3, "kilonewton", "kilonewtons")
_add("MN", "force", 1e6, "meganewton", "meganewtons")
_add("mN", "force", 1e-3)
_add("lbf", "force", 4.4482216152605)
_add("kgf", "force", 9.80665)
# pressure / stress
_add("Pa", "pressure", 1.0, "pascal", "pascals", "N/m^2")
_add("kPa", "pressure", 1e3, "kilopascal", "kilopascals", "kN/m^2")
_add("MPa", "pressure", 1e6, "megapascal", "megapascals", "N/mm^2", "MN/m^2")
_add("GPa", "pressure", 1e9, "gigapascal", "gigapascals", "kN/mm^2")
_add("bar", "pressure", 1e5, "bars")
_add("mbar", "pressure", 1e2)
_add("atm", "pressure", 101325.0, "atmosphere", "atmospheres")
_add("psi", "pressure", 6894.757293168)
_add("ksi", "pressure", 6894757.293168)
_add("mmHg", "pressure", 133.322387415)
# energy
_add("J", "energy", 1.0, "joule", "joules", "N*m_energy")
_add("kJ", "energy", 1e3, "kilojoule", "kilojoules")
_add("MJ", "energy", 1e6, "megajoule", "megajoules")
_add("GJ", "energy", 1e9)
_add("mJ", "energy", 1e-3, "millijoule", "millijoules")
_add("uJ", "energy", 1e-6)
_add("Wh", "energy", 3600.0, "watt-hour", "watt-hours")
_add("kWh", "energy", 3.6e6, "kW*h", "kilowatt-hour", "kilowatt-hours")
_add("cal", "energy", 4.184, "calorie", "calories")
_add("kcal", "energy", 4184.0, "kilocalorie", "kilocalories")
_add("BTU", "energy", 1055.05585262, "btu", "Btu")
_add("eV", "energy", 1.602176634e-19)
# power
_add("W", "power", 1.0, "watt", "watts")
_add("kW", "power", 1e3, "kilowatt", "kilowatts")
_add("MW", "power", 1e6, "megawatt", "megawatts")
_add("mW", "power", 1e-3, "milliwatt", "milliwatts")
_add("uW", "power", 1e-6)
_add("hp", "power", 745.6998715822702, "horsepower")
# frequency / angular
_add("Hz", "frequency", 1.0, "hertz")
_add("kHz", "frequency", 1e3, "kilohertz")
_add("MHz", "frequency", 1e6, "megahertz")
_add("GHz", "frequency", 1e9, "gigahertz")
_add("rad/s", "angular_velocity", 1.0, "rad*s^-1", "radians per second")
_add("rpm", "angular_velocity", 2 * math.pi / 60, "rev/min", "RPM", "r/min")
_add("rad", "angle", 1.0, "radian", "radians")
_add("deg", "angle", math.pi / 180, "°", "degree", "degrees")
# torque / moment — dimensionally identical to energy (N·m = J), and scraped
# answers use N·m for work, so the two share a dimension here.
_add("N*m", "energy", 1.0, "Nm", "N-m", "newton-metre", "newton-metres", "newton metres", "newton-meters")
_add("kN*m", "energy", 1e3, "kNm", "kN-m")
_add("N*mm", "energy", 1e-3, "Nmm", "N-mm")
_add("lbf*ft", "energy", 1.3558179483314004, "lb-ft", "ft-lb", "ft*lbf", "lbf-ft")
# electrical
_add("V", "voltage", 1.0, "volt", "volts")
_add("mV", "voltage", 1e-3, "millivolt", "millivolts")
_add("kV", "voltage", 1e3, "kilovolt", "kilovolts")
_add("uV", "voltage", 1e-6)
_add("A", "current", 1.0, "amp", "amps", "ampere", "amperes")
_add("mA", "current", 1e-3, "milliamp", "milliamps", "milliampere", "milliamperes")
_add("uA", "current", 1e-6)
_add("kA", "current", 1e3)
_add("ohm", "resistance", 1.0, "ohms", "Ohm", "Ohms", "Ω")
_add("kohm", "resistance", 1e3, "kOhm", "kOhms", "kohms", "kΩ", "kiloohm", "kiloohms")
_add("Mohm", "resistance", 1e6, "MOhm", "MOhms", "MΩ", "megaohm", "megaohms")
_add("mohm", "resistance", 1e-3, "mOhm", "mΩ", "milliohm", "milliohms")
_add("F", "capacitance", 1.0, "farad", "farads")
_add("mF", "capacitance", 1e-3)
_add("uF", "capacitance", 1e-6, "microfarad", "microfarads")
_add("nF", "capacitance", 1e-9, "nanofarad", "nanofarads")
_add("pF", "capacitance", 1e-12, "picofarad", "picofarads")
_add("H", "inductance", 1.0, "henry", "henries", "henrys")
_add("mH", "inductance", 1e-3, "millihenry", "millihenries")
_add("uH", "inductance", 1e-6, "microhenry", "microhenries")
_add("nH", "inductance", 1e-9)
_add("Ah", "charge_ah", 1.0, "A*h", "amp-hour", "amp-hours")
_add("mAh", "charge_ah", 1e-3, "mA*h")
# density / flow / chemistry
_add("kg/m^3", "density", 1.0, "kg*m^-3")
_add("g/cm^3", "density", 1e3, "g/cc", "g/mL", "g/ml", "kg/L", "kg/l", "g*cm^-3")
_add("g/L", "density", 1.0, "g/l")
_add("m^3/s", "flow", 1.0, "m^3*s^-1", "cumecs")
_add("L/s", "flow", 1e-3, "l/s", "litres per second", "liters per second")
_add("L/min", "flow", 1e-3 / 60, "l/min", "lpm", "LPM")
_add("m^3/h", "flow", 1 / 3600, "m^3/hr", "m^3*h^-1")
_add("mL/s", "flow", 1e-6, "ml/s")
_add("mol", "amount", 1.0, "mole", "moles", "mols")
_add("mmol", "amount", 1e-3, "millimole", "millimoles")
_add("kmol", "amount", 1e3)
_add("mol/L", "concentration", 1.0, "M", "mol/l", "mol*L^-1", "molar")
_add("mM", "concentration", 1e-3, "mmol/L", "mmol/l", "millimolar")
_add("g/mol", "molar_mass", 1.0, "g*mol^-1")
_add("Pa*s", "viscosity", 1.0, "Pa-s", "N*s/m^2", "kg/m*s")
_add("mPa*s", "viscosity", 1e-3, "cP", "centipoise", "mPa-s")
_add("m^2/s", "kinematic_viscosity", 1.0)
_add("cSt", "kinematic_viscosity", 1e-6, "mm^2/s")
_add("W/m*K", "conductivity", 1.0, "W/mK", "W*m^-1*K^-1", "W/m/K")
_add("W/m^2*K", "htc", 1.0, "W/m^2K", "W*m^-2*K^-1", "W/m^2/K")
_add("W/m^2", "flux", 1.0, "W*m^-2")
_add("kW/m^2", "flux", 1e3)
_add("N/m", "stiffness", 1.0, "N*m^-1")
_add("kN/m", "stiffness", 1e3)
_add("N/mm", "stiffness", 1e3)
_add("J/kg*K", "specific_heat", 1.0, "J/kgK", "J*kg^-1*K^-1", "J/kg/K", "J/kg*C", "J/kg°C")
_add("kJ/kg*K", "specific_heat", 1e3, "kJ/kgK", "kJ/kg/K")
# money (no exchange rates: a currency is just "money" with factor 1)
_add("money", "money", 1.0, "₹", "$", "Rs", "Rs.", "INR", "USD", "rupee", "rupees", "dollar", "dollars", "€", "EUR", "euro", "euros")
# dimensionless
_add("%", "dimensionless", 0.01, "percent", "per cent", "pct")
_add("", "dimensionless", 1.0)
# temperature (affine: factor unused)
_add("K", "temperature", 1.0, "kelvin", "kelvins")
_add("degC", "temperature", 1.0, "°C", "℃", "C", "celsius", "deg C", "degrees C", "degrees Celsius")
_add("degF", "temperature", 1.0, "°F", "℉", "fahrenheit", "deg F", "degrees F", "degrees Fahrenheit")

# Lowercase fallbacks that are unambiguous in an engineering reply.
_CASELESS_FALLBACK = {
    "mpa": "MPa", "gpa": "GPa", "kpa": "kPa", "kw": "kW", "khz": "kHz", "mhz": "MHz",
    "ghz": "GHz", "kn": "kN", "hz": "Hz", "kj": "kJ", "mj": "MJ", "kwh": "kWh",
    "pa": "Pa", "w": "W", "j": "J", "n": "N", "v": "V", "a": "A", "kv": "kV",
}


def lookup(unit: str) -> Optional[str]:
    """Canonical unit name for a spelling, or None if it is not a unit."""
    if unit is None:
        return None
    key = _squash(unit)
    if key in _ALIASES:
        return _ALIASES[key]
    # Only an all-lowercase spelling gets the forgiving lookup: "mpa" can only
    # mean MPa in an engineering answer, but "mPa" is a real (different) unit.
    if key == key.lower():
        return _CASELESS_FALLBACK.get(key)
    return None


def dimension(unit: str) -> Optional[str]:
    canonical = lookup(unit)
    return _UNITS[canonical][0] if canonical is not None else None


def to_si(value: float, unit: str) -> float:
    canonical = lookup(unit)
    if canonical is None:
        raise KeyError(f"unknown unit {unit!r}")
    dim, factor = _UNITS[canonical]
    if dim == "temperature":
        if canonical == "K":
            return value
        if canonical == "degC":
            return value + 273.15
        return (value - 32.0) * 5.0 / 9.0 + 273.15
    return value * factor


def from_si(value: float, unit: str) -> float:
    canonical = lookup(unit)
    if canonical is None:
        raise KeyError(f"unknown unit {unit!r}")
    dim, factor = _UNITS[canonical]
    if dim == "temperature":
        if canonical == "K":
            return value
        if canonical == "degC":
            return value - 273.15
        return (value - 273.15) * 9.0 / 5.0 + 32.0
    return value / factor


def convert(value: float, src: str, dst: str) -> float:
    if dimension(src) != dimension(dst):
        raise ValueError(f"cannot convert {src!r} ({dimension(src)}) to {dst!r} ({dimension(dst)})")
    return from_si(to_si(value, src), dst)


# --------------------------------------------------------------------------
# Text normalisation and quantity extraction
# --------------------------------------------------------------------------

_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")

_LATEX_SUBS = [
    (re.compile(r"\\(?:text|mathrm|mathbf|operatorname|textbf|mathit|rm)\s*\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\\(?:,|;|:|!| |quad|qquad)"), " "),
    (re.compile(r"\\times"), "×"),
    (re.compile(r"\\cdot"), "·"),
    (re.compile(r"\\approx"), "≈"),
    (re.compile(r"\\(?:Omega|ohm)"), "Ω"),
    (re.compile(r"\\(?:mu|micro)"), "µ"),
    (re.compile(r"\^\s*\{?\\circ\}?\s*"), "°"),
    (re.compile(r"\\degree"), "°"),
    (re.compile(r"\\%"), "%"),
    (re.compile(r"\\left|\\right"), ""),
    (re.compile(r"\\(?:d?frac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}"), r"(\1)/(\2)"),
    (re.compile(r"\^\{([^{}]*)\}"), r"^\1"),
    (re.compile(r"_\{([^{}]*)\}"), r"_\1"),
]


def normalize_text(text: str) -> str:
    """Fold LaTeX and unicode into something the quantity regex can read."""
    if not text:
        return ""
    t = text
    for pattern, repl in _LATEX_SUBS:
        t = pattern.sub(repl, t)
    t = t.replace("$", " $ ") if False else t  # currency '$' is meaningful; keep as-is
    t = t.replace("−", "-").replace("–", "-").replace("\u2009", " ").replace("\u202f", " ").replace("\xa0", " ")
    # 10⁻³ -> 10^-3 (a superscript run directly after a number or '10')
    t = re.sub(r"(?<=\d)([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)", lambda m: "^" + m.group(1).translate(_SUPERSCRIPT), t)
    return t


# A number: sign, integer (optionally with thousands separators), decimal part,
# exponent; optionally followed by an "× 10^n" multiplier.
_NUMBER = (
    r"(?P<sign>[-+])?"
    r"(?P<mant>(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:[eE](?P<eexp>[-+]?\d+))?"
    r"(?P<mult>\s*(?:×|x|\*)\s*10\s*(?:\^|\*\*)\s*(?:\((?P<pexp1>[-+]?\d+)\)|(?P<pexp2>[-+]?\d+)))?"
)
_CURRENCY_PREFIX = r"(?P<cur>₹|\$|€|Rs\.?|INR|USD|EUR)\s?"
# Numbers are found first; a unit is then read from the text that follows.
# Matching the two separately matters: a unit-shaped word that turns out not
# to be a unit ("and", "Rs") must not swallow the next number's prefix.
_NUM_RE = re.compile(r"(?<![A-Za-z0-9_.,^])(?:" + _CURRENCY_PREFIX + r")?" + _NUMBER)
_UNIT_RE = re.compile(
    r"[ \t]*(?P<unit>"
    r"(?:°\s?[CF]|℃|℉|%|[A-Za-zµμΩΩ$₹€°]"
    r"(?:[A-Za-z0-9µμΩΩ°^·⋅*/\-²³⁴]|\([^()\n]{1,24}\)|(?<=[A-Za-z]) (?=[A-Za-z]{1,2}\b))*)"
    r")"
)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.S)
_TRAILING_PUNCT = ".,;:!?)]}'\"`*_~"


@dataclass(frozen=True)
class Quantity:
    """One number in a reply, with its unit as written and converted to SI."""
    value: float          # numeric value as written (with any ×10^n applied)
    unit: str             # canonical unit name ("" if unitless)
    raw: str              # the matched text
    start: int
    end: int
    bold: bool = False
    before: str = ""      # up to 24 characters of context preceding the number
    after: str = ""       # the three characters right after the number+unit

    @property
    def dim(self) -> str:
        return _UNITS[self.unit][0]

    @property
    def si(self) -> float:
        return to_si(self.value, self.unit)


def _parse_number(m: re.Match) -> Optional[float]:
    mantissa = m.group("mant")
    if not mantissa:
        return None
    try:
        value = float(mantissa.replace(",", ""))
    except ValueError:
        return None
    if m.group("eexp"):
        value *= 10.0 ** int(m.group("eexp"))
    pexp = m.group("pexp1") or m.group("pexp2")
    if pexp:
        value *= 10.0 ** int(pexp)
    if m.group("sign") == "-":
        value = -value
    return value


def _resolve_unit(token: Optional[str]) -> tuple[str, int]:
    """Longest known unit at the start of `token`; returns (canonical, length used)."""
    if not token:
        return "", 0
    token = token.rstrip()
    # Only strip punctuation, never letters: "steps" must not become "s".
    trimmed = token
    while trimmed and trimmed[-1] in _TRAILING_PUNCT:
        candidate = lookup(trimmed)
        if candidate is not None:
            return candidate, len(trimmed)
        trimmed = trimmed[:-1]
    candidate = lookup(trimmed)
    if candidate is not None:
        return candidate, len(trimmed)
    # "kN(approx.)": a parenthetical glued to the unit is not part of it.
    if "(" in trimmed:
        head = trimmed.split("(", 1)[0]
        candidate = lookup(head) if head else None
        if candidate is not None:
            return candidate, len(head)
    # Two-word units like "deg C" / "kN m" are matched with a space; if the
    # whole thing is not a unit, fall back to the first word alone.
    if " " in trimmed:
        first = trimmed.split(" ", 1)[0]
        while first and first[-1] in _TRAILING_PUNCT:
            first = first[:-1]
        candidate = lookup(first)
        if candidate is not None:
            return candidate, len(first)
    return "", 0


def _bold_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _BOLD_RE.finditer(text)]


def extract_quantities(text: str) -> list[Quantity]:
    """Every number in `text`, each with a resolved (possibly empty) unit."""
    t = normalize_text(text)
    bold = _bold_spans(t)
    out: list[Quantity] = []
    consumed = 0  # end of the last unit read, so "mm^2" never yields a "2"
    for m in _NUM_RE.finditer(t):
        if m.start() < consumed:
            continue
        value = _parse_number(m)
        if value is None or math.isnan(value) or math.isinf(value):
            continue
        end = m.end()
        unit = "money" if m.group("cur") else ""
        if not unit:
            um = _UNIT_RE.match(t, end)
            if um:
                resolved, used = _resolve_unit(um.group("unit"))
                if resolved:
                    unit = resolved
                    end = um.start("unit") + used
        consumed = end
        start = m.start()
        in_bold = any(a <= start < b for a, b in bold)
        out.append(Quantity(value=value, unit=unit, raw=t[start:end], start=start, end=end, bold=in_bold,
                            before=t[max(0, start - 24):start], after=t[end:end + 3]))
    return out


def same_value(a: float, b: float, rel_tol: float, abs_tol: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
