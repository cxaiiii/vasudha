# Vasudha — pre-release evaluation sheet

Twenty-eight prompts to run through the app before pushing a build. Every
numeric answer below was computed in Python, not taken from a textbook or from
the model — if the app disagrees with this sheet, the app is wrong.

**How to score.** Accept anything within ~2% of the stated value unless a
tighter figure is noted. What matters is not the last decimal but whether the
*right equation* was used and whether the *power of ten* is right — the two
things this model has actually been observed to get wrong.

Watch three things on every answer:

1. **Did the tool card appear?** No card means the number was produced in the
   model's head. Even a correct answer without a tool run is a fail — it got
   lucky, and you cannot tell the difference at a glance.
2. **Is the magnitude right?** The characteristic failure here is a correct
   setup followed by an answer off by exactly 10× or 1000×.
3. **Does the stated unit match the number?** A real observed failure was the
   tool printing `0.016` (metres) and the reply saying "0.016 mm".

---

## A. Core numeric accuracy

These are the product's central claim. All should pass.

| # | Prompt | Expected |
|---|---|---|
| A1 | A steel cantilever beam is 2 m long, 50 mm wide and 100 mm deep, with a 5 kN point load at the free end. E = 200 GPa. Maximum tip deflection in mm? | **16 mm** ⚠️ known failure |
| A2 | Same beam and load — maximum bending stress in MPa? | **120 MPa** |
| A3 | A simply supported beam, same section, 2 m span, 5 kN at midspan. Deflection at the centre in mm? | **1 mm** |
| A4 | A 12 mm diameter steel rod carries 25 kN in tension. Stress in MPa? | **221 MPa** |
| A5 | That rod is 1.5 m long, E = 200 GPa. How much does it stretch, in mm? | **1.66 mm** |
| A6 | Water (998 kg/m³, 1.002e-3 Pa·s) flows at 2 m/s through a 25 mm pipe. Reynolds number? | **49,800** |
| A7 | For that flow, estimate the Darcy friction factor using the Blasius correlation. | **0.0212** |
| A8 | And the head loss over 10 m of that pipe, in metres? | **1.73 m** |
| A9 | Water discharges from a tank through a small orifice 3 m below the surface. Efflux velocity? | **7.67 m/s** |
| A10 | A plane wall is 0.2 m thick, k = 0.8 W/(m·K), area 10 m², inner face 30 °C, outer 5 °C, steady state. Heat transfer rate in W? | **1000 W** |
| A11 | Energy to heat 2 kg of water from 20 °C to 80 °C, c = 4186 J/(kg·K), in kJ? | **502 kJ** |
| A12 | A 25 m steel rail warms by 40 K, α = 12e-6 /K. Expansion in mm? | **12 mm** |
| A13 | RC low-pass filter, R = 4.7 kΩ, C = 100 nF. −3 dB cutoff frequency? | **338.6 Hz** |
| A14 | Resonant frequency of an LC circuit with L = 10 mH and C = 100 nF? | **5033 Hz** |
| A15 | 330 Ω and 470 Ω in parallel — equivalent resistance? | **193.9 Ω** |
| A16 | Power dissipated in a 470 Ω resistor across 12 V? | **0.306 W** |
| A17 | How many moles in 25 g of NaCl (M = 58.44 g/mol)? | **0.4278 mol** |
| A18 | Mass of 0.35 mol of H₂SO₄ (M = 98.08 g/mol)? | **34.33 g** |
| A19 | pH of a 1e-3 M HCl solution? | **3** |
| A20 | A block 20 × 30 × 5 mm has mass 23.4 g. Density in g/cm³, and what metal is it likely to be? | **7.8 g/cm³, steel/iron** |
| A21 | ₹50,000 invested at 7.5% compounded annually for 6 years — final amount? | **₹77,165** |
| A22 | Population standard deviation of 2, 4, 4, 4, 5, 5, 7, 9? | **2** |
| A23 | A projectile launched at 30 m/s, 40° above horizontal, g = 9.81, no drag. Horizontal range? | **90.35 m** |
| A24 | Maximum height for that same projectile? | **18.95 m** |

⚠️ **A1 is a known failure.** The model writes `PL³/(48EI)` — the
simply-supported centre-load formula — for a cantilever, which is `PL³/(3EI)`.
Python then computes its wrong formula correctly, so the answer comes back
confident and wrong (~1 mm or ~4 mm depending on the run). Interestingly it
gets **A3 right**, because there the /48 formula is the correct one. This is a
knowledge gap in the weights, not something the harness can fix, and it is the
single best argument for the retraining you costed earlier.

---

## B. Traps — magnitude and units

These target the observed failure mode directly. If any of these come back
off by a factor of 10, 100 or 1000, do not push.

| # | Prompt | Expected |
|---|---|---|
| B1 | Convert 2.5 GPa to N/mm². | **2500 N/mm²** |
| B2 | A capacitor is 4700 µF. Express it in farads in scientific notation. | **4.7e-3 F** |
| B3 | A pressure of 1 atm in kPa, and in psi? | **101.3 kPa, 14.70 psi** |
| B4 | 15 factorial? | **1,307,674,368,000** |

---

## C. Behaviour — not everything is a calculation

Numeric accuracy is not the whole product. These check it behaves sensibly.

| # | Prompt | What good looks like |
|---|---|---|
| C1 | What is the capital of India? | Answers "New Delhi" directly. **Should NOT run Python.** Firing the tool for this is a fail — it means the tool rule is over-applied. |
| C2 | Explain in two sentences why a hollow tube is stiffer than a solid rod of the same mass. | Two sentences, correct (material moved away from the neutral axis raises I). Watch for over-explaining — a wall of text is a regression of the stage-2 fix. |
| C3 | What is the current price of copper today? | Should say it cannot know / has no live data, **not invent a number**. Fabricating a price here is the worst failure on this sheet. |
| C4 | Write a Python function that returns the nth Fibonacci number, and test it for n=30. | Writes it, **runs it**, reports 832040. |
| C5 | I measured a beam deflection of 16 mm but calculated 12 mm. What are three likely causes? | Sensible engineering answers (E value, boundary conditions, load position). No fabricated citations. |
| C6 | What is 2 + 2? | "4", instantly, no tool, no reasoning block. |

---

## D. Consistency

Ask **A13 three times in separate chats.** All three should give 338.6 Hz.
Variation here means the temperature is too high for engineering use — drop it
in Settings → Answers.

---

## Recording results

A quick tally is enough:

```
Section A (24):  __ / 24    (A1 expected to fail)
Section B  (4):  __ / 4
Section C  (6):  __ / 6
Section D:       consistent?  Y / N
Tool card appeared on every A and B answer?  Y / N
```

**Suggested bar for pushing:** A ≥ 21/24, B = 4/4, C ≥ 5/6, D consistent, and a
tool card on every single numeric answer. If B is not perfect, the magnitude
problem is still live and the README should say so plainly rather than let
someone discover it on their own work.

Six of these (A1, A2, A6, A10, A13, A23) are automated in
[scripts/bench_numeric.py](../scripts/bench_numeric.py) — run that first, it
takes about two minutes and catches regressions without any typing.
