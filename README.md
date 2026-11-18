# Multi-Airfoil Aerodynamics Workbench

This Streamlit application compares NACA 4-digit airfoils and continuously
morphed wing shapes, estimates aerodynamic forces, and translates aerodynamic
drag into aircraft fuel or automotive fuel/EV energy use. These are engineering
estimates for comparative studies, not validated CFD or certified performance
predictions.

## Run the application

Install the packages listed in `requirements.txt`, then run:

```powershell
python -m streamlit run app.py
```

The requirements remain limited to `numpy`, `plotly`, and `streamlit`.

## Airfoil shape generation

The geometry functions accept either a four-digit NACA code or continuous shape
parameters (`max_camber_pct`, `camber_pos_pct`, and `thickness_pct`), with an
optional `flap_deg` setting. The parametric workbench supports camber from 0–6%,
camber position from 20–60% chord, thickness from 8–20%, and flap angles from
-30° to +30°. Positive flap deflection rotates the trailing portion downward.

The first NACA digit gives maximum camber in percent chord, the second gives
camber position in tenths of chord, and the final two digits give maximum
thickness in percent chord. `generate_parametric_airfoil` retains continuous
slider precision rather than rounding slider values to a NACA code.

## Aerodynamic estimates

Lift uses the thin-airfoil lift slope (`2π` per radian) and a numerically
integrated zero-lift angle from the NACA mean camber line. A deflected flap adds
an approximate half-flap-angle increment. The drag polar adds a thickness and
flap profile-drag term to the induced-drag estimate. Stall is flagged at an
effective angle of attack of 15°; past stall, lift is reduced and drag is
increased. This intentionally compact model is useful for comparisons, not
high-fidelity prediction.

## 100-airfoil batch sweep

The optimizer evaluates 100 distinct, valid NACA 4-digit profiles. It spans
0–6% camber, 20–60% camber position, and 8–20% thickness with deterministic
sampling of the valid digit combinations. Force coefficients and lift, drag,
L/D, and stall state are calculated as NumPy arrays. The leaderboard is sorted
by L/D, and the app provides an interactive drag-versus-L/D scatter plot and a
CSV download.

## ISA altitude density

`calculate_air_density` uses the International Standard Atmosphere troposphere
lapse-rate model from sea level through 12,000 m:

```text
T = 288.15 K - 0.0065 K/m × altitude
ρ = 1.225 kg/m³ × (T / 288.15 K)^(g / (R × 0.0065) - 1)
```

Here `g = 9.80665 m/s²` and `R = 287.05 J/(kg·K)`. Altitudes outside the
supported 0–12,000 m interval are rejected.

## Fuel and energy formulas

The calculation starts with dynamic pressure and aerodynamic drag:

```text
q = ½ ρ V²
D = q × area × Cd
Power = D × V
```

Aircraft thrust power is reported as `D × cruise speed` in kW. Fuel energy
input is estimated using 38% overall efficiency and 9.6 kWh/L for Jet fuel, or
30% and 8.7 kWh/L for Avgas; kg/100 km uses fuel densities of 0.80 and 0.72
kg/L, respectively. Liters/hour are `shaft power (kW) / (fuel energy density
(kWh/L) × efficiency)`. Fuel mass per 100 km converts the hourly mass rate by
cruise distance covered.

Automotive impact is evaluated at a fixed 100 km/h highway speed. Gasoline
assumes 8.9 kWh/L and 25% engine efficiency; EV assumes 90% battery-to-wheel
efficiency. Wheel energy per 100 km is `D × 100,000 m / 3,600,000 J/kWh`;
divide by gasoline energy per liter and engine efficiency for L/100 km, or by
EV drivetrain efficiency for kWh/100 km. The displayed additional consumption
is the current shape result minus the locked baseline result, so lower-drag
shapes can produce a negative delta (a modeled saving).

## Material, structure, and custom wing model

The material selector supplies density, surface roughness height, elastic
modulus, and an equivalent structure-volume fraction for carbon/epoxy, 6061
aluminum, fiberglass, ABS, and raw-layer 3D-printed material. The mass estimate
is `density × planform area × (mean chord × thickness ratio) × structure
fraction`; it is a comparative airfoil-envelope estimate, not a CAD-derived
bill of materials. Material-specific roughness modifies the smooth turbulent
skin-friction estimate using `Re_k = Re × ks / mean chord` and a documented
semi-empirical roughness multiplier. The reported `ΔCd0` is the additional
profile drag coefficient from that correction.

The tapered planform uses `area = span × (root chord + tip chord) / 2` and the
trapezoid mean aerodynamic chord. Taper and endplate/winglet height modify the
estimated Oswald efficiency and effective aspect ratio used in induced drag.
Gurney flap height adds a square-root lift increment plus a profile-drag
increment; an inverted section reverses the lift sign. These are compact
comparative models rather than CFD.

Elastic tip deflection uses a uniformly loaded cantilever estimate,
`δtip = 5 w L⁴ / (384 E I)`, with an equivalent section inertia based on the
material volume fraction. A bounded fraction of `δtip / L` is applied as lift
loss to represent aeroelastic flex. This deliberately does not resolve wing
twist, laminate layup, spars, joints, or nonlinear deformation.

Selig `.dat` uploads are parsed as numeric x/y pairs, normalized by chord, and
split into upper and lower surfaces at `argmin(x)`. Cp curves use a thin-airfoil
leading-edge pressure approximation from the calculated lift coefficient;
they visualize comparative pressure trends and are not a viscous panel/CFD
solution. The S1223 preset is explicitly an S1223-inspired parametric
approximation; upload measured coordinates when exact S1223 geometry is needed.

## Interface

- **Live Wind Tunnel & Fuel Impact:** overlays the morphed and baseline airfoil
  with illustrative curved streamlines, a Cp chart, telemetry (Re, Mach, ISA
  density), material mass/roughness estimates, and lift/drag polar curves. Live
  deltas use the locked baseline configuration at current conditions.
- **100-Airfoil Batch Optimizer:** runs the vectorized sweep, highlights the
  five best designs, displays the sortable leaderboard, exports CSV, and can
  load the top-ranked NACA profile into the sidebar controls.
- **Aircraft Cruise / Automotive Aero:** chooses the appropriate aircraft fuel
  or gasoline/EV energy estimate. Automotive aerodynamic impact is isolated to
  the modeled wing/spoiler area.
