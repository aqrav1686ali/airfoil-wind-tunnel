"""Airfoil geometry, comparative aerodynamic estimates, and energy models."""

import numpy as np


MATERIAL_PROPERTIES = {
    "Autoclave Carbon Fiber (Mirror Gloss)": {
        "density_kg_m3": 1600.0,
        "roughness_m": 0.4e-6,
        "youngs_modulus_pa": 70e9,
        "structure_fraction": 0.055,
    },
    "Aerospace Aluminum 6061 (Anodized)": {
        "density_kg_m3": 2700.0,
        "roughness_m": 1.6e-6,
        "youngs_modulus_pa": 69e9,
        "structure_fraction": 0.08,
    },
    "Fiberglass Composite (Gelcoat)": {
        "density_kg_m3": 1850.0,
        "roughness_m": 3.0e-6,
        "youngs_modulus_pa": 25e9,
        "structure_fraction": 0.075,
    },
    "Injection-Molded ABS Plastic": {
        "density_kg_m3": 1040.0,
        "roughness_m": 12.0e-6,
        "youngs_modulus_pa": 2.1e9,
        "structure_fraction": 0.10,
    },
    "3D-Printed PLA/PETG (Raw Layer Lines)": {
        "density_kg_m3": 1240.0,
        "roughness_m": 75.0e-6,
        "youngs_modulus_pa": 2.5e9,
        "structure_fraction": 0.12,
    },
}


def _validate_naca4_code(code: str) -> str:
    code = str(code).strip()
    if len(code) != 4 or not code.isdigit():
        raise ValueError("Airfoil code must contain exactly four digits.")
    return code


def calculate_isa_properties(altitude_m: float) -> dict:
    """Return ISA troposphere temperature, pressure, density, viscosity, and sound speed."""
    if not np.isfinite(altitude_m) or not 0.0 <= altitude_m <= 12000.0:
        raise ValueError("Altitude must be between 0 and 12,000 m.")
    temperature_k = 288.15 - 0.0065 * altitude_m
    gas_constant = 287.05
    gravity = 9.80665
    lapse_rate = 0.0065
    pressure_pa = 101325.0 * (temperature_k / 288.15) ** (
        gravity / (gas_constant * lapse_rate)
    )
    density_kg_m3 = pressure_pa / (gas_constant * temperature_k)
    viscosity_pa_s = (
        1.716e-5
        * (temperature_k / 273.15) ** 1.5
        * (273.15 + 110.4)
        / (temperature_k + 110.4)
    )
    speed_of_sound_ms = float(np.sqrt(1.4 * gas_constant * temperature_k))
    return {
        "temperature_k": temperature_k,
        "pressure_pa": pressure_pa,
        "air_density_kg_m3": density_kg_m3,
        "dynamic_viscosity_pa_s": viscosity_pa_s,
        "speed_of_sound_ms": speed_of_sound_ms,
    }


def calculate_air_density(altitude_m: float) -> float:
    """Return ISA density for the troposphere, limited to 0–12,000 m."""
    return float(calculate_isa_properties(altitude_m)["air_density_kg_m3"])


def calculate_reynolds_and_mach(
    chord_length_m: float, velocity_ms: float, altitude_m: float
):
    """Return Reynolds and Mach numbers using ISA viscosity and sound speed."""
    if not np.isfinite(chord_length_m) or chord_length_m <= 0.0:
        raise ValueError("Chord length must be a positive finite number.")
    if not np.isfinite(velocity_ms) or velocity_ms < 0.0:
        raise ValueError("Velocity must be a non-negative finite number.")
    isa = calculate_isa_properties(altitude_m)
    reynolds = (
        isa["air_density_kg_m3"] * velocity_ms * chord_length_m
        / isa["dynamic_viscosity_pa_s"]
    )
    mach = velocity_ms / isa["speed_of_sound_ms"]
    return float(reynolds), float(mach)


def _validate_parametric_shape(
    max_camber_pct: float,
    camber_pos_pct: float,
    thickness_pct: float,
    flap_deg: float,
) -> None:
    values = (max_camber_pct, camber_pos_pct, thickness_pct, flap_deg)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Airfoil shape parameters must be finite numbers.")
    if not 0.0 <= max_camber_pct <= 6.0:
        raise ValueError("Maximum camber must be between 0 and 6%.")
    if not 20.0 <= camber_pos_pct <= 60.0:
        raise ValueError("Camber position must be between 20 and 60%.")
    if not 8.0 <= thickness_pct <= 20.0:
        raise ValueError("Thickness must be between 8 and 20%.")
    _validate_flap(flap_deg)


def _validate_flap(flap_deg: float) -> None:
    if not np.isfinite(flap_deg) or not -30.0 <= flap_deg <= 30.0:
        raise ValueError("Flap angle must be between -30 and 30 degrees.")


def _airfoil_surfaces(
    max_camber: float,
    camber_position: float,
    thickness: float,
    flap_deg: float,
    num_points: int,
):
    x = np.linspace(0.0, 1.0, num_points)
    if max_camber == 0.0 or camber_position == 0.0:
        camber_line = np.zeros_like(x)
    else:
        front = x < camber_position
        camber_line = np.where(
            front,
            max_camber / camber_position**2
            * (2.0 * camber_position * x - x**2),
            max_camber / (1.0 - camber_position) ** 2
            * (
                (1.0 - 2.0 * camber_position)
                + 2.0 * camber_position * x
                - x**2
            ),
        )
    half_thickness = 5.0 * thickness * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )
    upper = camber_line + half_thickness
    lower = camber_line - half_thickness

    if flap_deg != 0.0:
        hinge_x = 0.75
        rotation = -np.radians(flap_deg)
        cosine = np.cos(rotation)
        sine = np.sin(rotation)
        flap = x >= hinge_x
        for surface in (upper, lower):
            hinge_y = np.interp(hinge_x, x, surface)
            dx = x[flap] - hinge_x
            dy = surface[flap] - hinge_y
            surface[flap] = hinge_y + dx * sine + dy * cosine
    return x, upper, lower


def generate_parametric_airfoil(
    max_camber_pct: float = 2.0,
    camber_pos_pct: float = 40.0,
    thickness_pct: float = 12.0,
    flap_deg: float = 0.0,
    num_points: int = 120,
):
    """Generate an airfoil from continuous percentage controls and a hinged flap."""
    _validate_parametric_shape(
        max_camber_pct, camber_pos_pct, thickness_pct, flap_deg
    )
    if not isinstance(num_points, int) or num_points < 2:
        raise ValueError("num_points must be an integer of at least 2.")
    return _airfoil_surfaces(
        max_camber_pct / 100.0,
        camber_pos_pct / 100.0,
        thickness_pct / 100.0,
        flap_deg,
        num_points,
    )


def generate_naca4_airfoil(
    code: str | None = None,
    num_points: int = 120,
    *,
    max_camber_pct: float = 2.0,
    camber_pos_pct: float = 40.0,
    thickness_pct: float = 12.0,
    flap_deg: float = 0.0,
):
    """Generate from a NACA 4-digit code, or continuous shape parameters."""
    if not isinstance(num_points, int) or num_points < 2:
        raise ValueError("num_points must be an integer of at least 2.")
    if code is not None:
        code = _validate_naca4_code(code)
        max_camber_pct = int(code[0])
        camber_pos_pct = int(code[1]) * 10.0
        thickness_pct = int(code[2:])
        if max_camber_pct == 0:
            camber_pos_pct = 40.0
        _validate_flap(flap_deg)
        return _airfoil_surfaces(
            max_camber_pct / 100.0,
            camber_pos_pct / 100.0,
            thickness_pct / 100.0,
            flap_deg,
            num_points,
        )
    return generate_parametric_airfoil(
        max_camber_pct,
        camber_pos_pct,
        thickness_pct,
        flap_deg,
        num_points,
    )


def generate_streamlines(angle_deg: float, num_points: int = 100):
    """Return a simple visualization field that varies with angle of attack."""
    x = np.linspace(-1.0, 1.0, num_points)
    return x, np.cos(np.radians(angle_deg)) * x


def generate_pressure_distribution(angle_deg: float, num_points: int = 100):
    """Return a normalized illustrative pressure-difference distribution."""
    x = np.linspace(0.0, 1.0, num_points)
    pressure_difference = 2.0 * np.pi * np.radians(angle_deg) * np.sqrt(
        np.maximum((1.0 - x) / np.maximum(x, 1e-6), 0.0)
    )
    return x, pressure_difference


def calculate_pressure_coefficients(x_over_c, lift_coeff_cl: float):
    """Estimate upper/lower Cp from a thin-airfoil leading-edge pressure model."""
    x = np.asarray(x_over_c, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("x/c coordinates must be a finite one-dimensional array.")
    if not np.isfinite(lift_coeff_cl):
        raise ValueError("Lift coefficient must be finite.")
    x_safe = np.clip(x, 0.005, 1.0)
    pressure_difference = lift_coeff_cl * np.sqrt(
        np.maximum((1.0 - x_safe) / x_safe, 0.0)
    )
    return -0.5 * pressure_difference, 0.5 * pressure_difference


def _zero_lift_angle(max_camber: np.ndarray, camber_position: np.ndarray):
    theta = np.linspace(0.0, np.pi, 181)
    x = (1.0 - np.cos(theta)) / 2.0
    x_values = x.reshape((1,) * np.ndim(max_camber) + (x.size,))
    camber = np.asarray(max_camber)[..., np.newaxis]
    position = np.asarray(camber_position)[..., np.newaxis]
    safe_position = np.clip(position, 1e-6, 1.0 - 1e-6)
    slope = np.where(
        x_values < safe_position,
        2.0 * camber / safe_position**2 * (safe_position - x_values),
        2.0 * camber / (1.0 - safe_position) ** 2 * (safe_position - x_values),
    )
    integrand = slope * (1.0 - np.cos(theta))
    integral = np.sum(
        (integrand[..., 1:] + integrand[..., :-1]) * np.diff(theta) / 2.0,
        axis=-1,
    )
    return integral / np.pi


def _profile_zero_lift_angle(profile_coordinates: dict):
    x_upper = np.asarray(profile_coordinates["x_upper"], dtype=float)
    y_upper = np.asarray(profile_coordinates["y_upper"], dtype=float)
    x_lower = np.asarray(profile_coordinates["x_lower"], dtype=float)
    y_lower = np.asarray(profile_coordinates["y_lower"], dtype=float)
    if min(x_upper.size, y_upper.size, x_lower.size, y_lower.size) < 3:
        raise ValueError("Custom airfoil coordinates need at least three points per surface.")
    if x_upper.size != y_upper.size or x_lower.size != y_lower.size:
        raise ValueError("Custom airfoil x and y coordinate lengths must match.")
    if not all(
        np.all(np.isfinite(values))
        for values in (x_upper, y_upper, x_lower, y_lower)
    ):
        raise ValueError("Custom airfoil coordinates must be finite.")
    upper_order = np.argsort(x_upper)
    lower_order = np.argsort(x_lower)
    grid = np.linspace(0.005, 0.995, 201)
    upper = np.interp(grid, x_upper[upper_order], y_upper[upper_order])
    lower = np.interp(grid, x_lower[lower_order], y_lower[lower_order])
    mean_line = 0.5 * (upper + lower)
    thickness_pct = 100.0 * float(np.max(upper - lower))

    theta = np.linspace(0.01, np.pi - 0.01, 201)
    x_theta = (1.0 - np.cos(theta)) / 2.0
    mean_line_slope = np.gradient(mean_line, grid)
    slope = np.interp(x_theta, grid, mean_line_slope)
    integrand = slope * (1.0 - np.cos(theta))
    integral = np.sum(
        (integrand[1:] + integrand[:-1]) * np.diff(theta) / 2.0
    )
    zero_lift_angle = float(integral / np.pi)
    return zero_lift_angle, thickness_pct


def _resolve_planform(
    wing_area_m2: float,
    span_m: float | None,
    root_chord_m: float | None,
    tip_chord_m: float | None,
    winglet_height_m: float,
):
    provided = (span_m, root_chord_m, tip_chord_m)
    if any(value is not None for value in provided):
        if any(value is None for value in provided):
            raise ValueError("Specify span, root chord, and tip chord together.")
        if not all(np.isfinite(value) and value > 0.0 for value in provided):
            raise ValueError("Span and root/tip chords must be positive and finite.")
        area = span_m * (root_chord_m + tip_chord_m) / 2.0
        if not np.isfinite(wing_area_m2) or wing_area_m2 <= 0.0:
            raise ValueError("Wing area must be positive and finite.")
    else:
        if not np.isfinite(wing_area_m2) or wing_area_m2 <= 0.0:
            raise ValueError("Wing area must be positive and finite.")
        span_m = np.sqrt(8.0 * wing_area_m2)
        root_chord_m = wing_area_m2 / span_m
        tip_chord_m = root_chord_m
        area = wing_area_m2
    if not np.isfinite(winglet_height_m) or winglet_height_m < 0.0:
        raise ValueError("Winglet/endplate height must be non-negative and finite.")

    taper_ratio = tip_chord_m / root_chord_m
    mean_aerodynamic_chord_m = (
        2.0
        / 3.0
        * root_chord_m
        * (1.0 + taper_ratio + taper_ratio**2)
        / (1.0 + taper_ratio)
    )
    effective_span_m = span_m + 2.0 * winglet_height_m
    effective_aspect_ratio = effective_span_m**2 / area
    oswald_efficiency = np.clip(
        0.78
        + 0.15 * (1.0 - abs(taper_ratio - 0.45) / 0.55)
        + 0.06 * (2.0 * winglet_height_m / span_m),
        0.65,
        0.95,
    )
    return {
        "wing_area_m2": float(area),
        "span_m": float(span_m),
        "root_chord_m": float(root_chord_m),
        "tip_chord_m": float(tip_chord_m),
        "mean_aerodynamic_chord_m": float(mean_aerodynamic_chord_m),
        "taper_ratio": float(taper_ratio),
        "effective_aspect_ratio": float(effective_aspect_ratio),
        "oswald_efficiency": float(oswald_efficiency),
    }


def calculate_material_effects(
    material: str,
    wing_area_m2: float,
    airspeed_ms: float,
    altitude_m: float,
    thickness_pct: float,
    lift_coeff_cl: float = 0.0,
    *,
    span_m: float | None = None,
    root_chord_m: float | None = None,
    tip_chord_m: float | None = None,
    winglet_height_m: float = 0.0,
    air_density: float | None = None,
):
    """Estimate material mass, roughness drag, and cantilever-flex lift loss.

    Mass uses a material-volume fraction of the airfoil envelope. Flex is an
    elastic cantilever deflection surrogate, not a detailed shell/FEA solution.
    """
    if material not in MATERIAL_PROPERTIES:
        raise ValueError("Select one of the supported wing/spoiler materials.")
    if not np.isfinite(airspeed_ms) or airspeed_ms < 0.0:
        raise ValueError("Airspeed must be a non-negative finite number.")
    if not np.isfinite(thickness_pct) or thickness_pct <= 0.0:
        raise ValueError("Thickness must be a positive finite percentage.")
    if not np.isfinite(lift_coeff_cl):
        raise ValueError("Lift coefficient must be finite.")
    planform = _resolve_planform(
        wing_area_m2,
        span_m,
        root_chord_m,
        tip_chord_m,
        winglet_height_m,
    )
    isa = calculate_isa_properties(altitude_m)
    density = (
        isa["air_density_kg_m3"] if air_density is None else float(air_density)
    )
    if not np.isfinite(density) or density <= 0.0:
        raise ValueError("Air density must be a positive finite number.")
    properties = MATERIAL_PROPERTIES[material]
    chord = planform["mean_aerodynamic_chord_m"]
    reynolds = density * airspeed_ms * chord / isa["dynamic_viscosity_pa_s"]
    if reynolds < 1.0:
        smooth_skin_friction = 0.0
    elif reynolds < 500000.0:
        smooth_skin_friction = 0.074 / reynolds**0.2 - 1742.0 / reynolds
    else:
        smooth_skin_friction = (
            0.455 / np.log10(reynolds) ** 2.58 - 1700.0 / reynolds
        )
    roughness_reynolds = (
        reynolds * properties["roughness_m"] / chord
        if chord > 0.0
        else 0.0
    )
    roughness_factor = 1.0 + 0.15 * np.log1p(roughness_reynolds / 30.0) ** 2
    roughness_drag_penalty_cd = max(
        0.0, 2.0 * smooth_skin_friction * (roughness_factor - 1.0) * 2.0
    )

    thickness_depth_m = chord * thickness_pct / 100.0
    material_volume_m3 = (
        planform["wing_area_m2"]
        * thickness_depth_m
        * properties["structure_fraction"]
    )
    structural_weight_kg = material_volume_m3 * properties["density_kg_m3"]
    semispan_m = planform["span_m"] / 2.0
    section_inertia_m4 = (
        properties["structure_fraction"]
        * chord
        * thickness_depth_m**3
        / 12.0
    )
    dynamic_pressure_pa = 0.5 * density * airspeed_ms**2
    load_per_span_n_m = (
        dynamic_pressure_pa * chord * abs(lift_coeff_cl)
    )
    flexural_rigidity = properties["youngs_modulus_pa"] * section_inertia_m4
    tip_deflection_m = (
        5.0 * load_per_span_n_m * semispan_m**4 / (384.0 * flexural_rigidity)
        if flexural_rigidity > 0.0
        else 0.0
    )
    aeroelastic_lift_loss_pct = min(
        35.0, 100.0 * 0.35 * tip_deflection_m / max(semispan_m, 1e-9)
    )
    return {
        **planform,
        "air_density_kg_m3": density,
        "reynolds_number": float(reynolds),
        "mach_number": float(airspeed_ms / isa["speed_of_sound_ms"]),
        "structural_weight_kg": float(structural_weight_kg),
        "roughness_m": properties["roughness_m"],
        "roughness_drag_penalty_cd": float(roughness_drag_penalty_cd),
        "tip_deflection_m": float(tip_deflection_m),
        "aeroelastic_lift_loss_pct": float(aeroelastic_lift_loss_pct),
        "dynamic_pressure_pa": float(dynamic_pressure_pa),
    }


def _profile_zero_lift_angle(profile_coordinates: dict):
    x_upper = np.asarray(profile_coordinates["x_upper"], dtype=float)
    y_upper = np.asarray(profile_coordinates["y_upper"], dtype=float)
    x_lower = np.asarray(profile_coordinates["x_lower"], dtype=float)
    y_lower = np.asarray(profile_coordinates["y_lower"], dtype=float)
    if min(x_upper.size, y_upper.size, x_lower.size, y_lower.size) < 3:
        raise ValueError("Custom airfoil coordinates need at least three points per surface.")
    if x_upper.size != y_upper.size or x_lower.size != y_lower.size:
        raise ValueError("Custom airfoil x and y coordinate lengths must match.")
    if not all(
        np.all(np.isfinite(values))
        for values in (x_upper, y_upper, x_lower, y_lower)
    ):
        raise ValueError("Custom airfoil coordinates must be finite.")
    upper_order = np.argsort(x_upper)
    lower_order = np.argsort(x_lower)
    grid = np.linspace(0.005, 0.995, 201)
    upper = np.interp(grid, x_upper[upper_order], y_upper[upper_order])
    lower = np.interp(grid, x_lower[lower_order], y_lower[lower_order])
    mean_line = 0.5 * (upper + lower)
    thickness_pct = 100.0 * float(np.max(upper - lower))
    theta = np.linspace(0.01, np.pi - 0.01, 201)
    x_theta = (1.0 - np.cos(theta)) / 2.0
    mean_line_slope = np.gradient(mean_line, grid)
    slope = np.interp(x_theta, grid, mean_line_slope)
    integrand = slope * (1.0 - np.cos(theta))
    zero_lift_angle = float(np.trapezoid(integrand, theta) / np.pi)
    return zero_lift_angle, thickness_pct


def calculate_aero_forces(
    naca_code: str | None,
    angle_deg: float,
    airspeed_ms: float,
    wing_area_m2: float,
    air_density: float | None = None,
    *,
    max_camber_pct: float = 2.0,
    camber_pos_pct: float = 40.0,
    thickness_pct: float = 12.0,
    flap_deg: float = 0.0,
    altitude_m: float = 0.0,
    material: str = "Autoclave Carbon Fiber (Mirror Gloss)",
    span_m: float | None = None,
    root_chord_m: float | None = None,
    tip_chord_m: float | None = None,
    winglet_height_m: float = 0.0,
    gurney_flap_pct: float = 0.0,
    inverted: bool = False,
    profile_coordinates: dict | None = None,
):
    """Estimate forces for parametric, NACA, or uploaded airfoil coordinates."""
    values = (angle_deg, airspeed_ms, wing_area_m2)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Angle, speed, and area must be finite numbers.")
    if airspeed_ms < 0.0:
        raise ValueError("Airspeed must be non-negative.")
    _validate_flap(flap_deg)
    if not np.isfinite(gurney_flap_pct) or not 0.0 <= gurney_flap_pct <= 5.0:
        raise ValueError("Gurney flap height must be between 0 and 5% chord.")
    isa = calculate_isa_properties(altitude_m)
    density = isa["air_density_kg_m3"] if air_density is None else air_density
    if not np.isfinite(density) or density <= 0.0:
        raise ValueError("Air density must be a positive finite number.")
    planform = _resolve_planform(
        wing_area_m2, span_m, root_chord_m, tip_chord_m, winglet_height_m
    )

    if profile_coordinates is not None:
        zero_lift_angle, thickness_pct = _profile_zero_lift_angle(
            profile_coordinates
        )
    elif naca_code is not None:
        code = _validate_naca4_code(naca_code)
        max_camber_pct = float(int(code[0]))
        camber_pos_pct = float(int(code[1]) * 10)
        thickness_pct = float(int(code[2:]))
        if max_camber_pct == 0.0 or camber_pos_pct == 0.0:
            max_camber_pct = 0.0
            camber_pos_pct = 40.0
        zero_lift_angle = float(
            _zero_lift_angle(
                np.asarray(max_camber_pct / 100.0),
                np.asarray(camber_pos_pct / 100.0),
            )
        )
    else:
        _validate_parametric_shape(
            max_camber_pct, camber_pos_pct, thickness_pct, flap_deg
        )
        zero_lift_angle = float(
            _zero_lift_angle(
                np.asarray(max_camber_pct / 100.0),
                np.asarray(camber_pos_pct / 100.0),
            )
        )

    raw_effective_angle = np.radians(angle_deg + 0.5 * flap_deg)
    effective_angle_deg = np.degrees(raw_effective_angle - zero_lift_angle)
    cl = 2.0 * np.pi * (raw_effective_angle - zero_lift_angle)
    cl += 0.15 * np.sqrt(gurney_flap_pct)
    is_stalled = abs(effective_angle_deg) >= 15.0
    if is_stalled:
        stall_factor = max(0.0, 1.0 - (abs(effective_angle_deg) - 15.0) / 10.0)
        cl *= stall_factor
    if inverted:
        cl = -cl

    material_effects = calculate_material_effects(
        material,
        planform["wing_area_m2"],
        airspeed_ms,
        altitude_m,
        thickness_pct,
        cl,
        span_m=planform["span_m"],
        root_chord_m=planform["root_chord_m"],
        tip_chord_m=planform["tip_chord_m"],
        winglet_height_m=winglet_height_m,
        air_density=density,
    )
    cl *= 1.0 - material_effects["aeroelastic_lift_loss_pct"] / 100.0
    thickness_fraction = thickness_pct / 100.0
    cd0 = (
        0.0076
        + 0.02 * thickness_fraction
        + 0.00008 * flap_deg**2
        + 0.0015 * gurney_flap_pct
        + material_effects["roughness_drag_penalty_cd"]
    )
    cd = cd0 + cl**2 / (
        np.pi
        * planform["oswald_efficiency"]
        * planform["effective_aspect_ratio"]
    )
    if is_stalled:
        cd += 0.5

    dynamic_pressure = 0.5 * density * airspeed_ms**2
    lift_newtons = dynamic_pressure * planform["wing_area_m2"] * cl
    drag_newtons = dynamic_pressure * planform["wing_area_m2"] * cd
    reynolds = material_effects["reynolds_number"]
    mach = material_effects["mach_number"]
    return {
        "lift_newtons": float(lift_newtons),
        "drag_newtons": float(drag_newtons),
        "lift_coeff_cl": float(cl),
        "drag_coeff_cd": float(cd),
        "ld_ratio": (
            abs(lift_newtons / drag_newtons)
            if drag_newtons > 0.0
            else float("inf")
        ),
        "is_stalled": bool(is_stalled),
        "streamlines": generate_streamlines(angle_deg),
        "pressure_distribution": generate_pressure_distribution(angle_deg),
        **material_effects,
        "material": material,
        "gurney_flap_pct": float(gurney_flap_pct),
        "inverted": bool(inverted),
        "zero_lift_angle_rad": float(zero_lift_angle),
        "reynolds_number": float(reynolds),
        "mach_number": float(mach),
    }


def compare_against_baseline(current_metrics: dict, baseline_metrics: dict):
    """Return signed percentage changes relative to baseline metrics."""
    comparisons = (
        ("lift_newtons", "lift_change_pct"),
        ("drag_newtons", "drag_change_pct"),
        ("ld_ratio", "efficiency_change_pct"),
    )
    changes = {}
    for metric, result_key in comparisons:
        baseline_value = float(baseline_metrics[metric])
        current_value = float(current_metrics[metric])
        changes[result_key] = (
            (current_value - baseline_value) / baseline_value * 100.0
            if baseline_value != 0.0
            else None
        )
    return changes


def _sweep_designs():
    designs = []
    for camber_digit in range(7):
        combinations = [
            (position_digit, thickness_pct)
            for position_digit in range(2, 7)
            for thickness_pct in range(8, 21)
        ]
        if camber_digit == 0:
            combinations = [(2, thickness_pct) for thickness_pct in range(8, 21)]
            selected = combinations
        else:
            count = 14 if camber_digit in (1, 3, 5) else 15
            indexes = np.linspace(0, len(combinations) - 1, count, dtype=int)
            selected = [combinations[index] for index in indexes]
        for position_digit, thickness_pct in selected:
            designs.append(
                {
                    "naca_code": f"{camber_digit}{position_digit}{thickness_pct:02d}",
                    "max_camber_pct": float(camber_digit),
                    "camber_pos_pct": float(position_digit * 10),
                    "thickness_pct": float(thickness_pct),
                }
            )
    return designs


def analyze_100_airfoils(
    angle_deg: float,
    airspeed_ms: float,
    wing_area_m2: float,
    altitude_m: float,
    *,
    material: str = "Autoclave Carbon Fiber (Mirror Gloss)",
    span_m: float | None = None,
    root_chord_m: float | None = None,
    tip_chord_m: float | None = None,
    winglet_height_m: float = 0.0,
    flap_deg: float = 0.0,
    gurney_flap_pct: float = 0.0,
    inverted: bool = False,
):
    """Vectorize a deterministic 100-profile sweep with current wing conditions."""
    values = (angle_deg, airspeed_ms, wing_area_m2, flap_deg, gurney_flap_pct)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Sweep operating conditions must be finite numbers.")
    if airspeed_ms < 0.0:
        raise ValueError("Airspeed must be non-negative.")
    _validate_flap(flap_deg)
    if not 0.0 <= gurney_flap_pct <= 5.0:
        raise ValueError("Gurney flap height must be between 0 and 5% chord.")
    if material not in MATERIAL_PROPERTIES:
        raise ValueError("Select one of the supported wing/spoiler materials.")
    isa = calculate_isa_properties(altitude_m)
    planform = _resolve_planform(
        wing_area_m2, span_m, root_chord_m, tip_chord_m, winglet_height_m
    )
    designs = _sweep_designs()
    if len(designs) != 100:
        raise RuntimeError("The airfoil sweep must contain exactly 100 designs.")

    camber = np.array([design["max_camber_pct"] for design in designs]) / 100.0
    position = np.array([design["camber_pos_pct"] for design in designs]) / 100.0
    thickness = np.array([design["thickness_pct"] for design in designs]) / 100.0
    zero_lift = _zero_lift_angle(camber, position)
    effective_angle = np.radians(angle_deg + 0.5 * flap_deg) - zero_lift
    cl = 2.0 * np.pi * effective_angle + 0.15 * np.sqrt(gurney_flap_pct)
    stalled = np.abs(np.degrees(effective_angle)) >= 15.0
    stall_factor = np.maximum(0.0, 1.0 - (np.abs(np.degrees(effective_angle)) - 15.0) / 10.0)
    cl = np.where(stalled, cl * stall_factor, cl)
    if inverted:
        cl = -cl

    material_effects = calculate_material_effects(
        material,
        planform["wing_area_m2"],
        airspeed_ms,
        altitude_m,
        float(np.mean([design["thickness_pct"] for design in designs])),
        0.0,
        span_m=planform["span_m"],
        root_chord_m=planform["root_chord_m"],
        tip_chord_m=planform["tip_chord_m"],
        winglet_height_m=winglet_height_m,
        air_density=isa["air_density_kg_m3"],
    )
    properties = MATERIAL_PROPERTIES[material]
    chord = planform["mean_aerodynamic_chord_m"]
    thickness_depth = chord * thickness
    section_inertia = (
        properties["structure_fraction"] * chord * thickness_depth**3 / 12.0
    )
    line_load = (
        0.5
        * isa["air_density_kg_m3"]
        * airspeed_ms**2
        * chord
        * np.abs(cl)
    )
    flexural_rigidity = properties["youngs_modulus_pa"] * section_inertia
    tip_deflection = np.divide(
        5.0 * line_load * (planform["span_m"] / 2.0) ** 4,
        384.0 * flexural_rigidity,
        out=np.zeros_like(line_load),
        where=flexural_rigidity > 0.0,
    )
    flex_loss = np.minimum(
        35.0,
        100.0 * 0.35 * tip_deflection / max(planform["span_m"] / 2.0, 1e-9),
    )
    cl *= 1.0 - flex_loss / 100.0

    roughness_cd = material_effects["roughness_drag_penalty_cd"]
    cd = (
        0.0076
        + 0.02 * thickness
        + 0.00008 * flap_deg**2
        + 0.0015 * gurney_flap_pct
        + roughness_cd
        + cl**2
        / (
            np.pi
            * planform["oswald_efficiency"]
            * planform["effective_aspect_ratio"]
        )
        + np.where(stalled, 0.5, 0.0)
    )
    dynamic_pressure = 0.5 * isa["air_density_kg_m3"] * airspeed_ms**2
    lifts = dynamic_pressure * planform["wing_area_m2"] * cl
    drags = dynamic_pressure * planform["wing_area_m2"] * cd
    ld_ratios = np.divide(
        np.abs(lifts),
        drags,
        out=np.full_like(drags, np.inf),
        where=drags > 0.0,
    )
    ranking = []
    for index, design in enumerate(designs):
        ranking.append(
            {
                **design,
                "lift_newtons": float(lifts[index]),
                "drag_newtons": float(drags[index]),
                "Cl": float(cl[index]),
                "Cd": float(cd[index]),
                "L/D": float(ld_ratios[index]),
                "is_stalled": bool(stalled[index]),
                "structural_weight_kg": float(
                    planform["wing_area_m2"]
                    * chord
                    * thickness[index]
                    * properties["structure_fraction"]
                    * properties["density_kg_m3"]
                ),
                "aeroelastic_lift_loss_pct": float(flex_loss[index]),
            }
        )
    ranking.sort(key=lambda result: result["L/D"], reverse=True)
    return {
        "ranking": ranking,
        "best_airfoil": ranking[0],
        "air_density_kg_m3": isa["air_density_kg_m3"],
        "reynolds_number": material_effects["reynolds_number"],
        "mach_number": material_effects["mach_number"],
    }


def calculate_fuel_and_energy(
    mode: str,
    airspeed_ms: float,
    wing_area_m2: float,
    drag_coeff_cd: float,
    altitude_m: float = 0.0,
    *,
    baseline_drag_coeff_cd: float | None = None,
    baseline_wing_area_m2: float | None = None,
    powertrain: str = "Jet",
):
    """Estimate aero-related energy use with explicit fuel/drivetrain assumptions."""
    if mode not in ("aircraft", "automotive"):
        raise ValueError("Mode must be 'aircraft' or 'automotive'.")
    values = (airspeed_ms, wing_area_m2, drag_coeff_cd)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Speed, area, and drag coefficient must be finite numbers.")
    if airspeed_ms <= 0.0 or wing_area_m2 <= 0.0 or drag_coeff_cd <= 0.0:
        raise ValueError("Speed, area, and drag coefficient must be positive.")
    if baseline_drag_coeff_cd is not None and (
        not np.isfinite(baseline_drag_coeff_cd) or baseline_drag_coeff_cd <= 0.0
    ):
        raise ValueError("Baseline drag coefficient must be positive and finite.")
    baseline_area = (
        wing_area_m2
        if baseline_wing_area_m2 is None
        else baseline_wing_area_m2
    )
    if not np.isfinite(baseline_area) or baseline_area <= 0.0:
        raise ValueError("Baseline wing area must be positive and finite.")
    density = calculate_air_density(altitude_m)

    if mode == "aircraft":
        if powertrain not in ("Jet", "Avgas"):
            raise ValueError("Aircraft powertrain must be 'Jet' or 'Avgas'.")
        fuel_data = {
            "Jet": (9.6, 0.80, 0.38),
            "Avgas": (8.7, 0.72, 0.30),
        }
        energy_kwh_per_liter, fuel_density_kg_l, efficiency = fuel_data[powertrain]

        def aircraft_consumption(coefficient, area):
            drag_force = 0.5 * density * airspeed_ms**2 * area * coefficient
            shaft_power_kw = drag_force * airspeed_ms / 1000.0
            liters_per_hour = shaft_power_kw / (
                energy_kwh_per_liter * efficiency
            )
            kg_per_100km = (
                liters_per_hour
                * fuel_density_kg_l
                / (airspeed_ms * 3.6)
                * 100.0
            )
            return drag_force, shaft_power_kw, liters_per_hour, kg_per_100km

        drag, thrust_power_kw, burn_l_hr, burn_kg_100km = aircraft_consumption(
            drag_coeff_cd, wing_area_m2
        )
        baseline_burn = None
        if baseline_drag_coeff_cd is not None:
            baseline_burn = aircraft_consumption(
                baseline_drag_coeff_cd, baseline_area
            )[2]
        return {
            "air_density_kg_m3": density,
            "drag_newtons": drag,
            "thrust_power_kw": thrust_power_kw,
            "fuel_burn_l_hr": burn_l_hr,
            "fuel_burn_kg_100km": burn_kg_100km,
            "baseline_consumption": baseline_burn,
            "extra_consumption": (
                burn_l_hr - baseline_burn if baseline_burn is not None else None
            ),
            "consumption_unit": "L/hr",
            "powertrain": powertrain,
        }

    if powertrain not in ("Gasoline", "EV"):
        raise ValueError("Automotive powertrain must be 'Gasoline' or 'EV'.")
    highway_speed_ms = 100.0 / 3.6

    def automotive_consumption(coefficient, area):
        drag_force = 0.5 * density * highway_speed_ms**2 * area * coefficient
        wheel_energy_kwh_100km = drag_force * 100000.0 / 3_600_000.0
        if powertrain == "EV":
            return wheel_energy_kwh_100km / 0.90
        return wheel_energy_kwh_100km / (8.9 * 0.25)

    highway_drag = 0.5 * density * highway_speed_ms**2 * wing_area_m2 * drag_coeff_cd
    consumption = automotive_consumption(drag_coeff_cd, wing_area_m2)
    baseline_consumption = (
        automotive_consumption(baseline_drag_coeff_cd, baseline_area)
        if baseline_drag_coeff_cd is not None
        else None
    )
    return {
        "air_density_kg_m3": density,
        "drag_newtons": highway_drag,
        "drag_power_loss_kw": highway_drag * highway_speed_ms / 1000.0,
        "highway_speed_ms": highway_speed_ms,
        "fuel_consumption_l_100km": consumption if powertrain == "Gasoline" else None,
        "energy_consumption_kwh_100km": consumption if powertrain == "EV" else None,
        "baseline_consumption": baseline_consumption,
        "extra_consumption": (
            consumption - baseline_consumption
            if baseline_consumption is not None
            else None
        ),
        "consumption_unit": "kWh/100km" if powertrain == "EV" else "L/100km",
        "powertrain": powertrain,
    }


if __name__ == "__main__":
    for angle in (5, 18):
        results = calculate_aero_forces("2412", angle, 50, 20)
        print(
            f"Angle: {angle} degrees - Lift: {results['lift_newtons']:.2f} N, "
            f"Drag: {results['drag_newtons']:.2f} N"
        )
