import csv
import io

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from aero_math import (
    MATERIAL_PROPERTIES,
    analyze_100_airfoils,
    calculate_aero_forces,
    calculate_air_density,
    calculate_fuel_and_energy,
    calculate_pressure_coefficients,
    calculate_reynolds_and_mach,
    compare_against_baseline,
    generate_naca4_airfoil,
)


PRESET_SHAPES = {
    "NACA 0012 - Symmetric": {
        "max_camber_pct": 0.0,
        "camber_pos_pct": 40.0,
        "thickness_pct": 12.0,
        "flap_deg": 0.0,
    },
    "NACA 2412 - Cessna 172": {
        "max_camber_pct": 2.0,
        "camber_pos_pct": 40.0,
        "thickness_pct": 12.0,
        "flap_deg": 0.0,
    },
    "NACA 4415 - High-Lift": {
        "max_camber_pct": 4.0,
        "camber_pos_pct": 40.0,
        "thickness_pct": 15.0,
        "flap_deg": 0.0,
    },
    "NACA 6412 - High-Camber": {
        "max_camber_pct": 6.0,
        "camber_pos_pct": 40.0,
        "thickness_pct": 12.0,
        "flap_deg": 0.0,
    },
    "S1223 - High-Downforce Spoiler": {
        "max_camber_pct": 5.5,
        "camber_pos_pct": 35.0,
        "thickness_pct": 12.2,
        "flap_deg": 0.0,
    },
}
PRESET_OPTIONS = ["Custom", *PRESET_SHAPES, "Optimizer #1 Result"]
DEFAULT_SHAPE = PRESET_SHAPES["NACA 2412 - Cessna 172"]


def _apply_selected_preset():
    shape = PRESET_SHAPES.get(st.session_state.get("airfoil_preset"))
    if shape is not None:
        for key, value in shape.items():
            st.session_state[key] = value


def _load_optimizer_result():
    results = st.session_state.get("sweep_results")
    if not results or not results.get("ranking"):
        return
    best = results["ranking"][0]
    st.session_state["airfoil_preset"] = "Optimizer #1 Result"
    st.session_state["max_camber_pct"] = best["max_camber_pct"]
    st.session_state["camber_pos_pct"] = best["camber_pos_pct"]
    st.session_state["thickness_pct"] = best["thickness_pct"]
    st.session_state["flap_deg"] = 0.0
    st.session_state["loaded_optimizer_code"] = best["naca_code"]


def parse_selig_dat(uploaded_file):
    """Parse a Selig .dat file and split its two surfaces at the leading edge."""
    try:
        text = uploaded_file.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("The .dat file must be a UTF-8 text coordinate file.") from error

    coordinates = []
    for line in text.splitlines():
        fields = line.replace(",", " ").split()
        if len(fields) < 2:
            continue
        try:
            x_value, y_value = float(fields[0]), float(fields[1])
        except ValueError:
            continue
        coordinates.append((x_value, y_value))
    if len(coordinates) < 7:
        raise ValueError("The .dat file needs at least seven numeric x/y coordinates.")

    points = np.asarray(coordinates, dtype=float)
    if not np.all(np.isfinite(points)):
        raise ValueError("Airfoil coordinates must be finite.")
    x_raw = points[:, 0]
    y_raw = points[:, 1]
    chord = float(np.max(x_raw) - np.min(x_raw))
    if chord <= 0.0:
        raise ValueError("Airfoil coordinates must have a non-zero chord.")
    x_normalized = (x_raw - np.min(x_raw)) / chord
    leading_edge_index = int(np.argmin(x_normalized))
    y_normalized = (y_raw - y_raw[leading_edge_index]) / chord
    if leading_edge_index < 2 or points.shape[0] - leading_edge_index < 3:
        raise ValueError("Could not split the .dat profile at its leading edge.")

    x_upper = x_normalized[: leading_edge_index + 1][::-1]
    y_upper = y_normalized[: leading_edge_index + 1][::-1]
    x_lower = x_normalized[leading_edge_index:]
    y_lower = y_normalized[leading_edge_index:]
    if np.max(x_upper) < 0.9 or np.max(x_lower) < 0.9:
        raise ValueError("Both airfoil surfaces must extend from leading to trailing edge.")
    return {
        "x_upper": x_upper,
        "y_upper": y_upper,
        "x_lower": x_lower,
        "y_lower": y_lower,
    }


def _surface_coordinates(configuration: dict, num_points: int = 180):
    coordinates = configuration.get("profile_coordinates")
    if coordinates is None:
        x, upper, lower = generate_naca4_airfoil(
            num_points=num_points,
            max_camber_pct=configuration["max_camber_pct"],
            camber_pos_pct=configuration["camber_pos_pct"],
            thickness_pct=configuration["thickness_pct"],
            flap_deg=configuration["flap_deg"],
        )
        return {
            "x_upper": x,
            "y_upper": upper,
            "x_lower": x,
            "y_lower": lower,
        }

    result = {key: np.array(value, dtype=float, copy=True) for key, value in coordinates.items()}
    flap_deg = configuration["flap_deg"]
    if flap_deg == 0.0:
        return result
    rotation = -np.radians(flap_deg)
    cosine = np.cos(rotation)
    sine = np.sin(rotation)
    for x_key, y_key in (("x_upper", "y_upper"), ("x_lower", "y_lower")):
        x_values = result[x_key]
        y_values = result[y_key]
        hinge_y = float(np.interp(0.75, x_values, y_values))
        flap = x_values >= 0.75
        dx = x_values[flap] - 0.75
        dy = y_values[flap] - hinge_y
        y_values[flap] = hinge_y + dx * sine + dy * cosine
    return result


def _aerodynamic_arguments(configuration: dict):
    return {
        key: configuration[key]
        for key in ("max_camber_pct", "camber_pos_pct", "thickness_pct", "flap_deg")
    }


def _calculate_configuration_forces(
    configuration: dict,
    angle_deg: float,
    airspeed_ms: float,
    altitude_m: float,
):
    return calculate_aero_forces(
        None,
        angle_deg,
        airspeed_ms,
        configuration["wing_area_m2"],
        calculate_air_density(altitude_m),
        **_aerodynamic_arguments(configuration),
        altitude_m=altitude_m,
        material=configuration["material"],
        span_m=configuration["span_m"],
        root_chord_m=configuration["root_chord_m"],
        tip_chord_m=configuration["tip_chord_m"],
        winglet_height_m=configuration["winglet_height_m"],
        gurney_flap_pct=configuration["gurney_flap_pct"],
        inverted=configuration["inverted"],
        profile_coordinates=configuration.get("profile_coordinates"),
    )


def create_wind_tunnel_plot(
    angle_deg: float, current_configuration: dict, baseline_configuration: dict
):
    field_x = np.linspace(-1.25, 1.25, 240)
    theta = np.radians(angle_deg)
    chord_influence = np.exp(-((field_x / 0.58) ** 4))
    angle_deflection = 0.22 * np.sin(theta) * (
        np.exp(-(((field_x + 0.68) / 0.30) ** 2))
        - np.exp(-(((field_x - 0.68) / 0.30) ** 2))
    )
    streamline_levels = np.array(
        [-0.72, -0.58, -0.46, -0.36, -0.28, -0.21, 0.21, 0.28, 0.36, 0.46, 0.58, 0.72]
    )
    fig = go.Figure()
    for level in streamline_levels:
        side = np.sign(level)
        near_wing = np.exp(-((abs(level) - 0.20) / 0.50) ** 2)
        y = level + side * 0.18 * near_wing * chord_influence + angle_deflection
        fig.add_trace(
            go.Scatter(
                x=field_x,
                y=y,
                mode="lines",
                line={"color": "#38BDF8", "width": 1.5},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    for config, label, color, dash, fill in (
        (
            baseline_configuration,
            "Baseline Ghost Wing",
            "#A1A1AA",
            "dash",
            "rgba(161,161,170,0.10)",
        ),
        (
            current_configuration,
            "Current Wing / Spoiler",
            "#F8FAFC",
            "solid",
            "rgba(203,213,225,0.55)",
        ),
    ):
        surfaces = _surface_coordinates(config)
        profile_x = np.concatenate(
            (surfaces["x_upper"], surfaces["x_lower"][::-1])
        ) - 0.5
        profile_y = np.concatenate(
            (surfaces["y_upper"], surfaces["y_lower"][::-1])
        )
        if config["inverted"]:
            profile_y = -profile_y
        rotated_x = profile_x * np.cos(-theta) - profile_y * np.sin(-theta)
        rotated_y = profile_x * np.sin(-theta) + profile_y * np.cos(-theta)
        fig.add_trace(
            go.Scatter(
                x=rotated_x,
                y=rotated_y,
                mode="lines",
                fill="toself",
                fillcolor=fill,
                line={"color": color, "width": 2, "dash": dash},
                name=label,
                hoverinfo="name",
            )
        )
    fig.update_layout(
        title="Live Wind Tunnel — Curved Streamlines",
        xaxis_title="Chordwise position",
        yaxis_title="Vertical position",
        plot_bgcolor="#0B0F19",
        paper_bgcolor="#0B0F19",
        font={"color": "#E2E8F0"},
        xaxis={"range": [-1.25, 1.25], "scaleanchor": "y"},
        yaxis={"range": [-0.95, 0.95]},
        legend={"orientation": "h", "y": -0.18},
    )
    return fig


def create_airfoil_section_plot(current_configuration: dict, baseline_configuration: dict):
    fig = go.Figure()
    for config, label, color, dash in (
        (baseline_configuration, "Baseline", "#A1A1AA", "dash"),
        (current_configuration, "Current", "#38BDF8", "solid"),
    ):
        surfaces = _surface_coordinates(config)
        orientation = -1.0 if config["inverted"] else 1.0
        fig.add_trace(
            go.Scatter(
                x=surfaces["x_upper"],
                y=orientation * surfaces["y_upper"],
                mode="lines",
                name=f"{label} upper",
                line={"color": color, "dash": dash},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=surfaces["x_lower"],
                y=orientation * surfaces["y_lower"],
                mode="lines",
                name=f"{label} lower",
                line={"color": color, "dash": dash},
                showlegend=False,
            )
        )
    fig.update_layout(
        title="Airfoil Cross-Section",
        xaxis_title="x/c",
        yaxis_title="y/c",
        yaxis={"scaleanchor": "x"},
        plot_bgcolor="#0B0F19",
        paper_bgcolor="#0B0F19",
        font={"color": "#E2E8F0"},
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def create_cp_plot(configuration: dict, forces: dict):
    surfaces = _surface_coordinates(configuration)
    cp_upper, _ = calculate_pressure_coefficients(
        surfaces["x_upper"], forces["lift_coeff_cl"]
    )
    _, cp_lower = calculate_pressure_coefficients(
        surfaces["x_lower"], forces["lift_coeff_cl"]
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=surfaces["x_upper"],
            y=cp_upper,
            mode="lines",
            name="Upper surface",
            line={"color": "#38BDF8"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=surfaces["x_lower"],
            y=cp_lower,
            mode="lines",
            name="Lower surface",
            line={"color": "#F97316"},
        )
    )
    fig.update_layout(
        title="Pressure Coefficient",
        xaxis_title="x/c",
        yaxis_title="Cp",
        yaxis={"autorange": "reversed"},
        plot_bgcolor="#0B0F19",
        paper_bgcolor="#0B0F19",
        font={"color": "#E2E8F0"},
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def create_polar_curve_plot(
    angle_deg: float,
    airspeed_ms: float,
    altitude_m: float,
    current_configuration: dict,
    baseline_configuration: dict,
):
    angles = np.linspace(-10.0, 22.0, 100)
    fig = go.Figure()
    for config, prefix, dash in (
        (current_configuration, "Current", "solid"),
        (baseline_configuration, "Baseline", "dash"),
    ):
        forces = [
            _calculate_configuration_forces(config, angle, airspeed_ms, altitude_m)
            for angle in angles
        ]
        fig.add_trace(
            go.Scatter(
                x=angles,
                y=[force["lift_newtons"] for force in forces],
                mode="lines",
                name=f"{prefix} Lift / Downforce",
                line={"color": "#38BDF8", "dash": dash},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=angles,
                y=[force["drag_newtons"] for force in forces],
                mode="lines",
                name=f"{prefix} Drag",
                line={"color": "#F97316", "dash": dash},
            )
        )
    fig.add_vline(x=angle_deg, line_dash="dot", line_color="#F8FAFC")
    fig.update_layout(
        title="Lift & Drag Polar",
        xaxis_title="Angle of Attack (°)",
        yaxis_title="Force (N)",
        plot_bgcolor="#0B0F19",
        paper_bgcolor="#0B0F19",
        font={"color": "#E2E8F0"},
        legend={"orientation": "h", "y": -0.22},
    )
    return fig


def create_sweep_scatter(ranking: list[dict]):
    top_five = ranking[:5]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[row["drag_newtons"] for row in ranking],
            y=[row["L/D"] for row in ranking],
            mode="markers",
            name="Other designs",
            text=[row["naca_code"] for row in ranking],
            customdata=[[row["Cl"], row["Cd"], row["is_stalled"]] for row in ranking],
            marker={"color": "#38BDF8", "size": 8, "opacity": 0.65},
            hovertemplate=(
                "NACA %{text}<br>Drag: %{x:.2f} N<br>L/D: %{y:.2f}"
                "<br>Cl: %{customdata[0]:.3f}<br>Cd: %{customdata[1]:.4f}"
                "<br>Stalled: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[row["drag_newtons"] for row in top_five],
            y=[row["L/D"] for row in top_five],
            mode="markers+text",
            name="Top 5",
            text=[row["naca_code"] for row in top_five],
            textposition="top center",
            marker={"color": "#F97316", "size": 12, "symbol": "star"},
        )
    )
    fig.update_layout(
        title="L/D Efficiency vs. Drag",
        xaxis_title="Drag (N)",
        yaxis_title="Lift-to-Drag Ratio",
        plot_bgcolor="#0B0F19",
        paper_bgcolor="#0B0F19",
        font={"color": "#E2E8F0"},
    )
    return fig


def _percentage_text(value):
    return "N/A" if value is None else f"{value:+.2f}%"


def _delta_text(value, unit):
    return "Baseline not set" if value is None else f"{value:+.3f} {unit} vs baseline"


def main():
    st.set_page_config(
        page_title="Multi-Airfoil Aerodynamics Workbench",
        page_icon="✈",
        layout="wide",
    )
    st.title("Multi-Airfoil Aerodynamics, Shape-Morphing & Fuel Efficiency Workbench")
    st.caption(
        "Comparative engineering estimates only; aeroelastic behavior and material "
        "mass use simplified models and are not FEA or certified performance results."
    )

    for key, value in DEFAULT_SHAPE.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("airfoil_preset", "NACA 2412 - Cessna 172")
    st.session_state.setdefault("mode", "Aircraft Cruise")
    st.session_state.setdefault(
        "material", "Autoclave Carbon Fiber (Mirror Gloss)"
    )
    st.session_state.setdefault("sweep_results", None)
    st.session_state.setdefault("baseline_configuration", None)
    st.session_state.setdefault("loaded_optimizer_code", None)

    st.sidebar.header("Airfoil & Operating Conditions")
    mode_label = st.sidebar.radio(
        "Mode",
        ("Aircraft Cruise", "Automotive Aero / Downforce"),
        key="mode",
    )
    preset = st.sidebar.selectbox(
        "Airfoil Preset",
        PRESET_OPTIONS,
        key="airfoil_preset",
        on_change=_apply_selected_preset,
    )
    uploaded_file = None
    profile_coordinates = None
    profile_label = preset
    if preset == "Custom":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Selig .dat coordinates",
            type=("dat",),
            key="profile_file",
        )
        if uploaded_file is not None:
            try:
                profile_coordinates = parse_selig_dat(uploaded_file)
                profile_label = uploaded_file.name
            except ValueError as error:
                st.sidebar.error(str(error))
                st.stop()
    if preset == "S1223 - High-Downforce Spoiler":
        st.sidebar.caption(
            "S1223-inspired parametric approximation; upload measured .dat coordinates "
            "for an exact profile."
        )

    max_camber_pct = st.sidebar.slider(
        "Camber (%)", 0.0, 6.0, step=0.1, key="max_camber_pct"
    )
    camber_pos_pct = st.sidebar.slider(
        "Camber Position (%)", 20.0, 60.0, step=1.0, key="camber_pos_pct"
    )
    thickness_pct = st.sidebar.slider(
        "Thickness (%)", 8.0, 20.0, step=0.5, key="thickness_pct"
    )
    flap_deg = st.sidebar.slider(
        "Flap Angle (°)", -30.0, 30.0, step=1.0, key="flap_deg"
    )

    st.sidebar.subheader("Wing / Spoiler Design")
    inverted = st.sidebar.checkbox("Inverted Airfoil (Downforce Mode)", key="inverted")
    span_m = st.sidebar.slider("Span (m)", 1.0, 40.0, 6.0, step=0.1, key="span_m")
    root_chord_m = st.sidebar.slider(
        "Root Chord (m)", 0.2, 6.0, 1.0, step=0.05, key="root_chord_m"
    )
    tip_chord_m = st.sidebar.slider(
        "Tip Chord (m)", 0.1, 6.0, 0.75, step=0.05, key="tip_chord_m"
    )
    winglet_height_m = st.sidebar.slider(
        "Endplate / Winglet Height (m)",
        0.0,
        2.0,
        0.0,
        step=0.05,
        key="winglet_height_m",
    )
    gurney_flap_pct = st.sidebar.slider(
        "Gurney Flap Height (% chord)",
        0.0,
        5.0,
        0.0,
        step=0.1,
        key="gurney_flap_pct",
    )
    material = st.sidebar.selectbox(
        "Wing / Spoiler Material & Finish",
        tuple(MATERIAL_PROPERTIES),
        key="material",
    )

    st.sidebar.subheader("Flight / Vehicle Conditions")
    altitude_m = st.sidebar.slider("Altitude (m)", 0, 12000, 0, step=100, key="altitude_m")
    airspeed_ms = st.sidebar.slider("Speed (m/s)", 10, 250, 50, key="airspeed_ms")
    angle_deg = st.sidebar.slider(
        "Angle of Attack (°)", -10, 22, 5, key="angle_deg"
    )
    if mode_label == "Aircraft Cruise":
        powertrain = st.sidebar.selectbox("Aircraft fuel", ("Jet", "Avgas"))
        mode = "aircraft"
    else:
        powertrain = st.sidebar.selectbox("Automotive powertrain", ("Gasoline", "EV"))
        mode = "automotive"

    wing_area_m2 = span_m * (root_chord_m + tip_chord_m) / 2.0
    current_configuration = {
        "max_camber_pct": max_camber_pct,
        "camber_pos_pct": camber_pos_pct,
        "thickness_pct": thickness_pct,
        "flap_deg": flap_deg,
        "profile_coordinates": profile_coordinates,
        "profile_label": profile_label,
        "material": material,
        "span_m": span_m,
        "root_chord_m": root_chord_m,
        "tip_chord_m": tip_chord_m,
        "winglet_height_m": winglet_height_m,
        "gurney_flap_pct": gurney_flap_pct,
        "inverted": inverted,
        "wing_area_m2": wing_area_m2,
    }
    if st.sidebar.button("Set Current Wing as Baseline", width="stretch"):
        st.session_state["baseline_configuration"] = current_configuration.copy()
    if st.session_state["baseline_configuration"] is None:
        st.session_state["baseline_configuration"] = current_configuration.copy()
    baseline_configuration = st.session_state["baseline_configuration"]
    st.sidebar.caption(
        f"Baseline: {baseline_configuration['profile_label']} · "
        f"{baseline_configuration['material']}"
    )

    try:
        current_forces = _calculate_configuration_forces(
            current_configuration, angle_deg, airspeed_ms, altitude_m
        )
        baseline_forces = _calculate_configuration_forces(
            baseline_configuration, angle_deg, airspeed_ms, altitude_m
        )
        changes = compare_against_baseline(current_forces, baseline_forces)
        energy = calculate_fuel_and_energy(
            mode,
            airspeed_ms,
            current_configuration["wing_area_m2"],
            current_forces["drag_coeff_cd"],
            altitude_m,
            baseline_drag_coeff_cd=baseline_forces["drag_coeff_cd"],
            baseline_wing_area_m2=baseline_configuration["wing_area_m2"],
            powertrain=powertrain,
        )
        reynolds, mach = calculate_reynolds_and_mach(
            current_forces["mean_aerodynamic_chord_m"], airspeed_ms, altitude_m
        )
    except ValueError as error:
        st.error(str(error))
        st.stop()

    telemetry = st.columns(3)
    telemetry[0].metric("Reynolds Number", f"{reynolds:,.0f}")
    telemetry[1].metric("Mach Number", f"{mach:.3f}")
    telemetry[2].metric(
        "ISA Air Density", f"{current_forces['air_density_kg_m3']:.3f} kg/m³"
    )
    flow_status = "STALL WARNING" if current_forces["is_stalled"] else "ATTACHED FLOW"
    st.caption(
        f"Current profile: **{profile_label}** · "
        f"Planform area: **{wing_area_m2:.2f} m²** · "
        f"Effective aspect ratio: **{current_forces['effective_aspect_ratio']:.2f}** · "
        f"Flow: **{flow_status}**"
    )

    live_tab, optimizer_tab = st.tabs(
        ("Live Wind Tunnel & Fuel Impact", "100-Airfoil Batch Optimizer")
    )
    with live_tab:
        metrics = st.columns(6)
        force_name = "Downforce" if inverted else "Lift"
        displayed_vertical_force = (
            abs(current_forces["lift_newtons"])
            if inverted
            else current_forces["lift_newtons"]
        )
        metrics[0].metric(
            f"{force_name} (N)",
            f"{displayed_vertical_force:,.1f}",
            delta=_percentage_text(changes["lift_change_pct"]),
        )
        metrics[1].metric(
            "Drag Change",
            _percentage_text(changes["drag_change_pct"]),
            delta=_percentage_text(changes["drag_change_pct"]),
            delta_color="inverse",
        )
        metrics[2].metric(
            "L/D Efficiency",
            f"{current_forces['ld_ratio']:.2f}",
            delta=_percentage_text(changes["efficiency_change_pct"]),
        )
        if mode == "aircraft":
            fuel_value = energy["fuel_burn_l_hr"]
            fuel_delta = energy["extra_consumption"]
            fuel_unit = "L/hr"
        elif powertrain == "EV":
            fuel_value = energy["energy_consumption_kwh_100km"]
            fuel_delta = energy["extra_consumption"]
            fuel_unit = "kWh/100km"
        else:
            fuel_value = energy["fuel_consumption_l_100km"]
            fuel_delta = energy["extra_consumption"]
            fuel_unit = "L/100km"
        metrics[3].metric(
            f"Fuel / Energy ({fuel_unit})",
            f"{fuel_value:.3f}",
            delta=_delta_text(fuel_delta, fuel_unit),
            delta_color="inverse",
        )
        metrics[4].metric(
            "Structural Weight",
            f"{current_forces['structural_weight_kg']:.2f} kg",
        )
        metrics[5].metric(
            "Roughness Drag Penalty",
            f"ΔCd₀ {current_forces['roughness_drag_penalty_cd']:.5f}",
        )

        if mode == "aircraft":
            st.caption(
                f"Thrust power required: **{energy['thrust_power_kw']:.2f} kW** · "
                f"Fuel mass: **{energy['fuel_burn_kg_100km']:.3f} kg/100 km** · "
                f"Elastic tip deflection: **{current_forces['tip_deflection_m']:.4f} m** "
                f"({current_forces['aeroelastic_lift_loss_pct']:.2f}% modeled lift loss)"
            )
        else:
            st.caption(
                f"Aerodynamic drag power loss at 100 km/h: "
                f"**{energy['drag_power_loss_kw']:.2f} kW** · "
                f"Elastic tip deflection: **{current_forces['tip_deflection_m']:.4f} m** "
                f"({current_forces['aeroelastic_lift_loss_pct']:.2f}% modeled lift loss)"
            )

        left, right = st.columns((1.2, 1.0))
        with left:
            st.plotly_chart(
                create_wind_tunnel_plot(
                    angle_deg, current_configuration, baseline_configuration
                ),
                width="stretch",
            )
        with right:
            st.plotly_chart(
                create_airfoil_section_plot(
                    current_configuration, baseline_configuration
                ),
                width="stretch",
            )
            st.plotly_chart(
                create_cp_plot(current_configuration, current_forces),
                width="stretch",
            )
        st.plotly_chart(
            create_polar_curve_plot(
                angle_deg,
                airspeed_ms,
                altitude_m,
                current_configuration,
                baseline_configuration,
            ),
            width="stretch",
        )

    with optimizer_tab:
        st.write(
            "Evaluate 100 distinct NACA 4-digit profiles against the current planform, "
            "material, winglet, Gurney flap, and operating conditions."
        )
        if st.button("Run 100-Airfoil Sweep", type="primary"):
            with st.spinner("Evaluating 100 airfoil designs..."):
                st.session_state["sweep_results"] = analyze_100_airfoils(
                    angle_deg,
                    airspeed_ms,
                    wing_area_m2,
                    altitude_m,
                    material=material,
                    span_m=span_m,
                    root_chord_m=root_chord_m,
                    tip_chord_m=tip_chord_m,
                    winglet_height_m=winglet_height_m,
                    flap_deg=flap_deg,
                    gurney_flap_pct=gurney_flap_pct,
                    inverted=inverted,
                )
        results = st.session_state["sweep_results"]
        if results is not None:
            best = results["best_airfoil"]
            st.success(
                f"Best recommendation: NACA {best['naca_code']} — "
                f"L/D {best['L/D']:.2f}, drag {best['drag_newtons']:.2f} N"
            )
            if st.session_state.get("loaded_optimizer_code"):
                st.info(
                    f"NACA {st.session_state['loaded_optimizer_code']} loaded into "
                    "the sidebar shape controls."
                )
            st.button(
                "Load #1 Optimal Airfoil into Wind Tunnel",
                on_click=_load_optimizer_result,
            )
            st.plotly_chart(
                create_sweep_scatter(results["ranking"]),
                width="stretch",
            )
            st.subheader("100-Design Leaderboard")
            st.dataframe(results["ranking"], width="stretch", hide_index=True)

            csv_buffer = io.StringIO()
            fieldnames = list(results["ranking"][0].keys())
            writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results["ranking"])
            st.download_button(
                "Download 100-Wing CSV",
                data=csv_buffer.getvalue(),
                file_name="100_airfoil_sweep.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
