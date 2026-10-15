"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button. Es gibt keinen ausblendbaren Regler: alle Regler sind in jedem Szenario sichtbar."""

import random
from dataclasses import dataclass

import streamlit as st

import ltr_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: object
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "p_slider": SettingSpec("p", int, C.DEFAULT_P, C.P_MIN, C.P_MAX),
    "sigma_slider": SettingSpec("su", float, C.DEFAULT_SIGMA, C.SIGMA_MIN, C.SIGMA_MAX),
    "truth_select": SettingSpec("t", str, C.DEFAULT_TRUTH, choices=C.TRUTHS),
    "q_slider": SettingSpec("q", int, C.DEFAULT_Q, C.Q_MIN, C.Q_MAX),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "loss_select": SettingSpec("lo", str, C.DEFAULT_LOSS, choices=C.LOSSES),
    "model_select": SettingSpec("m", str, C.DEFAULT_MODEL, choices=C.MODELS),
    "qtest_select": SettingSpec("qt", float, C.DEFAULT_Q_TEST, choices=C.Q_TEST_CHOICES),
    "l2_slider": SettingSpec("l2", float, C.DEFAULT_L2, C.L2_MIN, C.L2_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"p": "p_slider", "sigma_u": "sigma_slider", "truth": "truth_select", "q_groups": "q_slider", "k_train": "k_slider", "loss": "loss_select", "model": "model_select",
               "q_test": "qtest_select", "l2": "l2_slider", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
