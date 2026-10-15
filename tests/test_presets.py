"""Presets: Vollständigkeit, gültige Werte, erstes Preset = Standardeinstellungen; Adressparameter, Szenario und PDF."""

import ltr_constants as C
import ltr_evaluation as ev
import ltr_pdf_export as X
import ltr_presets as P
from ltr_scenario import Settings


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi, (name, key)
            if spec.choices is not None:
                assert p[key] in spec.choices, (name, key)


def test_first_preset_equals_the_default_settings():
    assert Settings(**C.PRESETS["Neue Spieler ohne Duelle"]) == Settings()


def test_preset_keys_are_exactly_the_settings_fields():
    assert set(P.PRESET_KEYS) == set(Settings.__dataclass_fields__)


def test_setting_specs_cover_every_preset_key_and_have_unique_parameters():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_presets_show_their_mechanism():
    assert C.PRESETS["Wenn nichts unerklärt bleibt"]["sigma_u"] == 0.0
    assert C.PRESETS["Merkmale gegen Duelle"]["q_test"] == 0.5
    nl = C.PRESETS["Das Netz lernt Nichtlineares"]
    assert nl["truth"] == C.TRUTHS[1] and nl["model"] == C.MODELS[1] and nl["q_groups"] >= 80


def test_constants_are_sound():
    assert len(C.SWEEP_SEEDS) == len(set(C.SWEEP_SEEDS)) == 60
    assert set(C.COLORS) >= set(C.MODELS) and set(C.LOSS_COLORS) == set(C.LOSSES)
    assert set(C.EXP_QS_SHORT) <= set(C.EXP_QS) and list(C.LAM_GRID) == sorted(C.LAM_GRID) and C.Q_TEST_CHOICES[0] == 0.0
    assert C.M_GROUP == 12 and C.EXP_WARM_FRACTIONS[-1] == 1.0


def test_pdf_is_built_for_every_preset():
    for name, p in C.PRESETS.items():
        s = Settings(**{**p, "q_groups": min(p["q_groups"], 12)})
        data = X.build_pdf(ev.analyse(s))
        assert data[:5] == b"%PDF-" and len(data) > 1500


def test_latin_filter_replaces_greek_and_math_symbols():
    out = X._latin("λ ≈ 0,5; σ → τ µ θ − **fett** „Zitat“ f̂")
    assert out == 'lambda ca. 0,5; sigma -> Tau mu theta - fett "Zitat" f'
    assert out.encode("latin-1")
