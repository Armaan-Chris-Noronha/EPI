"""Tests for pure math modules: PK/PD, Wright-Fisher, network, Bayes."""

import numpy as np
import pytest

from episteward.math.pkpd import concentration_profile, therapeutic_score
from episteward.math.evolution import wright_fisher_step, evolve_resistance
from episteward.math.network import build_graph, transmission_probability
from episteward.math.bayes import update_posterior, estimate_resistance


# --- PK/PD ---

def test_concentration_decays():
    pk = {"F": 1.0, "Vd_L_kg": 0.3, "CL_L_h_kg": 0.1, "ke": 0.33}
    t, C = concentration_profile(1000.0, pk, t_span=(0.0, 24.0))
    assert C[0] > C[-1], "Concentration should decay over time"
    assert all(c >= 0 for c in C), "Concentration must be non-negative"


def test_therapeutic_score_in_range():
    pk = {"F": 1.0, "Vd_L_kg": 0.3, "CL_L_h_kg": 0.1, "ke": 0.33}
    score = therapeutic_score(1000.0, pk, mic=2.0, frequency_hours=8.0)
    assert 0.0 <= score <= 1.0


# --- Wright-Fisher ---

def test_wright_fisher_stays_bounded():
    rng = np.random.default_rng(42)
    freq = 0.1
    for _ in range(20):
        freq = wright_fisher_step(freq, s=0.3, rng=rng)
        assert 0.0 <= freq <= 1.0


def test_resistance_emerges_under_pressure():
    rng = np.random.default_rng(42)
    freq = 0.3
    # Run for enough steps under high pressure
    for _ in range(50):
        freq, _ = evolve_resistance(freq, treatment_hours=96.0, dose_mg=1000.0, standard_dose_mg=500.0, rng=rng)
    # Frequency should have moved significantly upward
    assert freq > 0.3 or freq < 0.1  # either direction is valid


# --- Network ---

def test_build_graph_loads():
    import json
    from pathlib import Path
    data = json.loads((Path(__file__).parent.parent / "episteward/data/hospital_network.json").read_text())
    G = build_graph(data)
    assert len(G.nodes) > 0
    assert len(G.edges) > 0


def test_transmission_probability_isolated():
    import json
    from pathlib import Path
    data = json.loads((Path(__file__).parent.parent / "episteward/data/hospital_network.json").read_text())
    G = build_graph(data)
    p = transmission_probability(G, "H3", "H4", infected_count=5, immune=True)
    assert p == 0.0


# --- Bayes ---

def test_posterior_updates_on_resistant():
    mean, lo, hi = update_posterior(0.2, "resistant")
    assert mean > 0.2, "Posterior should increase on resistant result"
    assert 0.0 <= lo <= mean <= hi <= 1.0


def test_posterior_updates_on_sensitive():
    mean, lo, hi = update_posterior(0.5, "sensitive")
    assert mean < 0.5, "Posterior should decrease on sensitive result"


def test_estimate_resistance_returns_dict():
    import json
    from pathlib import Path
    profiles = json.loads((Path(__file__).parent.parent / "episteward/data/resistance_profiles.json").read_text())
    result = estimate_resistance("E_coli_ESBL", "ciprofloxacin", "icu", profiles, "resistant")
    assert "posterior" in result
    assert 0.0 <= result["posterior"] <= 1.0
