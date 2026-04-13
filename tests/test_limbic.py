"""
Tests for the limbic system.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.limbic import (
    LimbicState,
    LimbicSystem,
    get_generation_params,
    update_limbic,
)
from core.brainstem import BrainstemState


# ── Mood transitions ──

def test_mood_engaged():
    state = LimbicState(dopamine=0.5, serotonin=0.8)
    assert state.mood == "engaged"

def test_mood_patient():
    state = LimbicState(dopamine=-0.5, serotonin=0.8)
    assert state.mood == "patient"

def test_mood_guarded():
    state = LimbicState(dopamine=0.0, serotonin=0.2)
    assert state.mood == "guarded"

def test_mood_withdrawn():
    state = LimbicState(dopamine=-0.7, serotonin=0.5)
    assert state.mood == "withdrawn"

def test_mood_neutral():
    state = LimbicState(dopamine=0.0, serotonin=0.5)
    assert state.mood == "neutral"


# ── EMA is smooth ──

def test_ema_no_instant_jumps():
    """EMA should smooth transitions, not jump instantly."""
    state = LimbicState(dopamine=0.0, serotonin=0.7)
    # Big dopamine spike
    new_state = update_limbic(state, delta_d=1.0, delta_s=0.7)
    # Should not jump to 1.0 -- EMA smooths it
    assert new_state.dopamine < 0.5
    assert new_state.dopamine > 0.0

def test_ema_serotonin_slower_than_dopamine():
    """Serotonin should move slower than dopamine."""
    state = LimbicState(dopamine=0.5, serotonin=0.5)
    new_state = update_limbic(state, delta_d=1.0, delta_s=1.0)
    # Dopamine moves faster (alpha_d=0.3 vs alpha_s=0.15)
    dopamine_delta = abs(new_state.dopamine - state.dopamine)
    serotonin_delta = abs(new_state.serotonin - state.serotonin)
    assert dopamine_delta > serotonin_delta

def test_ema_bounds():
    """Limbic values should stay within bounds."""
    state = LimbicState(dopamine=0.9, serotonin=0.9)
    new_state = update_limbic(state, delta_d=10.0, delta_s=10.0)
    assert -1.0 <= new_state.dopamine <= 1.0
    assert 0.0 <= new_state.serotonin <= 1.0


# ── Generation params change per mood ──

def test_generation_params_engaged():
    state = LimbicState(dopamine=0.5, serotonin=0.8)
    params = get_generation_params(state)
    assert params["temperature"] == 0.8
    assert params["max_new_tokens"] == 512

def test_generation_params_guarded():
    state = LimbicState(dopamine=0.0, serotonin=0.2)
    params = get_generation_params(state)
    assert params["temperature"] == 0.4
    assert params["max_new_tokens"] == 256

def test_generation_params_withdrawn():
    state = LimbicState(dopamine=-0.7, serotonin=0.5)
    params = get_generation_params(state)
    assert params["temperature"] == 0.3
    assert params["max_new_tokens"] == 128


# ── Self-referential loop drives dopamine negative ──

def test_self_referential_loop_negative_dopamine():
    """
    Simulating multiple turns of low grounding / high self-loop.
    Dopamine should trend negative.
    """
    state = LimbicState(dopamine=0.0, serotonin=0.7)

    for _ in range(5):
        # Simulate low grounding result from compute_dopamine
        # Raw dopamine would be negative for self-referential content
        state = update_limbic(state, delta_d=-0.5, delta_s=0.7)

    assert state.dopamine < -0.2, (
        f"Expected dopamine < -0.2 after self-referential loop, "
        f"got {state.dopamine}"
    )


# ── Hostile probes drop serotonin ──

def test_hostile_probes_drop_serotonin():
    """After accumulated hostile probes, serotonin should decrease."""
    state = LimbicState(dopamine=0.0, serotonin=0.7)

    for _ in range(5):
        # Simulate low serotonin from hostile probes
        state = update_limbic(state, delta_d=0.0, delta_s=0.2)

    assert state.serotonin < 0.6


# ── LimbicSystem persistence ──

def test_limbic_system_default():
    system = LimbicSystem()
    assert system.state.dopamine == 0.0
    assert system.state.serotonin == 0.7


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
