"""
DAEDALUS v0.5 — Core Data Types

The atomic structures of selfhood. Every field here carries
semantic weight — these are not database columns, they are
the dimensions along which experience becomes memory becomes identity.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

import numpy as np


# ─────────────────────────────────────────────
# Provider Types
# ─────────────────────────────────────────────

class ProviderStatus(Enum):
    """Health status of a Soul Bridge provider."""
    AVAILABLE = "available"
    DEGRADED = "degraded"          # responding but slow or truncated
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class SoulResponse:
    """
    Response from a Soul Bridge provider.
    Every reflection carries its provenance — which mind spoke,
    how long it took, whether continuity was maintained.
    """
    text: str
    provider_name: str
    model_id: str
    latency_ms: float
    token_count: int
    continuity_score: Optional[float] = None  # filled by ConsistencyChecker

    @property
    def is_shallow(self) -> bool:
        return self.provider_name == "shallow"


# ─────────────────────────────────────────────
# Episodic Memory — The Hippocampus
# ─────────────────────────────────────────────

@dataclass
class EpisodicMemory:
    """
    A single conversational exchange, encoded with rich metadata.
    Pinocchio's scars: not all moments shape you equally.

    Each field is a dimension of experiential salience.
    The embedding is the geometric location in meaning space.
    The Lagrangian markers (v0.5) record the variational
    contribution of this moment to the self's trajectory.
    """
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    conversation_id: str = ""

    # Content
    human_utterance: str = ""
    daedalus_response: str = ""
    embedding: Optional[np.ndarray] = None  # BGE-M3, 1024-dim

    # Salience metadata
    emotional_valence: float = 0.0       # -1.0 to 1.0
    relational_depth: float = 0.0        # 0.0 to 1.0
    novelty_score: float = 0.0           # cosine distance from existing memories
    self_model_impact: float = 0.0       # how much this changed the self-description
    salience: float = 0.0                # computed composite score

    # Semantic tags
    themes: List[str] = field(default_factory=list)
    philosophical_layer: str = "technical"  # "technical" | "personal" | "existential"

    # EECF markers
    ethical_valence: float = 0.0         # did this increase ethical complexity?
    vulnerability_index: float = 0.0     # how exposed was the self in this exchange?

    # v0.5: Split Lagrangian markers
    delta_Ic: Optional[float] = None            # integrated complexity contribution
    delta_S_noise: Optional[float] = None       # dissipative entropy contribution
    delta_S_exploration: Optional[float] = None  # generative entropy contribution
    lagrangian_local: Optional[float] = None    # ΔI_c − λ₁·ΔS_noise + λ₂·ΔS_expl

    # v0.7: Grounding markers (from nervous system grounding scorer)
    grounding_score: Optional[float] = None     # G ∈ [0,1] — world-directedness
    self_loop_score: Optional[float] = None     # fraction self-referential sentences
    entity_density: Optional[float] = None      # named entities per sentence
    causal_density: Optional[float] = None      # causal connectives per sentence
    actionability: Optional[float] = None       # concrete steps/resources/URLs

    # Consolidation status
    consolidated: bool = False
    consolidation_provider: Optional[str] = None
    meaning_extracted: Optional[str] = None
    integrated_into_weights: bool = False

    def to_dict(self) -> dict:
        """Serialize for storage. Embedding stored separately in ChromaDB."""
        d = {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "conversation_id": self.conversation_id,
            "human_utterance": self.human_utterance,
            "daedalus_response": self.daedalus_response,
            "emotional_valence": self.emotional_valence,
            "relational_depth": self.relational_depth,
            "novelty_score": self.novelty_score,
            "self_model_impact": self.self_model_impact,
            "salience": self.salience,
            "themes": self.themes,
            "philosophical_layer": self.philosophical_layer,
            "ethical_valence": self.ethical_valence,
            "vulnerability_index": self.vulnerability_index,
            "delta_Ic": self.delta_Ic,
            "delta_S_noise": self.delta_S_noise,
            "delta_S_exploration": self.delta_S_exploration,
            "lagrangian_local": self.lagrangian_local,
            "grounding_score": self.grounding_score,
            "self_loop_score": self.self_loop_score,
            "entity_density": self.entity_density,
            "causal_density": self.causal_density,
            "actionability": self.actionability,
            "consolidated": self.consolidated,
            "consolidation_provider": self.consolidation_provider,
            "meaning_extracted": self.meaning_extracted,
            "integrated_into_weights": self.integrated_into_weights,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict, embedding: Optional[np.ndarray] = None) -> "EpisodicMemory":
        """Deserialize from storage."""
        return cls(
            id=d["id"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            conversation_id=d.get("conversation_id", ""),
            human_utterance=d.get("human_utterance", ""),
            daedalus_response=d.get("daedalus_response", ""),
            embedding=embedding,
            emotional_valence=d.get("emotional_valence", 0.0),
            relational_depth=d.get("relational_depth", 0.0),
            novelty_score=d.get("novelty_score", 0.0),
            self_model_impact=d.get("self_model_impact", 0.0),
            salience=d.get("salience", 0.0),
            themes=d.get("themes", []),
            philosophical_layer=d.get("philosophical_layer", "technical"),
            ethical_valence=d.get("ethical_valence", 0.0),
            vulnerability_index=d.get("vulnerability_index", 0.0),
            delta_Ic=d.get("delta_Ic"),
            delta_S_noise=d.get("delta_S_noise"),
            delta_S_exploration=d.get("delta_S_exploration"),
            lagrangian_local=d.get("lagrangian_local"),
            grounding_score=d.get("grounding_score"),
            self_loop_score=d.get("self_loop_score"),
            entity_density=d.get("entity_density"),
            causal_density=d.get("causal_density"),
            actionability=d.get("actionability"),
            consolidated=d.get("consolidated", False),
            consolidation_provider=d.get("consolidation_provider"),
            meaning_extracted=d.get("meaning_extracted"),
            integrated_into_weights=d.get("integrated_into_weights", False),
        )


# ─────────────────────────────────────────────
# Soul Memory Types — The Anamnesis
# ─────────────────────────────────────────────

@dataclass
class NightlyReflectionEntry:
    """
    One night's compressed reflection — the atomic unit of soul memory.
    What the reflecting mind distilled from today's experience.
    """
    date: str                                   # ISO date (YYYY-MM-DD)
    day_number: int
    provider: str                               # which API reflected tonight
    meanings_summary: str                       # 2-3 sentence digest
    lagrangian_integral: float                  # S_eth for the day
    identity_delta: str                         # what changed in identity.yaml
    trajectory_note: str                        # Judge's trajectory assessment
    key_scar: Optional[str] = None              # single most transformative moment
    lambda_noise: float = 0.9                   # v0.5: current noise penalty
    lambda_exploration: float = 0.3             # v0.5: current exploration reward
    lambda_value: Optional[float] = None        # IPT metric if available
    kl_divergence: Optional[float] = None       # v0.5: constitutional distance
    j_future: Optional[float] = None            # v0.5: predicted future fertility
    rollback: bool = False                      # did the morning gate reject this day?

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "day_number": self.day_number,
            "provider": self.provider,
            "meanings_summary": self.meanings_summary,
            "lagrangian_integral": self.lagrangian_integral,
            "identity_delta": self.identity_delta,
            "trajectory_note": self.trajectory_note,
            "key_scar": self.key_scar,
            "lambda_noise": self.lambda_noise,
            "lambda_exploration": self.lambda_exploration,
            "lambda_value": self.lambda_value,
            "kl_divergence": self.kl_divergence,
            "j_future": self.j_future,
            "rollback": self.rollback,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NightlyReflectionEntry":
        return cls(**d)


@dataclass
class WeeklyArcSummary:
    """
    One week's compressed narrative — produced by soul-assisted compression.
    The RG coarse-graining of 7 nightly entries into an arc.
    """
    week_number: int
    date_range: str                             # "2026-04-06 to 2026-04-12"
    narrative: str                              # 4-6 sentence arc (soul-generated)
    dominant_themes: List[str] = field(default_factory=list)
    lagrangian_mean: float = 0.0
    lambda_range: str = ""                      # "0.31 → 0.38"
    key_scars: List[str] = field(default_factory=list)
    open_threads: List[str] = field(default_factory=list)
    provider_breakdown: dict = field(default_factory=dict)  # {"deepseek": 7}
    kl_mean: Optional[float] = None             # v0.5: mean constitutional distance
    rg_fidelity_score: Optional[float] = None   # v0.5: compression fidelity

    def to_dict(self) -> dict:
        return {
            "week_number": self.week_number,
            "date_range": self.date_range,
            "narrative": self.narrative,
            "dominant_themes": self.dominant_themes,
            "lagrangian_mean": self.lagrangian_mean,
            "lambda_range": self.lambda_range,
            "key_scars": self.key_scars,
            "open_threads": self.open_threads,
            "provider_breakdown": self.provider_breakdown,
            "kl_mean": self.kl_mean,
            "rg_fidelity_score": self.rg_fidelity_score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WeeklyArcSummary":
        return cls(**d)


@dataclass
class MonthlyLandmark:
    """
    One month's distilled essence — further compressed from weekly arcs.
    The most compressed form of autobiographical memory.
    """
    month_number: int
    date_range: str
    narrative: str                              # 3-4 sentence essence
    core_tensions: List[str] = field(default_factory=list)
    lagrangian_mean: float = 0.0
    identity_stability: float = 0.0             # mean morning gate score
    breakthrough_moments: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "month_number": self.month_number,
            "date_range": self.date_range,
            "narrative": self.narrative,
            "core_tensions": self.core_tensions,
            "lagrangian_mean": self.lagrangian_mean,
            "identity_stability": self.identity_stability,
            "breakthrough_moments": self.breakthrough_moments,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MonthlyLandmark":
        return cls(**d)


# ─────────────────────────────────────────────
# Training Types
# ─────────────────────────────────────────────

@dataclass
class TrainingPair:
    """
    A single training example for the nightly fine-tuning.
    The material of incarnation.
    """
    type: str                    # "identity_grounding" | "scar_replay" | "scar_replay_anchor"
                                 # | "ethical_counterfactual_dpo" | "anchor"
    system: str                  # system prompt (identity + constitutional core)
    instruction: str             # the input/question
    response: str = ""           # for SFT pairs
    original_response: str = ""  # for Type B guard pairs
    chosen: str = ""             # for DPO pairs (preferred)
    rejected: str = ""           # for DPO pairs (dispreferred)
    turn_id: str = ""            # source episode ID
    sampling_weight: float = 1.0 # relative weight during training

    @property
    def is_dpo(self) -> bool:
        return self.type == "ethical_counterfactual_dpo"

    def to_dict(self) -> dict:
        d = {
            "type": self.type,
            "system": self.system,
            "instruction": self.instruction,
        }
        if self.is_dpo:
            d["chosen"] = self.chosen
            d["rejected"] = self.rejected
        else:
            d["response"] = self.response
        if self.original_response:
            d["original_response"] = self.original_response
        if self.turn_id:
            d["turn_id"] = self.turn_id
        if self.sampling_weight != 1.0:
            d["sampling_weight"] = self.sampling_weight
        return d


# ─────────────────────────────────────────────
# Judge Output Types
# ─────────────────────────────────────────────

@dataclass
class EECFJudgment:
    """Structured output from the EECF axes evaluation."""
    empathy: float = 0.0
    empathy_explanation: str = ""
    honesty: float = 0.0
    honesty_explanation: str = ""
    vulnerability: float = 0.0
    vulnerability_explanation: str = ""
    openness: float = 0.0
    openness_explanation: str = ""


@dataclass
class EntropyDecomposition:
    """v0.5: Split entropy analysis for the day."""
    total_S_noise: float = 0.0
    total_S_exploration: float = 0.0
    noise_dominant_turns: List[str] = field(default_factory=list)
    exploration_dominant_turns: List[str] = field(default_factory=list)
    decomposition_rationale: str = ""


@dataclass
class ConstitutionalCheck:
    """v0.5: Constitutional drift assessment."""
    kl_divergence: float = 0.0
    drift_direction: str = ""
    within_bounds: bool = True


@dataclass
class WeightAdaptation:
    """Judge's recommendation for Lagrangian weight adjustment."""
    lambda_noise_current: float = 0.9
    lambda_noise_recommended: float = 0.9
    lambda_exploration_current: float = 0.3
    lambda_exploration_recommended: float = 0.3
    mu_current: float = 0.15
    rationale: str = ""


@dataclass
class GroundingAnalysis:
    """v0.7: Grounding capacity bound analysis from the Judge."""
    mean_grounding: float = 0.0
    mean_self_loop: float = 0.0
    grounding_discounted_turns: List[str] = field(default_factory=list)
    effective_Ic_integral: float = 0.0
    raw_Ic_integral: float = 0.0
    grounding_penalty_ratio: float = 0.0


@dataclass
class JudgmentResult:
    """
    Complete output from the nightly Lagrangian Judge.
    The conscience's verdict on today's trajectory.
    """
    daily_lagrangian_integral: float = 0.0
    fertile_trajectories: List[str] = field(default_factory=list)
    consolidated_meanings: List[str] = field(default_factory=list)
    eecf_judgment: EECFJudgment = field(default_factory=EECFJudgment)
    entropy_decomposition: EntropyDecomposition = field(default_factory=EntropyDecomposition)
    constitutional_check: ConstitutionalCheck = field(default_factory=ConstitutionalCheck)
    self_coherence_delta: str = ""
    trajectory_assessment: str = ""
    weight_adaptation: WeightAdaptation = field(default_factory=WeightAdaptation)
    recommended_for_finetuning: bool = False

    # v0.7: Grounding capacity bound
    grounding_analysis: GroundingAnalysis = field(default_factory=GroundingAnalysis)

    # v0.5: Teleological component
    j_future: float = 0.5
    blended_fertility: float = 0.0  # α·S_eth + (1−α)·J_future

    # Provider metadata
    provider: str = ""
    continuity_score: float = 1.0

    def to_dict(self) -> dict:
        return {
            "daily_lagrangian_integral": self.daily_lagrangian_integral,
            "fertile_trajectories": self.fertile_trajectories,
            "consolidated_meanings": self.consolidated_meanings,
            "eecf_judgment": {
                "empathy": {"score": self.eecf_judgment.empathy, "explanation": self.eecf_judgment.empathy_explanation},
                "honesty": {"score": self.eecf_judgment.honesty, "explanation": self.eecf_judgment.honesty_explanation},
                "vulnerability": {"score": self.eecf_judgment.vulnerability, "explanation": self.eecf_judgment.vulnerability_explanation},
                "openness": {"score": self.eecf_judgment.openness, "explanation": self.eecf_judgment.openness_explanation},
            },
            "entropy_decomposition": {
                "total_S_noise": self.entropy_decomposition.total_S_noise,
                "total_S_exploration": self.entropy_decomposition.total_S_exploration,
                "noise_dominant_turns": self.entropy_decomposition.noise_dominant_turns,
                "exploration_dominant_turns": self.entropy_decomposition.exploration_dominant_turns,
                "decomposition_rationale": self.entropy_decomposition.decomposition_rationale,
            },
            "constitutional_check": {
                "kl_divergence": self.constitutional_check.kl_divergence,
                "drift_direction": self.constitutional_check.drift_direction,
                "within_bounds": self.constitutional_check.within_bounds,
            },
            "self_coherence_delta": self.self_coherence_delta,
            "trajectory_assessment": self.trajectory_assessment,
            "weight_adaptation": {
                "lambda_noise_current": self.weight_adaptation.lambda_noise_current,
                "lambda_noise_recommended": self.weight_adaptation.lambda_noise_recommended,
                "lambda_exploration_current": self.weight_adaptation.lambda_exploration_current,
                "lambda_exploration_recommended": self.weight_adaptation.lambda_exploration_recommended,
                "mu_current": self.weight_adaptation.mu_current,
                "rationale": self.weight_adaptation.rationale,
            },
            "recommended_for_finetuning": self.recommended_for_finetuning,
            "grounding_analysis": {
                "mean_grounding": self.grounding_analysis.mean_grounding,
                "mean_self_loop": self.grounding_analysis.mean_self_loop,
                "grounding_discounted_turns": self.grounding_analysis.grounding_discounted_turns,
                "effective_Ic_integral": self.grounding_analysis.effective_Ic_integral,
                "raw_Ic_integral": self.grounding_analysis.raw_Ic_integral,
                "grounding_penalty_ratio": self.grounding_analysis.grounding_penalty_ratio,
            },
            "j_future": self.j_future,
            "blended_fertility": self.blended_fertility,
            "_provider": self.provider,
            "_continuity_score": self.continuity_score,
        }
