"""
Orchestrator Agent
------------------
Coordinates all specialised sub-agents to process a single user request.

Image pipeline
--------------
    WasteClassificationAgent
            |
            v
    RecyclingRecommendationAgent
            |
            v
    DisposalGuidanceAgent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OrchestrationResult:
    classification: Optional[dict] = None       # from WasteClassificationAgent
    recycling:      Optional[dict] = None        # from RecyclingRecommendationAgent
    disposal:       Optional[dict] = None        # from DisposalGuidanceAgent
    assistant_reply: Optional[str] = None        # from EcoAssistantAgent
    errors: list[str] = field(default_factory=list)


class OrchestratorAgent:
    """
    Top-level agent that routes tasks to the correct specialised agents
    and assembles a unified OrchestrationResult.
    """

    def __init__(self) -> None:
        # Lazy imports avoid circular-import and heavy startup cost
        self._classification_agent = None
        self._recycling_agent      = None
        self._disposal_agent       = None
        self._assistant_agent      = None

    # ------------------------------------------------------------------
    # Internal helpers -- lazy initialisation
    # ------------------------------------------------------------------

    def _get_classification_agent(self):
        if self._classification_agent is None:
            from agents.waste_classification_agent import WasteClassificationAgent
            self._classification_agent = WasteClassificationAgent()
        return self._classification_agent

    def _get_recycling_agent(self):
        if self._recycling_agent is None:
            from agents.recycling_recommendation_agent import RecyclingRecommendationAgent
            self._recycling_agent = RecyclingRecommendationAgent()
        return self._recycling_agent

    def _get_disposal_agent(self):
        if self._disposal_agent is None:
            from agents.disposal_guidance_agent import DisposalGuidanceAgent
            self._disposal_agent = DisposalGuidanceAgent()
        return self._disposal_agent

    def _get_assistant_agent(self):
        if self._assistant_agent is None:
            from agents.eco_assistant_agent import EcoAssistantAgent
            self._assistant_agent = EcoAssistantAgent()
        return self._assistant_agent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_image(self, image) -> OrchestrationResult:
        """
        Full pipeline: classify -> recycling recommendation -> disposal guidance.

        Parameters
        ----------
        image : PIL.Image or file-like object
            The waste image uploaded by the user.

        Returns
        -------
        OrchestrationResult
            All fields populated on success; errors list is non-empty on failure.
        """
        result = OrchestrationResult()

        # -- Step 1: classify image --------------------------------------
        try:
            classification = self._get_classification_agent().classify(image)
            result.classification = classification
        except Exception as exc:
            result.errors.append(f"Classification failed: {exc}")
            return result   # cannot continue without a predicted class

        # -- Step 2: recycling recommendations ---------------------------
        try:
            predicted_class = classification["predicted_class"]
            result.recycling = self._get_recycling_agent().get_recommendations(predicted_class)
        except Exception as exc:
            result.errors.append(f"Recycling recommendation failed: {exc}")

        # -- Step 3: disposal guidance -----------------------------------
        try:
            result.disposal = self._get_disposal_agent().get_guidance(predicted_class)
        except Exception as exc:
            result.errors.append(f"Disposal guidance failed: {exc}")

        return result

    def answer_question(self, question: str, context: dict | None = None) -> str:
        """Forward a free-text question to the EcoAssistantAgent."""
        try:
            return self._get_assistant_agent().answer(question)
        except Exception as exc:
            return f"Eco Assistant encountered an error: {exc}"
