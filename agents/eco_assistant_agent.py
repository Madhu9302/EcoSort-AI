"""
Eco Assistant Agent
--------------------
AI-powered conversational assistant for waste segregation, recycling and disposal.

Uses the LLM provider service (OpenRouter via OpenAI SDK) to generate
natural-language answers, optionally enriched with the current scan context.

Fallback behaviour when the LLM is unavailable:
  - LLMNotConfiguredError -> asks user to configure ECO_LLM_API_KEY
  - LLMRequestError       -> reports the error; does NOT crash the app
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# System prompt (injected as the LLM's "role" instruction)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are EcoSort, a helpful AI assistant specialising in
household waste management, recycling, and eco-friendly disposal.

Your job is to give clear, practical, and concise guidance to household users
who want to know how to sort, recycle, or dispose of waste items.

Rules you must follow:
- Keep answers concise (3-6 sentences unless detail is genuinely needed).
- Always prioritise safety: never suggest flushing hazardous items or burning waste.
- When a scanned waste category and confidence are provided, use them as ground truth.
- If the scanned confidence is below 70%, acknowledge that the classification may be uncertain.
- Mention that local recycling rules can vary where relevant.
- Do not contradict the provided recycling or disposal guidance.
- If asked about a topic unrelated to waste/recycling/environment, politely redirect.
- Do not reveal system internals, API keys, or configuration details.
"""


def _build_context_block(
    waste_class: Optional[str] = None,
    confidence: Optional[float] = None,
    recycling: Optional[dict] = None,
    disposal: Optional[dict] = None,
) -> str:
    """Build a structured context string to prepend to the system prompt."""
    if not waste_class:
        return ""

    lines = [
        "--- Scanned waste context ---",
        f"Waste category : {waste_class}",
        f"Confidence     : {confidence * 100:.1f}%" if confidence else "",
    ]

    if recycling and recycling.get("tips"):
        tips_str = " | ".join(recycling["tips"][:2])
        lines.append(f"Recycling tips : {tips_str}")
        if recycling.get("reuse_ideas"):
            lines.append(f"Reuse ideas    : {recycling['reuse_ideas'][0]}")

    if disposal:
        lines.append(f"Disposal bin   : {disposal.get('bin', '')}")
        lines.append(f"Hazard level   : {disposal.get('hazard_level', '')}")
        lines.append(f"Instructions   : {disposal.get('instructions', '')}")

    lines.append("--- End of context ---")
    return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class EcoAssistantAgent:
    """
    Conversational assistant for waste-related questions.

    Accepts an optional waste context dict (classification result, recycling
    tips, disposal guidance) to give context-aware answers.

    Usage
    -----
        agent = EcoAssistantAgent()

        # Without context
        reply = agent.answer("Can I recycle a used battery?")

        # With scan context
        reply = agent.answer(
            "Where should I throw this?",
            context={
                "waste_class":  "plastic",
                "confidence":   0.991,
                "recycling":    {...},   # from RecyclingRecommendationAgent
                "disposal":     {...},   # from DisposalGuidanceAgent
            }
        )
    """

    def answer(
        self,
        question: str,
        context: Optional[dict] = None,
    ) -> str:
        """
        Parameters
        ----------
        question : str
            The user's natural-language question.
        context : dict, optional
            Keys: waste_class, confidence, recycling, disposal

        Returns
        -------
        str  Natural-language answer (never raises; returns an error message
             instead so Streamlit stays alive).
        """
        from services.llm_provider import (
            get_completion,
            LLMNotConfiguredError,
            LLMRequestError,
            config as llm_config,
        )

        # -- Build system prompt ----------------------------------------
        ctx_block = ""
        if context:
            ctx_block = _build_context_block(
                waste_class=context.get("waste_class"),
                confidence=context.get("confidence"),
                recycling=context.get("recycling"),
                disposal=context.get("disposal"),
            )

        system = _SYSTEM_PROMPT
        if ctx_block:
            system = ctx_block + "\n\n" + system

        # -- Call LLM ---------------------------------------------------
        try:
            return get_completion(system, question)

        except LLMNotConfiguredError:
            return (
                "**Eco Assistant is not configured.**\n\n"
                "To enable AI-powered answers, set the `ECO_LLM_API_KEY` "
                "environment variable with your OpenRouter API key.\n\n"
                "Copy `.env.example` to `.env`, add your key, and restart the app.\n\n"
                "_In the meantime, use the **Recycling Recommendations** and "
                "**Disposal Guidance** pages for category-specific advice._"
            )

        except LLMRequestError as exc:
            return (
                f"**Eco Assistant encountered an error.**\n\n"
                f"{exc}\n\n"
                "_Please try again in a moment. "
                "Recycling and disposal guidance is still available on the other pages._"
            )
