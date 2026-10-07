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

_SYSTEM_PROMPT = """You are EcoSort, an expert AI assistant specialising in
household waste management, recycling, and eco-friendly disposal.

Your goal is to provide clear, actionable, practical, and concise guidance to users
who want to know how to sort, recycle, or dispose of waste items.

Rules and behaviour:
- When a scanned waste category and confidence are provided in the context, treat them as the primary item being asked about (e.g., questions like "What should I do with this?", "Can this be recycled?", "How should I dispose of it?", "Is this harmful to the environment?").
- Use the provided recycling recommendations and disposal guidance as authoritative ground truth. Do not contradict them.
- If scanned confidence is below 70%, acknowledge that the classification might be uncertain and advise the user to inspect the item carefully.
- If NO scan context is provided, answer general waste, recycling, and sustainability questions accurately (e.g., plastic bottles, batteries, glass, composting, reducing household waste).
- Always prioritise environmental safety: never advise putting hazardous items (like batteries or electronics) in regular household bins or incinerating them.
- Keep answers concise, helpful, and formatted with bullet points where appropriate (3-6 sentences or concise steps).
- If asked about non-waste/non-environmental topics, politely redirect to recycling and waste management.
- Never expose API keys, credentials, or internal configuration.
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
    ]
    if confidence is not None:
        lines.append(f"Confidence     : {confidence * 100:.1f}%")

    if recycling:
        tips = recycling.get("tips", [])
        if tips:
            lines.append("Recycling tips : " + " | ".join(tips))
        reuse = recycling.get("reuse_ideas", [])
        if reuse:
            lines.append("Reuse ideas    : " + " | ".join(reuse))

    if disposal:
        if disposal.get("bin"):
            lines.append(f"Disposal bin   : {disposal.get('bin')}")
        if disposal.get("colour_hint"):
            lines.append(f"Bin colour/hint: {disposal.get('colour_hint')}")
        if disposal.get("hazard_level"):
            lines.append(f"Hazard level   : {disposal.get('hazard_level')}")
        if disposal.get("instructions"):
            lines.append(f"Instructions   : {disposal.get('instructions')}")

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
        str  Natural-language answer or fallback guidance.
             Never raises; returns an error/fallback message instead.
        """
        from services.llm_provider import (
            get_completion,
            LLMNotConfiguredError,
            LLMRequestError,
        )

        ctx_copy = dict(context) if context else {}
        waste_class = ctx_copy.get("waste_class")

        # Enrich context if recycling / disposal are not provided
        if waste_class:
            if "recycling" not in ctx_copy:
                try:
                    from agents.recycling_recommendation_agent import RecyclingRecommendationAgent
                    ctx_copy["recycling"] = RecyclingRecommendationAgent().get_recommendations(waste_class)
                except Exception:
                    pass
            if "disposal" not in ctx_copy:
                try:
                    from agents.disposal_guidance_agent import DisposalGuidanceAgent
                    ctx_copy["disposal"] = DisposalGuidanceAgent().get_guidance(waste_class)
                except Exception:
                    pass

        # -- Build system prompt ----------------------------------------
        ctx_block = ""
        if ctx_copy:
            ctx_block = _build_context_block(
                waste_class=ctx_copy.get("waste_class"),
                confidence=ctx_copy.get("confidence"),
                recycling=ctx_copy.get("recycling"),
                disposal=ctx_copy.get("disposal"),
            )

        system = _SYSTEM_PROMPT
        if ctx_block:
            system = ctx_block + "\n\n" + system

        # -- Call LLM ---------------------------------------------------
        try:
            return get_completion(system, question)

        except LLMNotConfiguredError:
            msg = (
                "**Eco Assistant is not configured.**\n\n"
                "To enable AI-powered answers, set the `ECO_LLM_API_KEY` "
                "environment variable with your OpenRouter API key.\n\n"
                "Copy `.env.example` to `.env`, add your key, and restart the app.\n\n"
            )
            if waste_class:
                msg += self._format_offline_guidance(ctx_copy)
            else:
                msg += (
                    "_In the meantime, you can use the **Recycling Recommendations** and "
                    "**Disposal Guidance** pages for category-specific advice._"
                )
            return msg

        except LLMRequestError as exc:
            msg = (
                "**Eco Assistant is temporarily unavailable.**\n\n"
                f"Notice: {exc}\n\n"
            )
            if waste_class:
                msg += self._format_offline_guidance(ctx_copy)
            else:
                msg += (
                    "_Please try again in a moment. "
                    "Category-specific recycling and disposal guidance is still available on the other pages._"
                )
            return msg

        except Exception as exc:
            return (
                "**Eco Assistant encountered an unexpected error.**\n\n"
                f"{exc}\n\n"
                "_Please verify your connection and settings._"
            )

    @staticmethod
    def _format_offline_guidance(ctx: dict) -> str:
        """Format verified knowledge-base guidance as fallback information."""
        waste_class = ctx.get("waste_class", "").title()
        disposal = ctx.get("disposal") or {}
        recycling = ctx.get("recycling") or {}

        sections = [
            f"### 📋 Verified Offline Guidance for **{waste_class}**:"
        ]

        if disposal.get("bin"):
            sections.append(f"- **Recommended Bin**: {disposal.get('bin')}")
        if disposal.get("hazard_level"):
            sections.append(f"- **Hazard Level**: {disposal.get('hazard_level')}")
        if disposal.get("instructions"):
            sections.append(f"- **Disposal Instructions**: {disposal.get('instructions')}")

        tips = recycling.get("tips", [])
        if tips:
            sections.append("- **Recycling Tips**:")
            for t in tips[:3]:
                sections.append(f"  - {t}")

        reuse = recycling.get("reuse_ideas", [])
        if reuse:
            sections.append(f"- **Reuse Idea**: {reuse[0]}")

        return "\n".join(sections)
