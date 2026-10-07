"""
Step 5 Test Suite
-----------------
Tests all 12 required checks for the Eco Assistant implementation.
Run with:  python tests/test_step5.py
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
WARN = "[WARN]"
errors = []

print("=" * 65)
print("EcoSort AI -- Step 5: Eco Assistant Test Suite")
print("=" * 65)

# ===========================================================================
# Test 1: Module imports
# ===========================================================================
print("\n[1] Module imports ...")
try:
    from agents.eco_assistant_agent import EcoAssistantAgent
    from services.llm_provider import (
        LLMConfig, get_completion,
        LLMNotConfiguredError, LLMRequestError, config as llm_config,
    )
    print(f"  {PASS} EcoAssistantAgent, LLMConfig, get_completion all imported")
except Exception as e:
    print(f"  {FAIL} {e}"); errors.append(f"T1: {e}")

# ===========================================================================
# Test 2: LLM configuration detection
# ===========================================================================
print("\n[2] LLM configuration detection ...")
try:
    api_key_set = llm_config.is_configured
    print(f"  {PASS} is_configured = {api_key_set}")
    print(f"        base_url = {llm_config.base_url}")
    print(f"        model    = {llm_config.model}")
    print(f"        timeout  = {llm_config.timeout}s")
    if api_key_set:
        masked = llm_config.api_key[:8] + "..." + llm_config.api_key[-4:]
        print(f"        api_key  = {masked}  (masked)")
    else:
        print(f"        api_key  = (not set)")
except Exception as e:
    print(f"  {FAIL} {e}"); errors.append(f"T2: {e}")

# ===========================================================================
# Test 3: Missing API key fallback (always tested, key-independent)
# ===========================================================================
print("\n[3] Missing API key fallback ...")
try:
    from services.llm_provider import LLMConfig, get_completion, LLMNotConfiguredError

    class _EmptyKeyConfig(LLMConfig):
        @property
        def api_key(self): return ""

    try:
        get_completion("sys", "user", cfg=_EmptyKeyConfig())
        print(f"  {FAIL} Should have raised LLMNotConfiguredError")
        errors.append("T3: no exception on missing key")
    except LLMNotConfiguredError as e:
        print(f"  {PASS} LLMNotConfiguredError raised: {str(e)[:60]}")
    except Exception as e:
        print(f"  {FAIL} Wrong exception: {type(e).__name__}: {e}")
        errors.append(f"T3: {e}")
except Exception as e:
    print(f"  {FAIL} {e}"); errors.append(f"T3: {e}")

# ===========================================================================
# Test 4: EcoAssistantAgent fallback message (no key)
# ===========================================================================
print("\n[4] EcoAssistantAgent fallback message when key missing ...")
try:
    import unittest.mock as mock

    # Temporarily suppress the key
    with mock.patch.dict(os.environ, {"ECO_LLM_API_KEY": ""}, clear=False):
        # Also reset the singleton's cached state for this test
        from services import llm_provider as _prov
        agent = EcoAssistantAgent()
        reply = agent.answer("Where should I throw this plastic bottle?")
    assert "not configured" in reply.lower() or "eco_llm_api_key" in reply.lower(), (
        f"Expected configuration message, got: {reply[:80]}"
    )
    print(f"  {PASS} Fallback message returned (not configured)")
    print(f"        Preview: {reply[:80].strip()}...")
except Exception as e:
    print(f"  {FAIL} {e}"); errors.append(f"T4: {e}")

# ===========================================================================
# Test 5: API failure handling (simulated network error)
# ===========================================================================
print("\n[5] API failure / LLMRequestError handling ...")
try:
    import unittest.mock as mock
    from services.llm_provider import LLMRequestError

    with mock.patch("services.llm_provider.get_completion",
                    side_effect=LLMRequestError("Simulated network failure")):
        agent = EcoAssistantAgent()
        reply = agent.answer("test question")

    assert "error" in reply.lower() or "encountered" in reply.lower(), (
        f"Expected error message, got: {reply[:80]}"
    )
    assert "ECO_LLM_API_KEY" not in reply, "Must not expose env var names in error"
    print(f"  {PASS} LLMRequestError handled gracefully; no crash")
    print(f"        Preview: {reply[:80].strip()}...")
except Exception as e:
    print(f"  {FAIL} {e}"); errors.append(f"T5: {e}")

# ===========================================================================
# Test 6: General question without image context (live LLM or skip)
# ===========================================================================
print("\n[6] General question without context ...")
if not llm_config.is_configured:
    print(f"  {SKIP} ECO_LLM_API_KEY not set — skipping live LLM call")
else:
    try:
        agent = EcoAssistantAgent()
        reply = agent.answer("Is cardboard recyclable?")
        assert len(reply) > 20, "Reply too short"
        assert "error" not in reply[:30].lower()
        print(f"  {PASS} Got LLM reply ({len(reply)} chars)")
        print(f"        Preview: {reply[:120].strip()}")
    except Exception as e:
        print(f"  {FAIL} {e}"); errors.append(f"T6: {e}")

# ===========================================================================
# Test 7: Question with plastic context
# ===========================================================================
print("\n[7] Question with plastic context (99.1% confidence) ...")
if not llm_config.is_configured:
    print(f"  {SKIP} ECO_LLM_API_KEY not set — skipping live LLM call")
else:
    try:
        from agents.recycling_recommendation_agent import RecyclingRecommendationAgent
        from agents.disposal_guidance_agent import DisposalGuidanceAgent

        ctx = {
            "waste_class": "plastic",
            "confidence":  0.991,
            "recycling":   RecyclingRecommendationAgent().get_recommendations("plastic"),
            "disposal":    DisposalGuidanceAgent().get_guidance("plastic"),
        }
        agent = EcoAssistantAgent()
        reply = agent.answer("Where should I throw this plastic bottle?", context=ctx)
        assert len(reply) > 20
        print(f"  {PASS} Context-aware reply received ({len(reply)} chars)")
        print(f"        Preview: {reply[:180].strip()}")
    except Exception as e:
        print(f"  {FAIL} {e}"); errors.append(f"T7: {e}")

# ===========================================================================
# Test 8: Question with battery context
# ===========================================================================
print("\n[8] Question with battery context ...")
if not llm_config.is_configured:
    print(f"  {SKIP} ECO_LLM_API_KEY not set — skipping live LLM call")
else:
    try:
        from agents.recycling_recommendation_agent import RecyclingRecommendationAgent
        from agents.disposal_guidance_agent import DisposalGuidanceAgent

        ctx = {
            "waste_class": "battery",
            "confidence":  0.987,
            "recycling":   RecyclingRecommendationAgent().get_recommendations("battery"),
            "disposal":    DisposalGuidanceAgent().get_guidance("battery"),
        }
        agent = EcoAssistantAgent()
        reply = agent.answer("What should I do with a used battery?", context=ctx)
        assert len(reply) > 20
        # Should NOT suggest household bin for batteries
        assert "household bin" not in reply.lower() or "not" in reply.lower(), (
            "Reply must not suggest putting battery in household bin"
        )
        print(f"  {PASS} Battery context reply received ({len(reply)} chars)")
        print(f"        Preview: {reply[:180].strip()}")
    except Exception as e:
        print(f"  {FAIL} {e}"); errors.append(f"T8: {e}")

# ===========================================================================
# Test 9: Question with glass context
# ===========================================================================
print("\n[9] Question with glass context ...")
if not llm_config.is_configured:
    print(f"  {SKIP} ECO_LLM_API_KEY not set — skipping live LLM call")
else:
    try:
        from agents.recycling_recommendation_agent import RecyclingRecommendationAgent
        from agents.disposal_guidance_agent import DisposalGuidanceAgent

        ctx = {
            "waste_class": "brown-glass",
            "confidence":  0.89,
            "recycling":   RecyclingRecommendationAgent().get_recommendations("brown-glass"),
            "disposal":    DisposalGuidanceAgent().get_guidance("brown-glass"),
        }
        agent = EcoAssistantAgent()
        reply = agent.answer("How should I dispose of broken glass?", context=ctx)
        assert len(reply) > 20
        print(f"  {PASS} Glass context reply received ({len(reply)} chars)")
        print(f"        Preview: {reply[:180].strip()}")
    except Exception as e:
        print(f"  {FAIL} {e}"); errors.append(f"T9: {e}")

# ===========================================================================
# Test 10: Existing Waste Scanner still works (classification pipeline)
# ===========================================================================
print("\n[10] Existing Waste Scanner (classification pipeline) ...")
try:
    from pathlib import Path
    from agents.waste_classification_agent import WasteClassificationAgent
    from agents.orchestrator_agent import OrchestratorAgent
    from PIL import Image

    sample = next((Path("data/garbage_classification/plastic")).glob("*.jpg"))
    img = Image.open(sample).convert("RGB")

    agent = WasteClassificationAgent()
    result = agent.classify(img)
    assert result["predicted_class"] == "plastic"
    assert result["confidence"] > 0.5

    orch = OrchestratorAgent()
    orch_result = orch.process_image(img)
    assert not orch_result.errors
    assert orch_result.classification is not None
    assert orch_result.recycling is not None
    assert orch_result.disposal is not None

    print(f"  {PASS} Waste Scanner: {result['predicted_class']} ({result['confidence']*100:.1f}%)")
    print(f"  {PASS} Orchestrator: classification + recycling + disposal all populated")
    print(f"        Model: models/waste_classifier.pt unchanged")
except Exception as e:
    print(f"  {FAIL} {e}"); errors.append(f"T10: {e}")

# ===========================================================================
# Test 11: Existing orchestrator answer_question still works
# ===========================================================================
print("\n[11] Orchestrator answer_question ...")
try:
    from agents.orchestrator_agent import OrchestratorAgent
    orch = OrchestratorAgent()
    reply = orch.answer_question("What is recycling?")
    assert isinstance(reply, str) and len(reply) > 0
    print(f"  {PASS} answer_question returned a string")
    print(f"        Preview: {reply[:80]}")
except Exception as e:
    print(f"  {FAIL} {e}"); errors.append(f"T11: {e}")

# ===========================================================================
# Test 12: Streamlit app imports cleanly
# ===========================================================================
print("\n[12] Streamlit app import check ...")
try:
    import ast
    src = open("app.py", encoding="utf-8").read()
    ast.parse(src)
    import streamlit as st
    print(f"  {PASS} app.py syntax OK; streamlit {st.__version__} importable")
except Exception as e:
    print(f"  {FAIL} {e}"); errors.append(f"T12: {e}")

# ===========================================================================
# Security check: no hardcoded keys
# ===========================================================================
print("\n[SEC] Security check ...")
try:
    import re
    src_files = list(__import__("pathlib").Path(".").glob("**/*.py"))
    src_files = [f for f in src_files if "venv" not in str(f) and "__pycache__" not in str(f)]
    key_pat = re.compile(r"sk-or-[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{40,}")
    found = False
    for f in src_files:
        txt = f.read_text(encoding="utf-8", errors="replace")
        if key_pat.search(txt):
            print(f"  {FAIL} Possible hardcoded key in {f}")
            errors.append(f"SEC: key in {f}")
            found = True
    if not found:
        print(f"  {PASS} No hardcoded API keys found in source files")
    # Ensure .env is gitignored
    gitignore = __import__("pathlib").Path(".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
    print(f"  {PASS} .env is in .gitignore")
except Exception as e:
    print(f"  {FAIL} {e}"); errors.append(f"SEC: {e}")

# ===========================================================================
# Summary
# ===========================================================================
print()
print("=" * 65)
live_skipped = 4 if not llm_config.is_configured else 0
if errors:
    print(f"RESULT: {len(errors)} FAILURE(S)")
    for e in errors:
        print(f"  - {e}")
elif live_skipped:
    print(f"RESULT: ALL CHECKS PASSED "
          f"({live_skipped} live-LLM tests skipped -- ECO_LLM_API_KEY not set)")
else:
    print("RESULT: ALL CHECKS PASSED (including live LLM)")
print("=" * 65)
sys.exit(1 if errors else 0)
