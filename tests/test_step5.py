"""
Step 5 Test Suite
-----------------
Tests all 8 required checks for Eco Assistant (using mocked LLM responses
so zero OpenRouter credits are consumed) plus regression and security checks.

Run with:  python tests/test_step5.py
"""
import sys, os
from unittest import mock
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

PASS = "[PASS]"
FAIL = "[FAIL]"
errors = []

print("=" * 65)
print("EcoSort AI -- Step 5: Eco Assistant Test Suite")
print("=" * 65)

# ===========================================================================
# 1. Eco Assistant initialization
# ===========================================================================
print("\n[1] Eco Assistant initialization ...")
try:
    from agents.eco_assistant_agent import EcoAssistantAgent
    from services.llm_provider import (
        LLMConfig, get_completion,
        LLMNotConfiguredError, LLMRequestError, config as llm_config,
    )
    agent = EcoAssistantAgent()
    assert agent is not None
    assert hasattr(agent, "answer")
    assert llm_config.base_url.startswith("https://")
    print(f"  {PASS} EcoAssistantAgent initialized successfully")
    print(f"        Base URL: {llm_config.base_url}")
    print(f"        Configured model: {llm_config.model}")
except Exception as e:
    print(f"  {FAIL} {e}"); errors.append(f"T1: {e}")

# ===========================================================================
# 2. Missing API key fallback
# ===========================================================================
print("\n[2] Missing API key fallback ...")
try:
    with mock.patch.dict(os.environ, {"ECO_LLM_API_KEY": ""}, clear=False):
        agent = EcoAssistantAgent()
        # Test without context
        reply_no_ctx = agent.answer("How do I recycle paper?")
        assert "not configured" in reply_no_ctx.lower() or "eco_llm_api_key" in reply_no_ctx.lower(), (
            f"Expected configuration guidance, got: {reply_no_ctx[:80]}"
        )

        # Test with context (should include offline category guidance)
        reply_ctx = agent.answer("Where does this go?", context={"waste_class": "plastic"})
        assert "plastic" in reply_ctx.lower(), "Fallback must include category advice"
        assert "not configured" in reply_ctx.lower() or "eco_llm_api_key" in reply_ctx.lower()
        print(f"  {PASS} Missing key handled cleanly with offline fallback guidance")
        print(f"        Preview: {reply_ctx[:90].strip()}...")
except Exception as e:
    print(f"  {FAIL} {e}"); errors.append(f"T2: {e}")

# ===========================================================================
# 3. Successful mocked LLM response
# ===========================================================================
print("\n[3] Successful mocked LLM response ...")
try:
    mock_reply = "Plastic bottles can be recycled in your blue recycling bin after rinsing."
    with mock.patch("services.llm_provider.get_completion", return_value=mock_reply) as mocked_get:
        agent = EcoAssistantAgent()
        reply = agent.answer("Can I recycle this bottle?")
        assert reply == mock_reply
        assert mocked_get.called
        print(f"  {PASS} Mocked LLM response received accurately")
        print(f"        Reply: {reply}")
except Exception as e:
    print(f"  {FAIL} {e}"); errors.append(f"T3: {e}")

# ===========================================================================
# 4. OpenRouter/API failure fallback
# ===========================================================================
print("\n[4] OpenRouter/API failure fallback ...")
try:
    with mock.patch("services.llm_provider.get_completion",
                    side_effect=LLMRequestError("Simulated OpenRouter 503 error")):
        agent = EcoAssistantAgent()
        reply = agent.answer("How should I dispose of this?", context={"waste_class": "metal"})
        assert "unavailable" in reply.lower() or "error" in reply.lower(), (
            f"Expected unavailable/error notice, got: {reply[:80]}"
        )
        assert "metal" in reply.lower(), "Expected category fallback for metal"
        print(f"  {PASS} API failure caught gracefully without crashing")
        print(f"        Preview: {reply[:90].strip()}...")
except Exception as e:
    print(f"  {FAIL} {e}"); errors.append(f"T4: {e}")

# ===========================================================================
# 5. Context-aware response
# ===========================================================================
print("\n[5] Context-aware response ...")
try:
    captured = {}

    def fake_get_completion(system_prompt, user_message, cfg=None):
        captured["system"] = system_prompt
        captured["question"] = user_message
        return "Context-aware response generated."

    with mock.patch("services.llm_provider.get_completion", side_effect=fake_get_completion):
        agent = EcoAssistantAgent()
        scan_ctx = {
            "waste_class": "cardboard",
            "confidence": 0.985,
            "recycling": {"tips": ["Flatten boxes before binning."], "reuse_ideas": ["Use as seedling trays."]},
            "disposal": {"bin": "Blue Bin", "hazard_level": "None", "instructions": "Keep dry and place in bin."}
        }
        res = agent.answer("Can this be recycled?", context=scan_ctx)
        assert res == "Context-aware response generated."
        sys_txt = captured.get("system", "")
        assert "cardboard" in sys_txt
        assert "98.5%" in sys_txt
        assert "Blue Bin" in sys_txt
        assert "Flatten boxes" in sys_txt
        print(f"  {PASS} Scan context properly formatted into LLM instructions")
        print(f"        Context confirmed: category, 98.5% confidence, disposal bin, recycling tips")
except Exception as e:
    print(f"  {FAIL} {e}"); errors.append(f"T5: {e}")

# ===========================================================================
# 6. Plastic question
# ===========================================================================
print("\n[6] Plastic question ...")
try:
    mock_plastic_response = (
        "Plastic containers are typically recyclable in your blue bin. "
        "Rinse out any food or drink residue and check the resin code (1 and 2 are widely accepted)."
    )
    with mock.patch("services.llm_provider.get_completion", return_value=mock_plastic_response) as mocked_get:
        from agents.recycling_recommendation_agent import RecyclingRecommendationAgent
        from agents.disposal_guidance_agent import DisposalGuidanceAgent

        plastic_ctx = {
            "waste_class": "plastic",
            "confidence": 0.991,
            "recycling": RecyclingRecommendationAgent().get_recommendations("plastic"),
            "disposal": DisposalGuidanceAgent().get_guidance("plastic"),
        }
        agent = EcoAssistantAgent()
        reply = agent.answer("How do I dispose of a plastic bottle?", context=plastic_ctx)
        assert "plastic" in reply.lower()
        args, kwargs = mocked_get.call_args
        sent_system = args[0]
        assert "plastic" in sent_system
        assert "99.1%" in sent_system
        print(f"  {PASS} Plastic context & question handled correctly")
        print(f"        Reply: {reply[:100]}...")
except Exception as e:
    print(f"  {FAIL} {e}"); errors.append(f"T6: {e}")

# ===========================================================================
# 7. Battery question
# ===========================================================================
print("\n[7] Battery question ...")
try:
    mock_battery_response = (
        "Never place batteries in your regular household bin as they pose a serious fire hazard. "
        "Take them to a dedicated battery drop-off box at a local supermarket or recycling centre."
    )
    with mock.patch("services.llm_provider.get_completion", return_value=mock_battery_response) as mocked_get:
        from agents.recycling_recommendation_agent import RecyclingRecommendationAgent
        from agents.disposal_guidance_agent import DisposalGuidanceAgent

        battery_ctx = {
            "waste_class": "battery",
            "confidence": 0.987,
            "recycling": RecyclingRecommendationAgent().get_recommendations("battery"),
            "disposal": DisposalGuidanceAgent().get_guidance("battery"),
        }
        agent = EcoAssistantAgent()
        reply = agent.answer("What should I do with batteries?", context=battery_ctx)
        assert "never" in reply.lower() and "household bin" in reply.lower()
        args, kwargs = mocked_get.call_args
        sent_system = args[0]
        assert "battery" in sent_system
        assert "Hazardous Waste" in sent_system or "High" in sent_system
        print(f"  {PASS} Battery question handled with high-hazard safety rules")
        print(f"        Reply: {reply[:100]}...")
except Exception as e:
    print(f"  {FAIL} {e}"); errors.append(f"T7: {e}")

# ===========================================================================
# 8. Glass question
# ===========================================================================
print("\n[8] Glass question ...")
try:
    mock_glass_response = (
        "Brown glass bottles can be recycled in the brown glass section of your local bottle bank "
        "or kerbside glass box. Rinse thoroughly and recycle lids separately."
    )
    with mock.patch("services.llm_provider.get_completion", return_value=mock_glass_response) as mocked_get:
        from agents.recycling_recommendation_agent import RecyclingRecommendationAgent
        from agents.disposal_guidance_agent import DisposalGuidanceAgent

        glass_ctx = {
            "waste_class": "brown-glass",
            "confidence": 0.890,
            "recycling": RecyclingRecommendationAgent().get_recommendations("brown-glass"),
            "disposal": DisposalGuidanceAgent().get_guidance("brown-glass"),
        }
        agent = EcoAssistantAgent()
        reply = agent.answer("Can glass be recycled?", context=glass_ctx)
        assert "glass" in reply.lower()
        args, kwargs = mocked_get.call_args
        sent_system = args[0]
        assert "brown-glass" in sent_system
        print(f"  {PASS} Glass question handled with bottle bank / glass bin instructions")
        print(f"        Reply: {reply[:100]}...")
except Exception as e:
    print(f"  {FAIL} {e}"); errors.append(f"T8: {e}")

# ===========================================================================
# Regression checks
# ===========================================================================
print("\n[REGRESSION] Existing Waste Scanner & Orchestrator pipeline ...")
try:
    from agents.waste_classification_agent import WasteClassificationAgent
    from agents.orchestrator_agent import OrchestratorAgent
    from PIL import Image

    sample_img_path = next((Path("data/garbage_classification/plastic")).glob("*.jpg"))
    img = Image.open(sample_img_path).convert("RGB")

    clf_agent = WasteClassificationAgent()
    clf_result = clf_agent.classify(img)
    assert clf_result["predicted_class"] == "plastic"
    assert clf_result["confidence"] > 0.5

    orch = OrchestratorAgent()
    orch_result = orch.process_image(img)
    assert not orch_result.errors
    assert orch_result.classification is not None
    assert orch_result.recycling is not None
    assert orch_result.disposal is not None

    # Test orchestrator answer_question forwarding
    with mock.patch("agents.eco_assistant_agent.EcoAssistantAgent.answer", return_value="Orchestrator answer OK") as mocked_answer:
        orch_reply = orch.answer_question("Where does this go?", context={"waste_class": "plastic"})
        assert orch_reply == "Orchestrator answer OK"
        mocked_answer.assert_called_once_with("Where does this go?", context={"waste_class": "plastic"})

    print(f"  {PASS} Model loaded & classified plastic ({clf_result['confidence']*100:.1f}%)")
    print(f"  {PASS} Orchestrator pipeline fully operational (classification + recycling + disposal + Q&A)")
except Exception as e:
    print(f"  {FAIL} {e}"); errors.append(f"REGRESSION: {e}")

print("\n[REGRESSION] Streamlit app syntax & dependencies ...")
try:
    import ast
    src = open("app.py", encoding="utf-8").read()
    ast.parse(src)
    import streamlit as st
    print(f"  {PASS} app.py syntax valid, Streamlit {st.__version__} importable")
except Exception as e:
    print(f"  {FAIL} {e}"); errors.append(f"STREAMLIT: {e}")

print("\n[SEC] Security check (no exposed keys, .env gitignored) ...")
try:
    import re
    src_files = list(Path(".").glob("**/*.py"))
    src_files = [f for f in src_files if "venv" not in str(f) and "__pycache__" not in str(f)]
    key_pat = re.compile(r"sk-or-[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{40,}")
    found = False
    for f in src_files:
        txt = f.read_text(encoding="utf-8", errors="replace")
        if key_pat.search(txt):
            print(f"  {FAIL} Hardcoded key found in {f}")
            errors.append(f"SEC: key in {f}")
            found = True
    if not found:
        print(f"  {PASS} No hardcoded API keys in source files")

    gitignore_txt = Path(".gitignore").read_text(encoding="utf-8", errors="replace")
    assert ".env" in gitignore_txt, ".env missing from .gitignore"
    print(f"  {PASS} .env is protected in .gitignore")

    env_ex = Path(".env.example").read_text(encoding="utf-8", errors="replace")
    assert "your_openrouter_api_key_here" in env_ex
    print(f"  {PASS} .env.example contains only placeholders")
except Exception as e:
    print(f"  {FAIL} {e}"); errors.append(f"SEC: {e}")

# ===========================================================================
# Summary
# ===========================================================================
print()
print("=" * 65)
if errors:
    print(f"RESULT: {len(errors)} TEST(S) FAILED")
    for e in errors:
        print(f"  - {e}")
else:
    print("RESULT: ALL 8 STEP 5 TESTS + ALL REGRESSION CHECKS PASSED")
print("=" * 65)
sys.exit(1 if errors else 0)
