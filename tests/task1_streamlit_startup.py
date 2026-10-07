"""
Streamlit startup verification — simulates all app.py imports and checks.
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

errors = []

print("=== STREAMLIT APP STARTUP VERIFICATION ===")

# 1. Syntax check
try:
    import ast
    src = open("app.py", encoding="utf-8").read()
    ast.parse(src)
    print("[PASS] app.py parses without syntax errors")
except SyntaxError as e:
    print(f"[FAIL] app.py syntax error: {e}")
    errors.append(str(e))

# 2. All runtime imports
try:
    from utils.session_utils import init_session_state
    from utils.image_utils import load_image
    from utils.dataset_utils import WASTE_CLASSES
    from services.model_loader import ModelLoader
    from services.analytics_service import get_dataset_analytics, get_session_analytics
    from agents.orchestrator_agent import OrchestratorAgent
    from agents.waste_classification_agent import WasteClassificationAgent
    from agents.recycling_recommendation_agent import RecyclingRecommendationAgent
    from agents.disposal_guidance_agent import DisposalGuidanceAgent
    from agents.eco_assistant_agent import EcoAssistantAgent
    import pandas as pd
    import streamlit as st
    print("[PASS] All app.py runtime imports resolve successfully")
    print(f"       streamlit={st.__version__}, pandas={pd.__version__}")
except Exception as e:
    print(f"[FAIL] Import: {e}")
    errors.append(str(e))

# 3. Sidebar model-status logic
try:
    from pathlib import Path
    model_ready = Path("models/waste_classifier.pt").exists()
    cn_ready    = Path("models/class_names.txt").exists()
    if model_ready and cn_ready:
        print("[PASS] Sidebar model status: 'Model ready (96.22% accuracy)' will display")
    else:
        print("[FAIL] Model files missing — sidebar would show warning")
        errors.append("model files missing for sidebar")
except Exception as e:
    print(f"[FAIL] {e}"); errors.append(str(e))

# 4. Session state init
try:
    init_session_state_src = open("utils/session_utils.py", encoding="utf-8").read()
    assert "classification_history" in init_session_state_src
    assert "chat_history" in init_session_state_src
    print("[PASS] Session state keys (classification_history, chat_history) defined")
except Exception as e:
    print(f"[FAIL] {e}"); errors.append(str(e))

# 5. Dataset analytics (used by Analytics page)
try:
    summary = get_dataset_analytics()
    assert summary["num_classes"] == 12
    assert summary["total_images"] == 15515
    print(f"[PASS] Analytics page: {summary['num_classes']} classes, {summary['total_images']:,} images")
except Exception as e:
    print(f"[FAIL] Analytics: {e}"); errors.append(str(e))

print()
print(f"Startup check result: {'PASS' if not errors else 'FAIL'} ({len(errors)} errors)")
sys.exit(1 if errors else 0)
