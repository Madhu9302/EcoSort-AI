"""
Task 1 - Startup and import verification
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

errors = []

print("=== TASK 1: STARTUP / IMPORT CHECK ===")

# streamlit
try:
    import streamlit as st
    print(f"[PASS] streamlit {st.__version__} imported")
except Exception as e:
    print(f"[FAIL] streamlit: {e}"); errors.append(str(e))

# all app modules
try:
    from utils.session_utils import init_session_state
    from utils.dataset_utils import WASTE_CLASSES, get_dataset_summary
    from utils.image_utils import load_image
    print("[PASS] utils.*  (session_utils, dataset_utils, image_utils)")
except Exception as e:
    print(f"[FAIL] utils: {e}"); errors.append(str(e))

try:
    from services.model_loader import ModelLoader
    from services.analytics_service import get_dataset_analytics, get_session_analytics
    print("[PASS] services.* (model_loader, analytics_service)")
except Exception as e:
    print(f"[FAIL] services: {e}"); errors.append(str(e))

try:
    from agents.waste_classification_agent import WasteClassificationAgent
    from agents.recycling_recommendation_agent import RecyclingRecommendationAgent
    from agents.disposal_guidance_agent import DisposalGuidanceAgent
    from agents.eco_assistant_agent import EcoAssistantAgent
    from agents.orchestrator_agent import OrchestratorAgent
    print("[PASS] agents.* (all 5 agents)")
except Exception as e:
    print(f"[FAIL] agents: {e}"); errors.append(str(e))

# model files
try:
    from pathlib import Path
    pt = Path("models/waste_classifier.pt")
    cn = Path("models/class_names.txt")
    tm = Path("models/training_metrics.json")
    ev = Path("models/evaluation_report.json")
    cm = Path("models/confusion_matrix.png")
    for f in [pt, cn, tm, ev, cm]:
        assert f.exists(), f"Missing: {f}"
    print(f"[PASS] All 5 model artefacts present")
    print(f"       waste_classifier.pt : {pt.stat().st_size//1024:,} KB")
except Exception as e:
    print(f"[FAIL] Model files: {e}"); errors.append(str(e))

# model load + device
try:
    loader = ModelLoader()
    loader.load()
    import torch
    cuda = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda else "CPU"
    print(f"[PASS] ModelLoader: loaded, device={loader.device}, classes={len(loader.get_class_names())}")
    print(f"       GPU: {dev_name}")
except Exception as e:
    print(f"[FAIL] ModelLoader.load(): {e}"); errors.append(str(e))

# model path is relative (not hardcoded absolute)
try:
    import services.model_loader as ml_mod
    import inspect
    src = inspect.getsource(ml_mod)
    assert "D:\\" not in src and "C:\\" not in src, "Hardcoded absolute path found in model_loader!"
    print("[PASS] model_loader uses relative paths (no hardcoded absolute paths)")
except Exception as e:
    print(f"[WARN] Path check: {e}")

print()
print(f"Task 1 result: {'PASS' if not errors else 'FAIL'} ({len(errors)} errors)")
sys.exit(1 if errors else 0)
