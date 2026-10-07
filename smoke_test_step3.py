"""
Step 3 Smoke Test
-----------------
Verifies model loading, inference, class order, CUDA, and Streamlit import.
Run with:  python smoke_test_step3.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "[PASS]"
FAIL = "[FAIL]"

errors = []

# ---------------------------------------------------------------------------
# 1. Import all affected modules
# ---------------------------------------------------------------------------
print("=" * 60)
print("EcoSort AI -- Step 3 Smoke Test")
print("=" * 60)
print()
print("[1] Importing modules ...")
try:
    from services.model_loader import ModelLoader
    from agents.waste_classification_agent import WasteClassificationAgent
    from agents.orchestrator_agent import OrchestratorAgent
    from agents.recycling_recommendation_agent import RecyclingRecommendationAgent
    from agents.disposal_guidance_agent import DisposalGuidanceAgent
    print(f"  {PASS} All modules imported successfully.")
except Exception as exc:
    print(f"  {FAIL} Import error: {exc}")
    errors.append(str(exc))
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Load the trained checkpoint
# ---------------------------------------------------------------------------
print()
print("[2] Loading model checkpoint ...")
try:
    loader = ModelLoader()
    loader.load()
    print(f"  {PASS} Model loaded successfully.")
    print(f"        Device : {loader.device}")
    print(f"        Classes: {len(loader.get_class_names())}")
except Exception as exc:
    print(f"  {FAIL} Model load error: {exc}")
    errors.append(str(exc))
    sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Confirm 12-class output
# ---------------------------------------------------------------------------
print()
print("[3] Verifying model output dimensions ...")
try:
    import torch
    model = loader.get_model()
    dummy = torch.zeros(1, 3, 224, 224).to(loader.device)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 12), f"Expected (1,12) got {out.shape}"
    print(f"  {PASS} Output shape: {tuple(out.shape)}  (12 classes confirmed)")
except Exception as exc:
    print(f"  {FAIL} {exc}")
    errors.append(str(exc))

# ---------------------------------------------------------------------------
# 4. Verify class order matches class_names.txt
# ---------------------------------------------------------------------------
print()
print("[4] Verifying class order ...")
try:
    from pathlib import Path
    txt_classes = [
        l.strip()
        for l in Path("models/class_names.txt").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    loaded_classes = loader.get_class_names()
    assert txt_classes == loaded_classes, (
        f"Class mismatch!\n  txt:    {txt_classes}\n  loaded: {loaded_classes}"
    )
    for i, cls in enumerate(loaded_classes):
        print(f"    {i:2d}  {cls}")
    print(f"  {PASS} Class order matches class_names.txt exactly.")
except Exception as exc:
    print(f"  {FAIL} {exc}")
    errors.append(str(exc))

# ---------------------------------------------------------------------------
# 5. Load a real dataset image and run inference
# ---------------------------------------------------------------------------
print()
print("[5] Running inference on a real dataset image ...")
try:
    import glob as _glob
    dataset_root = Path("data/garbage_classification")
    # Grab first image from 'plastic' (a mid-difficulty class)
    sample_images = list((dataset_root / "plastic").glob("*.jpg"))
    if not sample_images:
        sample_images = list(dataset_root.rglob("*.jpg"))
    assert sample_images, "No dataset images found under data/garbage_classification"
    sample_path = sample_images[0]

    agent  = WasteClassificationAgent()
    result = agent.classify_from_path(sample_path)

    print(f"  {PASS} Inference completed.")
    print(f"        Image          : {sample_path.name}")
    print(f"        Predicted class: {result['predicted_class']}")
    print(f"        Confidence     : {result['confidence']*100:.1f}%")
    print(f"        Top-3:")
    for rank, (cls, prob) in enumerate(result["top3"], 1):
        print(f"          {rank}. {cls:<15} {prob*100:.1f}%")
    print(f"        Device         : {result['device']}")
except Exception as exc:
    print(f"  {FAIL} Inference error: {exc}")
    errors.append(str(exc))

# ---------------------------------------------------------------------------
# 6. CUDA check
# ---------------------------------------------------------------------------
print()
print("[6] Checking CUDA availability ...")
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        print(f"  {PASS} CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"        Model device: {loader.device}")
        assert str(loader.device) == "cuda", "Expected CUDA device"
    else:
        print(f"       CUDA not available -- CPU mode active.")
    print(f"  {PASS} Device check passed.")
except Exception as exc:
    print(f"  {FAIL} {exc}")
    errors.append(str(exc))

# ---------------------------------------------------------------------------
# 7. Orchestrator end-to-end
# ---------------------------------------------------------------------------
print()
print("[7] Testing full orchestrator pipeline ...")
try:
    from PIL import Image
    img = Image.open(sample_path).convert("RGB")
    orch_result = OrchestratorAgent().process_image(img)
    assert not orch_result.errors, f"Orchestrator errors: {orch_result.errors}"
    assert orch_result.classification is not None
    assert orch_result.recycling      is not None
    assert orch_result.disposal       is not None
    print(f"  {PASS} Orchestrator pipeline completed.")
    print(f"        Classified as  : {orch_result.classification['predicted_class']}")
    print(f"        Recycling tips : {len(orch_result.recycling['tips'])} tips")
    print(f"        Disposal bin   : {orch_result.disposal['bin']}")
except Exception as exc:
    print(f"  {FAIL} {exc}")
    errors.append(str(exc))

# ---------------------------------------------------------------------------
# 8. Streamlit app import
# ---------------------------------------------------------------------------
print()
print("[8] Verifying Streamlit app imports ...")
try:
    import streamlit as st
    print(f"  {PASS} Streamlit {st.__version__} imported successfully.")
except Exception as exc:
    print(f"  {FAIL} {exc}")
    errors.append(str(exc))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
if errors:
    print(f"RESULT: {len(errors)} error(s) found:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("RESULT: ALL CHECKS PASSED")
print("=" * 60)
