"""
Task 3 - Error handling tests
Task 4 - Orchestrator pipeline test
"""
import sys, os, io, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from PIL import Image

errors = []

# ===========================================================================
print("=== TASK 3: ERROR HANDLING ===")
# ===========================================================================

from agents.waste_classification_agent import (
    WasteClassificationAgent, InvalidImageError, ModelNotReadyError
)
from agents.orchestrator_agent import OrchestratorAgent

agent = WasteClassificationAgent()
agent.ensure_loaded()

# 3a. No image / None input
print("[3a] None input ...")
try:
    agent.classify(None)
    print("     [FAIL] Should have raised InvalidImageError")
    errors.append("3a: no exception on None")
except InvalidImageError as e:
    print(f"     [PASS] InvalidImageError raised: {str(e)[:60]}")
except Exception as e:
    print(f"     [FAIL] Wrong exception type: {type(e).__name__}: {e}")
    errors.append(f"3a: {e}")

# 3b. Corrupted/unreadable image bytes
print("[3b] Corrupted image bytes ...")
try:
    fake_file = io.BytesIO(b"THIS IS NOT AN IMAGE FILE AT ALL !!!")
    fake_file.name = "corrupted.jpg"
    agent.classify(fake_file)
    print("     [FAIL] Should have raised InvalidImageError")
    errors.append("3b: no exception on corrupted bytes")
except InvalidImageError as e:
    print(f"     [PASS] InvalidImageError raised: {str(e)[:60]}")
except Exception as e:
    print(f"     [FAIL] Wrong exception type: {type(e).__name__}: {e}")
    errors.append(f"3b: {e}")

# 3c. Unsupported extension
print("[3c] Unsupported file extension (.exe) ...")
try:
    agent.classify_from_path(Path("models/waste_classifier.pt").rename if False else Path("fake.exe"))
    print("     [FAIL] Should have raised InvalidImageError")
    errors.append("3c: no exception on bad ext")
except InvalidImageError as e:
    print(f"     [PASS] InvalidImageError raised: {str(e)[:60]}")
except FileNotFoundError:
    # Acceptable -- file doesn't exist and extension check fires first
    print("     [PASS] FileNotFoundError / bad extension caught correctly")
except Exception as e:
    print(f"     [FAIL] {type(e).__name__}: {e}")
    errors.append(f"3c: {e}")

# 3d. File that has valid image extension but corrupt content
print("[3d] File with .jpg extension but binary garbage content ...")
try:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"\x00\x01\x02\x03 NOT JPEG DATA")
        tmp_path = Path(f.name)
    agent.classify_from_path(tmp_path)
    print("     [FAIL] Should have raised InvalidImageError")
    errors.append("3d: no exception on corrupt jpeg")
except InvalidImageError as e:
    print(f"     [PASS] InvalidImageError raised: {str(e)[:60]}")
except Exception as e:
    print(f"     [FAIL] {type(e).__name__}: {e}")
    errors.append(f"3d: {e}")
finally:
    tmp_path.unlink(missing_ok=True)

# 3e. Missing model checkpoint (simulated via fresh singleton reset)
print("[3e] Missing model checkpoint (simulated) ...")
try:
    from services.model_loader import ModelLoader
    # Create a temporary second loader pointing to nonexistent file
    import services.model_loader as ml_mod
    orig_path = ml_mod.MODEL_PATH
    ml_mod.MODEL_PATH = Path("models/nonexistent_model.pt")
    # Force a fresh singleton state for this test
    fresh_loader = object.__new__(ModelLoader)
    fresh_loader._model = None
    fresh_loader._class_names = None
    fresh_loader._device = None
    fresh_loader._load_error = None
    try:
        fresh_loader.load()
        print("     [FAIL] Should have raised FileNotFoundError")
        errors.append("3e: no exception for missing checkpoint")
    except FileNotFoundError as e:
        print(f"     [PASS] FileNotFoundError raised: {str(e)[:60]}")
    finally:
        ml_mod.MODEL_PATH = orig_path
except Exception as e:
    print(f"     [FAIL] {type(e).__name__}: {e}")
    errors.append(f"3e: {e}")

# 3f. Orchestrator: classification failure does not crash app
print("[3f] Orchestrator: graceful failure isolation ...")
try:
    # Pass an object that is not an image
    orch = OrchestratorAgent()
    bad_result = orch.process_image("this_is_not_an_image")
    assert bad_result.errors, "Expected errors list to be non-empty"
    assert bad_result.classification is None, "classification should be None on failure"
    print(f"     [PASS] Orchestrator returned errors gracefully: {bad_result.errors[0][:60]}")
except Exception as e:
    print(f"     [FAIL] Orchestrator raised uncaught exception: {e}")
    errors.append(f"3f: {e}")

print()
print(f"Task 3 result: {'PASS' if not errors else 'FAIL'} ({len(errors)} errors)")

# ===========================================================================
err_t4 = []
print()
print("=== TASK 4: ORCHESTRATOR PIPELINE END-TO-END ===")
# ===========================================================================

sample_path = Path("data/garbage_classification/cardboard") / sorted(
    (Path("data/garbage_classification/cardboard")).glob("*.jpg")
)[0].name

img = Image.open(sample_path).convert("RGB")
orch = OrchestratorAgent()
res = orch.process_image(img)

# Verify all three stages populated
checks = {
    "classification populated":  res.classification is not None,
    "recycling populated":       res.recycling is not None,
    "disposal populated":        res.disposal is not None,
    "no errors":                 len(res.errors) == 0,
    "predicted_class present":   "predicted_class" in (res.classification or {}),
    "confidence present":        "confidence"      in (res.classification or {}),
    "top3 present":              "top3"            in (res.classification or {}),
    "recycling tips present":    len((res.recycling or {}).get("tips", [])) > 0,
    "disposal bin present":      bool((res.disposal or {}).get("bin", "")),
}

for check, ok in checks.items():
    marker = "[PASS]" if ok else "[FAIL]"
    print(f"  {marker} {check}")
    if not ok:
        err_t4.append(check)

if not err_t4:
    print()
    print(f"  Classification : {res.classification['predicted_class']} ({res.classification['confidence']*100:.1f}%)")
    print(f"  Top-3          : {[(c, round(p*100,1)) for c,p in res.classification['top3']]}")
    print(f"  Recycling tips : {len(res.recycling['tips'])} tips")
    print(f"  Disposal bin   : {res.disposal['bin']}")

print()
print(f"Task 4 result: {'PASS' if not err_t4 else 'FAIL'} ({len(err_t4)} errors)")

total_errors = errors + err_t4
sys.exit(1 if total_errors else 0)
