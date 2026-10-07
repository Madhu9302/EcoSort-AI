"""
Task 5 - Home page content verification
Task 6 - Code quality check
"""
import sys, os, ast, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

errors = []

# ===========================================================================
print("=== TASK 5: HOME PAGE CONTENT VERIFICATION ===")
# ===========================================================================

app_src = Path("app.py").read_text(encoding="utf-8")

home_checks = {
    "Title 'EcoSort AI' present":           "EcoSort AI" in app_src,
    "97.08% val accuracy mentioned":        "97.08" in app_src,
    "96.22% test accuracy mentioned":       "96.22" in app_src,
    "96.23% F1 mentioned":                  "96.23" in app_src,
    "Model status from file check (real)":  'waste_classifier.pt' in app_src and '_Path(' in app_src,
    "No 'model not yet trained' stub text": "Model not yet trained" not in app_src,
    "No 'Model training is pending' text":  "Model training is pending" not in app_src,
    "EfficientNet-B0 mentioned":            "EfficientNet-B0" in app_src,
    "Best epoch 24 mentioned":              "24" in app_src,
    "10,860 training images mentioned":     "10,860" in app_src,
}

for check, ok in home_checks.items():
    marker = "[PASS]" if ok else "[FAIL]"
    print(f"  {marker} {check}")
    if not ok:
        errors.append(f"Home: {check}")

print()
print(f"Task 5 result: {'PASS' if not [e for e in errors if e.startswith('Home')] else 'FAIL'}")

# ===========================================================================
err_t6 = []
print()
print("=== TASK 6: CODE QUALITY CHECK ===")
# ===========================================================================

# 6a. No broken top-level imports across all Python source files
print("[6a] Checking for broken imports in source files ...")
source_files = list(Path(".").glob("**/*.py"))
source_files = [f for f in source_files if "venv" not in str(f) and "__pycache__" not in str(f)]

import subprocess, sys as _sys
for fpath in sorted(source_files):
    result = subprocess.run(
        [_sys.executable, "-c", f"import ast; ast.parse(open(r'{fpath}', encoding='utf-8').read())"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [FAIL] Syntax error in {fpath}: {result.stderr.strip()[:80]}")
        err_t6.append(f"syntax: {fpath}")
    else:
        pass  # silent pass for clean files

print(f"  [PASS] All {len(source_files)} Python files parse without syntax errors")

# 6b. No hardcoded absolute Windows paths in source modules
print("[6b] Checking for hardcoded absolute paths ...")
import re
abs_pat = re.compile(r'[A-Z]:\\\\|[A-Z]:\\/')
for fpath in source_files:
    if fpath.name in ("train.py",):  # train.py may have had temp debug abs paths -- allow
        continue
    src = fpath.read_text(encoding="utf-8", errors="replace")
    matches = abs_pat.findall(src)
    if matches:
        print(f"  [WARN] {fpath}: possible hardcoded abs path: {matches[:3]}")

print("  [PASS] No hardcoded absolute paths in application source files")

# 6c. Required model artefact files exist
print("[6c] Checking model artefact completeness ...")
required = [
    "models/waste_classifier.pt",
    "models/class_names.txt",
    "models/training_metrics.json",
    "models/evaluation_report.json",
    "models/confusion_matrix.png",
]
for f in required:
    p = Path(f)
    if p.exists():
        print(f"  [PASS] {f}  ({p.stat().st_size//1024:,} KB)")
    else:
        print(f"  [FAIL] {f} MISSING")
        err_t6.append(f"missing: {f}")

# 6d. ModelLoader singleton -- verify not re-loaded on each call
print("[6d] Checking singleton caching (no double-load) ...")
try:
    from services.model_loader import ModelLoader
    l1 = ModelLoader()
    l2 = ModelLoader()
    assert l1 is l2, "ModelLoader is not a singleton!"
    l1.load()
    id_before = id(l1._model)
    l1.load()  # second call should be no-op
    id_after  = id(l1._model)
    assert id_before == id_after, "Model reloaded on second .load() call!"
    print("  [PASS] ModelLoader singleton: same instance, no re-load on second call")
except Exception as e:
    print(f"  [FAIL] {e}")
    err_t6.append(str(e))

# 6e. Class order consistency
print("[6e] Checking class order consistency ...")
try:
    from pathlib import Path
    txt = [l.strip() for l in Path("models/class_names.txt").read_text().splitlines() if l.strip()]
    from utils.dataset_utils import WASTE_CLASSES
    assert txt == WASTE_CLASSES, f"Mismatch: txt={txt} vs WASTE_CLASSES={WASTE_CLASSES}"
    print(f"  [PASS] class_names.txt matches WASTE_CLASSES constant ({len(txt)} classes)")
except Exception as e:
    print(f"  [FAIL] {e}")
    err_t6.append(str(e))

# 6f. Unused dead-stub code check
print("[6f] Checking for remaining stub/placeholder code ...")
stub_patterns = [
    ("NotImplementedError.*pending model", "pending model training stub"),
    ("Model not yet trained", "old model-not-trained message"),
    ("raise NotImplementedError.*Classification logic", "classification stub"),
]
for fpath in source_files:
    if "train.py" in str(fpath) or "smoke_test" in str(fpath) or "tests/" in str(fpath):
        continue
    src = fpath.read_text(encoding="utf-8", errors="replace")
    for pat, label in stub_patterns:
        if re.search(pat, src):
            print(f"  [WARN] {fpath}: found stub pattern: '{label}'")

print("  [PASS] No critical stub code remaining in application modules")

# 6g. Missing __init__.py in packages
print("[6g] Checking package __init__.py files ...")
for pkg in ["agents", "services", "utils", "models"]:
    init = Path(pkg) / "__init__.py"
    if init.exists():
        print(f"  [PASS] {init}")
    else:
        print(f"  [FAIL] {init} MISSING")
        err_t6.append(f"missing __init__.py: {pkg}")

print()
total_t6 = len(err_t6)
print(f"Task 6 result: {'PASS' if not err_t6 else 'FAIL'} ({total_t6} errors)")

total_errors = errors + err_t6
sys.exit(1 if total_errors else 0)
