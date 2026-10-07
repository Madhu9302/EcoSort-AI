"""
Task 2 - Waste Scanner: 7-class inference test
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from agents.waste_classification_agent import WasteClassificationAgent
from agents.recycling_recommendation_agent import RecyclingRecommendationAgent
from agents.disposal_guidance_agent import DisposalGuidanceAgent

print("=== TASK 2: WASTE SCANNER - 7 CLASS INFERENCE ===")

agent = WasteClassificationAgent()
agent.ensure_loaded()

test_classes = [
    "plastic",
    "cardboard",
    "paper",
    "metal",
    "brown-glass",
    "biological",
    "battery",
]

results = []
errors = []

for cls in test_classes:
    cls_dir = Path("data/garbage_classification") / cls
    imgs = sorted(cls_dir.glob("*.jpg"))
    assert imgs, f"No images found in {cls_dir}"
    img_path = imgs[0]

    try:
        r = agent.classify_from_path(img_path)
        correct = r["predicted_class"] == cls
        marker = "[PASS]" if correct else "[WARN]"
        results.append({
            "class": cls, "image": img_path.name,
            "predicted": r["predicted_class"],
            "confidence": r["confidence"],
            "correct": correct, "top3": r["top3"],
            "device": r["device"],
        })
        top3_str = ", ".join(f"{c}({p*100:.0f}%)" for c, p in r["top3"])
        print(f"{marker} {cls:<15} -> {r['predicted_class']:<15} {r['confidence']*100:.1f}%  top3=[{top3_str}]")

        # Verify recycling + disposal chain
        rec  = RecyclingRecommendationAgent().get_recommendations(r["predicted_class"])
        disp = DisposalGuidanceAgent().get_guidance(r["predicted_class"])
        assert rec["tips"],  f"No recycling tips for {r['predicted_class']}"
        assert disp["bin"],  f"No disposal bin for {r['predicted_class']}"
        print(f"       recycling: {rec['tips'][0][:60]}...")
        print(f"       disposal : {disp['bin']}")

    except Exception as e:
        print(f"[FAIL] {cls}: {e}")
        errors.append(f"{cls}: {e}")

correct_count = sum(1 for r in results if r.get("correct"))
print()
print(f"Images tested     : {len(test_classes)}")
print(f"Correct (top-1)   : {correct_count}/{len(test_classes)}")
print(f"Sample accuracy   : {correct_count/len(test_classes)*100:.0f}%")
print(f"Inference device  : {results[0]['device'] if results else 'N/A'}")
print()
print(f"Task 2 result: {'PASS' if not errors else 'FAIL'} ({len(errors)} errors)")
sys.exit(1 if errors else 0)
