# EcoSort AI
## Agentic Waste Segregation & Recycling Assistant

EcoSort AI is an AI-powered waste management assistant that classifies household
waste from uploaded images and provides recycling and disposal guidance.

---

## 📂 Project Structure

```
EcoSort-AI/
│
├── app.py                          # Streamlit entry point
│
├── agents/                         # Specialised AI agents
│   ├── orchestrator_agent.py       # Coordinates all sub-agents
│   ├── waste_classification_agent.py
│   ├── recycling_recommendation_agent.py
│   ├── disposal_guidance_agent.py
│   └── eco_assistant_agent.py
│
├── services/                       # Business-logic services
│   ├── model_loader.py             # Loads the trained ML model
│   └── analytics_service.py        # Dataset & session statistics
│
├── models/                         # Trained model artefacts (gitignored)
│   └── waste_classifier.pt         # ← produced during training (Step 2)
│
├── utils/                          # Utility helpers
│   ├── dataset_utils.py
│   ├── image_utils.py
│   └── session_utils.py
│
├── data/
│   └── garbage_classification/     # Original dataset (DO NOT MODIFY)
│       ├── battery/       (945 images)
│       ├── biological/    (985 images)
│       ├── brown-glass/   (607 images)
│       ├── cardboard/     (891 images)
│       ├── clothes/       (5325 images)
│       ├── green-glass/   (629 images)
│       ├── metal/         (769 images)
│       ├── paper/         (1050 images)
│       ├── plastic/       (865 images)
│       ├── shoes/         (1977 images)
│       ├── trash/         (697 images)
│       └── white-glass/   (775 images)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the application
```bash
streamlit run app.py
```

---

## 🤖 Agentic Architecture

```
OrchestratorAgent
├── WasteClassificationAgent    ← classifies image via transfer-learning model
├── RecyclingRecommendationAgent ← provides recycling & reuse tips
├── DisposalGuidanceAgent       ← explains correct disposal method & bin
└── EcoAssistantAgent           ← answers free-text eco questions (LLM)
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Garbage Classification Dataset |
| Location | `data/garbage_classification/` |
| Classes | 12 |
| Total images | 15,515 |
| Format | JPEG |
| Corrupted files | 0 |

**Class distribution:**

| Class | Images |
|---|---|
| clothes | 5,325 |
| shoes | 1,977 |
| paper | 1,050 |
| biological | 985 |
| battery | 945 |
| cardboard | 891 |
| plastic | 865 |
| white-glass | 775 |
| metal | 769 |
| trash | 697 |
| green-glass | 629 |
| brown-glass | 607 |

---

## 🗺️ Development Roadmap

| Step | Task | Status |
|---|---|---|
| 1 | Dataset inspection & project scaffold | ✅ Done |
| 2 | Transfer-learning model training (EfficientNet / MobileNetV3) | ✅ Done |
| 3 | Integrate trained model into WasteClassificationAgent | ✅ Done |
| 4 | LLM integration for EcoAssistantAgent | ✅ Done |
| 5 | Full pipeline testing & UI polish | ✅ Done |

---

## ⚠️ Dataset Rules

- Do **not** rename, delete, or modify any original dataset files.
- Training/validation/test splitting is handled programmatically in Step 2.
- The `data/` directory is excluded from Git (see `.gitignore`).

---

## 📄 License

This project is for educational purposes.
