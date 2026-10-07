"""
EcoSort AI – Streamlit Application
====================================
Agentic Waste Segregation & Recycling Assistant

Entry point:  streamlit run app.py

Pages
-----
1. 🏠 Home
2. 🗑️ Waste Image Scanner
3. ♻️ Recycling Recommendations
4. 🚮 Disposal Guidance
5. 🤖 Eco Assistant
6. 📊 Waste Analytics
"""

import streamlit as st
from utils.session_utils import init_session_state

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialise session state
init_session_state()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Recycle001.svg/240px-Recycle001.svg.png",
    width=80,
)
st.sidebar.title("EcoSort AI")
st.sidebar.caption("Agentic Waste Segregation & Recycling Assistant")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "🗑️ Waste Image Scanner",
        "♻️ Recycling Recommendations",
        "🚮 Disposal Guidance",
        "🤖 Eco Assistant",
        "📊 Waste Analytics",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Model status")
from pathlib import Path as _Path
if _Path("models/waste_classifier.pt").exists():
    st.sidebar.success("Model ready (96.22% accuracy)")
else:
    st.sidebar.warning("Model not found")

# ---------------------------------------------------------------------------
# Page: Home
# ---------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("♻️ EcoSort AI")
    st.subheader("Agentic Waste Segregation & Recycling Assistant")

    st.markdown(
        """
        Welcome to **EcoSort AI** – an intelligent waste management assistant that helps you:

        | Feature | Description |
        |---|---|
        | 🗑️ **Waste Image Scanner** | Upload a photo and identify the waste category |
        | ♻️ **Recycling Recommendations** | Get actionable recycling and reuse tips |
        | 🚮 **Disposal Guidance** | Learn the correct bin and disposal method |
        | 🤖 **Eco Assistant** | Ask any question about recycling and waste |
        | 📊 **Waste Analytics** | View dataset and session statistics |

        ---

        ### 🤖 Agentic Architecture

        EcoSort AI uses a **multi-agent architecture** with specialised agents coordinated
        by an orchestrator:

        ```
        OrchestratorAgent
        ├── WasteClassificationAgent   ← identifies waste from image
        ├── RecyclingRecommendationAgent ← provides recycling tips
        ├── DisposalGuidanceAgent      ← explains disposal method
        └── EcoAssistantAgent          ← answers free-text questions
        ```

        ---

        ### ✅ Project Status

        The **EfficientNet-B0** classification model has been trained and is ready.

        | Metric | Value |
        |---|---|
        | Best validation accuracy | **97.08%** |
        | Test accuracy | **96.22%** |
        | Weighted F1-score | **96.23%** |
        | Best epoch | 24 / 30 |
        | Training images | 10,860 |
        | Test images | 2,328 |
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset classes", "12")
    col2.metric("Total images", "15,515")
    col3.metric("Test accuracy", "96.22%")
    col4.metric("Weighted F1", "96.23%")

# ---------------------------------------------------------------------------
# Page: Waste Image Scanner
# ---------------------------------------------------------------------------
elif page == "🗑️ Waste Image Scanner":
    st.title("🗑️ Waste Image Scanner")
    st.caption(
        "Upload a photo of your waste item. "
        "The AI will identify it and provide recycling and disposal guidance."
    )

    uploaded_file = st.file_uploader(
        "Upload a waste image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear photo of the waste item you want to classify.",
    )

    if uploaded_file is not None:
        from utils.image_utils import load_image
        from agents.orchestrator_agent import OrchestratorAgent

        # Load and preview image
        try:
            image = load_image(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the uploaded image: {exc}")
            st.stop()

        col_img, col_result = st.columns([1, 1])
        with col_img:
            st.image(image, caption=uploaded_file.name, use_container_width=True)

        with col_result:
            st.subheader("Analysis")

            analyze_btn = st.button("🔍 Analyze Waste", type="primary", use_container_width=True)

            if analyze_btn:
                with st.spinner("Classifying..."):
                    orchestrator = OrchestratorAgent()
                    result = orchestrator.process_image(image)

                if result.errors:
                    for err in result.errors:
                        st.error(err)
                else:
                    clf = result.classification

                    # -- Top prediction ----------------------------------
                    confidence_pct = clf["confidence"] * 100
                    st.success(
                        f"**Predicted class:** {clf['predicted_class'].upper()}  \n"
                        f"**Confidence:** {confidence_pct:.1f}%"
                    )

                    # -- Confidence bar ----------------------------------
                    st.progress(clf["confidence"])

                    # -- Top-3 predictions -------------------------------
                    st.markdown("**Top 3 predictions:**")
                    for rank, (cls_name, prob) in enumerate(clf["top3"], 1):
                        bar_val = min(int(prob * 100), 100)
                        st.markdown(
                            f"`{rank}.` **{cls_name}** — {prob*100:.1f}%"
                        )

                    st.caption(f"_Inference device: {clf['device']}_")

                    # Record in session history
                    from utils.session_utils import record_classification
                    record_classification(
                        waste_class=clf["predicted_class"],
                        confidence=clf["confidence"],
                        image_name=uploaded_file.name,
                    )
                    st.session_state["model_loaded"] = True

        # -- Recycling & disposal guidance (shown after scan) ------------
        if "classification" in st.session_state.get("last_result", {}):
            pass  # expanded below via expanders

        # Show recycling + disposal in expanders if result exists in session
        last_scan = (
            st.session_state.classification_history[-1]
            if st.session_state.classification_history
            else None
        )

        if last_scan and uploaded_file is not None:
            predicted = last_scan["predicted_class"]
            if analyze_btn or last_scan.get("image") == uploaded_file.name:
                from agents.recycling_recommendation_agent import RecyclingRecommendationAgent
                from agents.disposal_guidance_agent import DisposalGuidanceAgent

                with st.expander("♻️ Recycling Recommendations", expanded=True):
                    rec = RecyclingRecommendationAgent().get_recommendations(predicted)
                    for tip in rec["tips"]:
                        st.markdown(f"- {tip}")
                    if rec["reuse_ideas"]:
                        st.markdown("**Reuse ideas:**")
                        for idea in rec["reuse_ideas"]:
                            st.markdown(f"- {idea}")

                with st.expander("🚮 Disposal Guidance", expanded=True):
                    disp = DisposalGuidanceAgent().get_guidance(predicted)
                    hazard_color = {"High": "🔴", "Low": "🟡", "None": "🟢"}.get(
                        disp["hazard_level"], "⚪"
                    )
                    st.markdown(f"**Bin:** {disp['bin']}")
                    st.markdown(f"**Hazard:** {hazard_color} {disp['hazard_level']}")
                    st.info(disp["instructions"])

# ---------------------------------------------------------------------------
# Page: Recycling Recommendations
# ---------------------------------------------------------------------------
elif page == "♻️ Recycling Recommendations":
    st.title("♻️ Recycling Recommendations")
    st.markdown(
        "Select a waste category to see recycling tips and reuse ideas, "
        "or scan an image on the Waste Image Scanner page for automatic suggestions."
    )

    from agents.recycling_recommendation_agent import RecyclingRecommendationAgent
    from utils.dataset_utils import WASTE_CLASSES

    agent = RecyclingRecommendationAgent()

    selected = st.selectbox("Select a waste category", sorted(WASTE_CLASSES))
    if selected:
        result = agent.get_recommendations(selected)
        st.subheader(f"♻️ {selected.title()}")

        st.markdown("**Recycling Tips:**")
        for tip in result["tips"]:
            st.markdown(f"- {tip}")

        if result["reuse_ideas"]:
            st.markdown("**Reuse Ideas:**")
            for idea in result["reuse_ideas"]:
                st.markdown(f"- {idea}")

# ---------------------------------------------------------------------------
# Page: Disposal Guidance
# ---------------------------------------------------------------------------
elif page == "🚮 Disposal Guidance":
    st.title("🚮 Disposal Guidance")
    st.markdown(
        "Select a waste category to see where and how to dispose of it correctly."
    )

    from agents.disposal_guidance_agent import DisposalGuidanceAgent
    from utils.dataset_utils import WASTE_CLASSES

    agent = DisposalGuidanceAgent()

    selected = st.selectbox("Select a waste category", sorted(WASTE_CLASSES))
    if selected:
        result = agent.get_guidance(selected)

        st.subheader(f"🚮 {selected.title()}")

        hazard_colour = {"High": "🔴", "Low": "🟡", "None": "🟢"}.get(
            result["hazard_level"], "⚪"
        )
        st.markdown(f"**Hazard level:** {hazard_colour} {result['hazard_level']}")
        st.markdown(f"**Correct bin:** {result['bin']}")
        st.markdown(f"{result['colour_hint']}")
        st.info(result["instructions"])

# ---------------------------------------------------------------------------
# Page: Eco Assistant
# ---------------------------------------------------------------------------
elif page == "🤖 Eco Assistant":
    st.title("🤖 Eco Assistant")

    from services.llm_provider import config as _llm_cfg
    from agents.eco_assistant_agent import EcoAssistantAgent

    # -- Status banner ---------------------------------------------------
    if _llm_cfg.is_configured:
        st.caption(
            f"AI assistant powered by **{_llm_cfg.model}** via OpenRouter. "
            "Ask anything about waste, recycling, or disposal."
        )
    else:
        st.warning(
            "**Eco Assistant is not configured.** "
            "Set `ECO_LLM_API_KEY` (and optionally `ECO_LLM_BASE_URL`, `ECO_LLM_MODEL`) "
            "in your `.env` file to enable AI-powered answers.\n\n"
            "See `.env.example` for setup instructions.",
            icon="⚙️",
        )

    # -- Scan context panel (shown when a scan has been performed) -------
    last_scan = (
        st.session_state.classification_history[-1]
        if st.session_state.get("classification_history")
        else None
    )

    scan_context = None
    if last_scan:
        from agents.recycling_recommendation_agent import RecyclingRecommendationAgent
        from agents.disposal_guidance_agent import DisposalGuidanceAgent

        predicted = last_scan["predicted_class"]
        confidence = last_scan.get("confidence", 0.0)
        recycling = RecyclingRecommendationAgent().get_recommendations(predicted)
        disposal  = DisposalGuidanceAgent().get_guidance(predicted)
        scan_context = {
            "waste_class": predicted,
            "confidence":  confidence,
            "recycling":   recycling,
            "disposal":    disposal,
        }

        with st.expander("Current scan context (used by assistant)", expanded=False):
            st.markdown(
                f"**Last scanned item:** `{predicted}` "
                f"— confidence **{confidence * 100:.1f}%**"
            )
            st.caption(
                "The assistant will use this context when you ask questions "
                "like 'Where should I throw this?' or 'Can I recycle it?'."
            )

    # -- Chat history ----------------------------------------------------
    assistant = EcoAssistantAgent()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # -- New user input --------------------------------------------------
    user_input = st.chat_input("Ask about recycling, waste disposal, or eco tips...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            response = assistant.answer(user_input, context=scan_context)

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    # -- Clear chat button -----------------------------------------------
    if st.session_state.chat_history:
        if st.button("Clear chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

# ---------------------------------------------------------------------------
# Page: Waste Analytics
# ---------------------------------------------------------------------------
elif page == "📊 Waste Analytics":
    st.title("📊 Waste Analytics")

    from services.analytics_service import get_dataset_analytics, get_session_analytics
    import pandas as pd

    # Dataset analytics
    st.subheader("📂 Dataset Overview")
    try:
        summary = get_dataset_analytics()
        col1, col2 = st.columns(2)
        col1.metric("Total classes", summary["num_classes"])
        col2.metric("Total images", f"{summary['total_images']:,}")

        df = pd.DataFrame(
            list(summary["class_counts"].items()),
            columns=["Class", "Image Count"],
        ).sort_values("Image Count", ascending=False)

        st.bar_chart(df.set_index("Class"))
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Could not load dataset analytics: {e}")

    st.divider()

    # Session analytics
    st.subheader("🔍 Session Classification History")
    history = st.session_state.classification_history
    if not history:
        st.info("No classifications performed yet in this session.")
    else:
        session_stats = get_session_analytics(history)
        col1, col2, col3 = st.columns(3)
        col1.metric("Scans this session", session_stats["total_scans"])
        col2.metric(
            "Avg confidence",
            f"{session_stats['avg_confidence']:.1%}",
        )
        col3.metric("Unique classes found", len(session_stats["class_counts"]))

        st.dataframe(
            pd.DataFrame(history),
            use_container_width=True,
            hide_index=True,
        )
