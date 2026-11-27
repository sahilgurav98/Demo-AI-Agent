import json
import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from agent_setup import GPT_OSS_MODELS, build_classroom_agent
# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Agentic AI Explorer | Class 12 Demo",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
    .hero-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem 2rem;
        border-radius: 14px;
        border: 1px solid #334155;
        color: #f8fafc;
        margin-bottom: 1.2rem;
    }
    .hero-box h1 {
        margin: 0;
        font-size: 1.9rem;
        color: #38bdf8;
    }
    .hero-box p {
        margin: 0.4rem 0 0 0;
        color: #cbd5e1;
        font-size: 1rem;
    }
    .tool-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        background-color: #1e3a8a;
        color: #dbeafe;
        border: 1px solid #3b82f6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 2. LOAD API KEYS (st.secrets -> os.environ)
# ---------------------------------------------------------
def load_api_keys():
    keys = ["GROQ_API_KEY", "TAVILY_API_KEY", "OPENWEATHER_API_KEY"]
    for k in keys:
        if k in st.secrets and st.secrets[k]:
            os.environ[k] = st.secrets[k]


load_api_keys()


# ---------------------------------------------------------
# 3. SIDEBAR: CONTROLS & CLASSROOM GUIDE
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Agent Control Panel")

    model_choice = st.selectbox(
        "Groq GPT-OSS Model",
        options=list(GPT_OSS_MODELS),
        index=0,
        help="Select between GPT-OSS 120B (higher reasoning) and GPT-OSS 20B (faster speed).",
    ),


    st.divider()
    st.subheader("🔌 Connected Tools Status")

    groq_ok = bool(os.getenv("GROQ_API_KEY"))
    tavily_ok = bool(os.getenv("TAVILY_API_KEY"))
    weather_ok = bool(os.getenv("OPENWEATHER_API_KEY"))

    st.write(f"{'✅' if groq_ok else '❌'} **Groq LLM Engine**")
    st.write(f"{'✅' if tavily_ok else '❌'} **Tavily Web Search**")
    st.write(f"{'✅' if weather_ok else '❌'} **OpenWeather Live API**")
    st.write("✅ **Python AST Calculator**")
    st.write("✅ **Wikipedia Encyclopedia**")

    if not (groq_ok and tavily_ok and weather_ok):
        st.warning("Add missing API keys in `.streamlit/secrets.toml` or Streamlit Cloud Secrets.")

    st.divider()
    with st.expander("📘 How an AI Agent Works", expanded=True):
        st.markdown(
            """
            1. **Perceive:** Reads your question.
            2. **Reason:** Decides *if* a tool is needed.
            3. **Act:** Calls external APIs (Weather, Web, Math, Wiki).
            4. **Observe:** Reads the tool's output.
            5. **Respond:** Combines facts into a final answer.
            """
        )

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------
# 4. HERO HEADER & SAMPLE CLASSROOM PROMPTS
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-box">
        <h1>🤖 Agentic AI Live Classroom Lab</h1>
        <p>Watch an LLM think, pick real-world tools, fetch live data, and solve multi-step tasks autonomously.</p>
        <div style="margin-top: 0.7rem;">
            <span class="tool-badge">🌦️ OpenWeather API</span>
            <span class="tool-badge">🌐 Tavily Web Search</span>
            <span class="tool-badge">🧮 Precision Math</span>
            <span class="tool-badge">📚 Wikipedia</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("💡 **Click any classroom demo prompt below to test the agent immediately:**")

col1, col2, col3, col4 = st.columns(4)
sample_prompts = {
    "🌦️ Live Weather": "What is the current weather and humidity in Mumbai and New Delhi?",
    "🌐 Latest Space News": "Search the web for the latest ISRO space mission updates.",
    "🧮 Math + Weather": "Find the current temperature in Tokyo in Celsius, then calculate (temp * 9/5) + 32.",
    "📚 Multi-Tool Research": "Explain Newton's Second Law using Wikipedia and calculate the force for mass = 12.5 kg and acceleration = 9.8 m/s^2.",
}

selected_sample = None
with col1:
    if st.button("🌦️ Live Weather", use_container_width=True):
        selected_sample = sample_prompts["🌦️ Live Weather"]
with col2:
    if st.button("🌐 Latest Space News", use_container_width=True):
        selected_sample = sample_prompts["🌐 Latest Space News"]
with col3:
    if st.button("🧮 Math + Weather", use_container_width=True):
        selected_sample = sample_prompts["🧮 Math + Weather"]
with col4:
    if st.button("📚 Multi-Tool Research", use_container_width=True):
        selected_sample = sample_prompts["📚 Multi-Tool Research"]


# ---------------------------------------------------------
# 5. CHAT HISTORY & AGENT EXECUTION WITH LIVE TRACE
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("steps"):
            with st.expander("🔍 View Agent's Step-by-Step Thinking & Tool Calls", expanded=False):
                for step in msg["steps"]:
                    st.markdown(step)
        st.markdown(msg["content"])

# Handle user input (either from button or chat box)
user_query = st.chat_input("Ask the AI Agent anything (e.g., weather, live news, math, science)...")
if selected_sample:
    user_query = selected_sample

if user_query:
    if not groq_ok:
        st.error("GROQ_API_KEY is missing. Please configure it to run the agent.")
        st.stop()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Run the Agent and stream its step-by-step reasoning
    with st.chat_message("assistant"):
        trace_logs = []
        final_answer = ""

        with st.status("🧠 Agent is thinking and selecting tools...", expanded=True) as status:
            try:
                agent = build_classroom_agent(model_name=model_choice)

                # Convert session history into LangChain messages
                lc_messages = []
                for m in st.session_state.messages:
                    if m["role"] == "user":
                        lc_messages.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant":
                        lc_messages.append(AIMessage(content=m["content"]))

                step_num = 1
                # Stream graph updates node-by-node so students see the agent loop live
                for event in agent.stream(
                    {"messages": lc_messages},
                    config={"recursion_limit": 15},
                    stream_mode="updates",
                ):
                    for node_name, node_output in event.items():
                        new_msgs = node_output.get("messages", [])
                        for message in new_msgs:
                            # 1. Model decided to call one or more tools
                            if isinstance(message, AIMessage) and message.tool_calls:
                                for tc in message.tool_calls:
                                    tool_name = tc["name"]
                                    tool_args = json.dumps(tc["args"])
                                    log_entry = (
                                        f"**Step {step_num} — 🛠️ Decision (Tool Call):** "
                                        f"Calling `{tool_name}` with arguments `{tool_args}`"
                                    )
                                    trace_logs.append(log_entry)
                                    st.markdown(log_entry)
                                    step_num += 1

                            # 2. Tool returned an observation
                            elif isinstance(message, ToolMessage):
                                preview = str(message.content)
                                if len(preview) > 350:
                                    preview = preview[:350] + "..."
                                log_entry = (
                                    f"**Step {step_num} — 📥 Observation (`{message.name}`):**\n"
                                    f"> {preview}"
                                )
                                trace_logs.append(log_entry)
                                st.markdown(log_entry)
                                step_num += 1

                            # 3. Model produced final text answer
                            elif isinstance(message, AIMessage) and message.content:
                                final_answer = message.content

                if not trace_logs:
                    no_tool_msg = "**Step 1 — 💡 Direct Knowledge:** No external tool needed; answered directly."
                    trace_logs.append(no_tool_msg)
                    st.markdown(no_tool_msg)

                status.update(
                    label=f"✅ Task Complete! ({len(trace_logs)} agent step(s) executed)",
                    state="complete",
                    expanded=True,
                )

            except Exception as e:
                status.update(label="❌ Agent encountered an error", state="error")
                final_answer = f"Sorry, an error occurred while running the agent: `{str(e)}`"

        st.markdown(final_answer)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": final_answer,
                "steps": trace_logs,
            }
        )