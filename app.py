import streamlit as st
from agent import run_agent

st.set_page_config(page_title="JobScout AI", page_icon="🧭", layout="centered")

st.title(" JobScout AI")
st.caption("An autonomous job-search agent. It decides on its own which tools to call, "
           "in what order, and when to stop — watch its reasoning below.")

with st.sidebar:
    st.header("About")
    st.write(
        "JobScout AI uses the ReAct (Reason + Act) agentic pattern on top of "
        "Groq's Llama 3.3 function-calling API. Given a resume and a job "
        "keyword, the agent autonomously searches jobs, scores resume fit, "
        "finds skill gaps, and — only if the match is strong — drafts a "
        "cover letter. None of that sequence is hardcoded; the model decides."
    )
    st.markdown("**Tools available to the agent:**")
    st.markdown("- `search_jobs`\n- `analyze_resume_match`\n- `get_skill_gap`\n- `draft_cover_letter`")

resume_text = st.text_area(
    "Paste your resume text",
    height=200,
    placeholder="Paste resume content here...",
)
keyword = st.text_input("Job keyword to search for", placeholder="e.g. developer, AI, backend")

run = st.button(" Run Agent", type="primary")

if run:
    if not resume_text.strip() or not keyword.strip():
        st.warning("Please provide both a resume and a job keyword.")
    else:
        st.subheader("Agent Reasoning Trace")
        trace_container = st.container()
        steps_log = []

        def render_trace(step):
            steps_log.append(step)
            with trace_container:
                if step["type"] == "tool_call":
                    st.markdown(f"**Step {step['step']} — Agent decides to call `{step['tool']}`**")
                    st.json(step["args"])
                elif step["type"] == "tool_result":
                    with st.expander(f"Result from `{step['tool']}`"):
                        st.code(step["result"])
                elif step["type"] == "final":
                    st.markdown("**✅ Agent finished — final summary:**")

        with st.spinner("Agent is thinking..."):
            final_answer = run_agent(resume_text, keyword, trace_callback=render_trace)

        st.subheader("Final Result")
        st.write(final_answer)