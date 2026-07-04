# JobScout AI — Autonomous Job Search Agent

JobScout AI is an agentic AI system that automates the repetitive parts of job
hunting: searching postings, scoring resume fit, identifying skill gaps, and
drafting tailored cover letters — all decided and sequenced by the agent
itself, not by hardcoded logic.

## Why this is agentic (not just a prompt wrapper)

Most "AI features" are a single prompt → single response. JobScout AI instead
runs a **ReAct-style reasoning loop**:

1. The agent is given a goal (find and evaluate matching jobs for a resume).
2. At each step, it **decides which tool to call** — `search_jobs`,
   `analyze_resume_match`, `get_skill_gap`, or `draft_cover_letter`.
3. It **observes the tool's result**.
4. It **decides the next action** based on that result (e.g. it only calls
   `draft_cover_letter` if the match score it computed is 60+).
5. It stops on its own once it has enough information to give a final answer.

None of steps 2–5 are hardcoded — the sequence and choice of tools is
determined by the LLM (Groq's `llama-3.3-70b-versatile`) via native
function-calling, not a fixed if/else pipeline.

## Tech Stack

- **LLM / reasoning:** Groq API, Llama 3.3 70B, native tool-calling
- **Agent loop:** Python (`agent.py`) — plain ReAct loop, no framework
  dependency, so the mechanics are fully visible and auditable
- **UI:** Streamlit — renders the agent's step-by-step reasoning trace live
- **Tools:**
  - `search_jobs(keyword)` — searches job postings (mock data for demo;
    swappable for a real jobs API)
  - `analyze_resume_match(resume, job_description)` — LLM-scored fit (0–100)
  - `get_skill_gap(resume, job_description)` — missing skills list
  - `draft_cover_letter(resume, job_description, company)` — tailored draft

## Setup

\`\`\`bash
git clone https://github.com/CODEBRAKERBOYY/JobScout-AI.git
cd JobScout-AI
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"
streamlit run app.py
\`\`\`

Get a free Groq API key at https://console.groq.com

## Status

Working local prototype (staging). Not yet deployed for external users.