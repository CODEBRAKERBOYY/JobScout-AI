"""
JobScout AI - Autonomous Job Search Agent
------------------------------------------
Implements a ReAct-style agentic loop using Groq's function-calling API.
The LLM decides which tool to call, when to call it, and when to stop.
Nothing about the sequence of steps is hardcoded - only the tools themselves
and the goal given in the system prompt.
"""

import os
import json
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# MOCK JOB DATABASE
# In a real product this would call a jobs API (LinkedIn, Indeed, Greenhouse, etc.)
# Kept as a static list here so the agent can run without external API keys
# beyond Groq itself.
# ---------------------------------------------------------------------------
MOCK_JOBS = [
    {
        "id": 1,
        "title": "Junior Full Stack Developer",
        "company": "Nimbus Tech",
        "description": "Looking for a junior developer skilled in React, Node.js, "
                        "Express, and MongoDB. Experience with REST APIs and JWT "
                        "authentication is a plus. Should be comfortable with Git "
                        "and basic CI/CD workflows.",
    },
    {
        "id": 2,
        "title": "AI/ML Engineer - Entry Level",
        "company": "Vertex Analytics",
        "description": "Seeking a machine learning engineer with experience in "
                        "PyTorch or TensorFlow, computer vision, and model "
                        "deployment. Familiarity with LLM APIs (OpenAI, Groq) "
                        "and prompt engineering is a strong plus.",
    },
    {
        "id": 3,
        "title": "Backend Engineer (Python)",
        "company": "Cascade Systems",
        "description": "We need a backend engineer comfortable with Flask or "
                        "Django, SQL databases, and building scalable REST APIs. "
                        "Experience with Docker and cloud deployment preferred.",
    },
]


# ---------------------------------------------------------------------------
# TOOLS - the agent chooses which of these to call, in which order.
# ---------------------------------------------------------------------------

def search_jobs(keyword: str):
    """Search the mock job board for postings matching a keyword."""
    keyword = keyword.lower()
    results = [
        job for job in MOCK_JOBS
        if keyword in job["title"].lower() or keyword in job["description"].lower()
    ]
    if not results:
        results = MOCK_JOBS  # fallback so the agent always has something to reason about
    return json.dumps(results)


def analyze_resume_match(resume_text: str, job_description: str):
    """Use the LLM to score how well a resume matches a job description (0-100)."""
    prompt = (
        "You are an ATS resume matching engine. Compare the resume to the job "
        "description and return ONLY a JSON object with keys 'score' (0-100 "
        "integer) and 'reasoning' (one sentence).\n\n"
        f"RESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{job_description}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def get_skill_gap(resume_text: str, job_description: str):
    """Use the LLM to identify skills present in the job description but missing from the resume."""
    prompt = (
        "Compare the resume to the job description. Return ONLY a JSON object "
        "with key 'missing_skills' as a list of short strings.\n\n"
        f"RESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{job_description}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def draft_cover_letter(resume_text: str, job_description: str, company: str):
    """Use the LLM to draft a short, tailored cover letter."""
    prompt = (
        f"Write a concise, natural-sounding 3-paragraph cover letter for a "
        f"candidate applying to {company}, based on this resume and job "
        f"description. Do not use placeholder brackets.\n\n"
        f"RESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{job_description}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return response.choices[0].message.content


TOOL_FUNCTIONS = {
    "search_jobs": search_jobs,
    "analyze_resume_match": analyze_resume_match,
    "get_skill_gap": get_skill_gap,
    "draft_cover_letter": draft_cover_letter,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": "Search for job postings matching a keyword (e.g. 'developer', 'AI').",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_resume_match",
            "description": "Score how well the candidate's resume matches a specific job description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resume_text": {"type": "string"},
                    "job_description": {"type": "string"},
                },
                "required": ["resume_text", "job_description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_skill_gap",
            "description": "Identify skills required by the job but missing from the resume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resume_text": {"type": "string"},
                    "job_description": {"type": "string"},
                },
                "required": ["resume_text", "job_description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_cover_letter",
            "description": "Draft a tailored cover letter. Only call this if the match score is 60 or above.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resume_text": {"type": "string"},
                    "job_description": {"type": "string"},
                    "company": {"type": "string"},
                },
                "required": ["resume_text", "job_description", "company"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are JobScout, an autonomous job-search agent. You are given a "
    "candidate's resume and a job keyword. Your goal is to: "
    "1) search for matching jobs, "
    "2) pick the single best-fitting job posting, "
    "3) analyze the resume match score for that job, "
    "4) identify the skill gap, and "
    "5) only if the match score is 60 or higher, draft a tailored cover letter. "
    "If the score is below 60, do NOT draft a cover letter - instead explain "
    "why the fit is weak. Decide which tool to call and in what order based on "
    "the results you observe at each step. When you are finished, give a final "
    "summary to the user. Think step by step and only call one tool at a time."
)


def run_agent(resume_text: str, keyword: str, trace_callback=None):
    """
    Runs the agentic loop. trace_callback(step_dict) is called after every
    step so a UI (e.g. Streamlit) can render the agent's reasoning live.
    Returns the final text response from the agent.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Here is my resume:\n{resume_text}\n\n"
                       f"Find me jobs related to: {keyword}",
        },
    ]

    max_steps = 8
    for step in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0,
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            # Agent decided it's done - this is the final answer.
            if trace_callback:
                trace_callback({"type": "final", "content": msg.content})
            return msg.content

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            if trace_callback:
                trace_callback({
                    "type": "tool_call",
                    "step": step + 1,
                    "tool": fn_name,
                    "args": fn_args,
                })

            fn = TOOL_FUNCTIONS.get(fn_name)
            result = fn(**fn_args) if fn else f"Error: unknown tool {fn_name}"

            if trace_callback:
                trace_callback({
                    "type": "tool_result",
                    "step": step + 1,
                    "tool": fn_name,
                    "result": result,
                })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": fn_name,
                "content": str(result),
            })

    return "Agent stopped after reaching max steps without a final answer."