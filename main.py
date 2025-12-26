import streamlit as st
import subprocess
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import time

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

API_KEY = st.secrets["OPENAI_API_KEY"]

ACTIVATE_BAT = r"C:\ProgramData\miniconda3\Scripts\activate.bat"
CONDA_ENV = "deep_generator_2"
BPS_CLI = r"D:\Projects\Mini\AI\BPS\DeclarativeProcessSimulation\cli.py"
DEFAULT_ROOT = r"D:\Projects\Mini\AI\BPS\DeclarativeProcessSimulation"
    
# ---------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------

class ChatState(TypedDict):
    history: List[dict]
    user_input: str
    response: str

# ---------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------

model = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=API_KEY,
)

# ---------------------------------------------------------------------
# SUBPROCESS CORE
# ---------------------------------------------------------------------

def _base_cmd():
    return (
        f'cmd /c "'
        f'set PYTHONIOENCODING=utf-8 && '
        f'"{ACTIVATE_BAT}" {CONDA_ENV} && '
    )


def run_subprocess(cmd: str, timeout: int = 600) -> str:
    try:
        print(f"Running command: {cmd}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        return "❌ Process timed out."

    if process.returncode != 0:
        return (
            f"❌ Process failed\n\n"
            f"**stderr:**\n```\n{stderr.strip()}\n```\n\n"
            f"**stdout:**\n```\n{stdout.strip()}\n```"
        )

    return f"✅ Success\n\n```\n{stdout.strip()}\n```"

# ---------------------------------------------------------------------
# TOOLS
# ---------------------------------------------------------------------

def list_logs(root: str = DEFAULT_ROOT) -> str:
    """List available logs that can be run"""
    cmd = _base_cmd() + f'python {BPS_CLI} list-logs --root "{root}"' + '"'
    return run_subprocess(cmd, timeout=300)


def run_pipeline(
    log: str,
    root: str = DEFAULT_ROOT,
    rep: int = 1,
    variant: str = "Rules Based Random Choice",
) -> str:
    """Run the full BPS pipeline"""
    cmd = (
        _base_cmd()
        + f'python {BPS_CLI} run '
        f'--root "{root}" '
        f'--log "{log}" '
        f'--rep {rep} '
        f'--variant "{variant}"'
        + '"'
    )
    return run_subprocess(cmd)

# ---------------------------------------------------------------------
# TOOL SCHEMA FOR LLM
# ---------------------------------------------------------------------

tools_definitions = [
    {
        "type": "function",
        "function": {
            "name": "list_logs",
            "description": "List the available event logs that can be used to run the BPS pipeline",
            "parameters": {
                "type": "object",
                "properties": {
                    "root": {
                        "type": "string",
                        "description": "Project root path",
                        "default": DEFAULT_ROOT,
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_pipeline",
            "description": "Run the full BPS pipeline for a given CSV log",
            "parameters": {
                "type": "object",
                "properties": {
                    "log": {
                        "type": "string",
                        "description": "CSV filename of the log to run (log must include the .csv extension)",
                    },
                    "root": {
                        "type": "string",
                        "default": DEFAULT_ROOT,
                    },
                    "rep": {
                        "type": "integer",
                        "default": 1,
                    },
                    "variant": {
                        "type": "string",
                        "default": "Rules Based Random Choice",
                    },
                },
                "required": ["log"],
            },
        },
    },
]

TOOL_MAP = {
    "list_logs": list_logs,
    "run_pipeline": run_pipeline,
}

# ---------------------------------------------------------------------
# CHATBOT NODE
# ---------------------------------------------------------------------

def chatbot_node(state: ChatState):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a BPS pipeline assistant.\n"
                "You can:\n"
                "- list available logs\n"
                "- run the full pipeline for a given log\n\n"
                "Use tools whenever the user asks to list logs or run the pipeline."
            ),
        }
    ]

    messages.extend(state["history"])
    messages.append({"role": "user", "content": state["user_input"]})

    llm_response = model.bind_tools(tools_definitions).invoke(messages)

    response_text = ""

    if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
        results = []
        for tool_call in llm_response.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]

            if name in TOOL_MAP:
                with st.spinner(f"⚙️ Executing {name}..."):
                    try:
                        results.append(TOOL_MAP[name](**args))
                    except Exception as e:
                        results.append(f"❌ Tool error: {e}")

        response_text = "\n\n".join(results)
    else:
        response_text = llm_response.content

    new_history = state["history"] + [
        {"role": "user", "content": state["user_input"]},
        {"role": "assistant", "content": response_text},
    ]

    return {"response": response_text, "history": new_history}

# ---------------------------------------------------------------------
# GRAPH
# ---------------------------------------------------------------------

graph = StateGraph(ChatState)
graph.add_node("chatbot", chatbot_node)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)
chat_graph = graph.compile()

# ---------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------

st.set_page_config(page_title="BPS Chatbot", layout="centered")
st.title("🤖 BPS Pipeline Chatbot")
st.caption("Ask me to list logs or run the BPS pipeline")

if "history" not in st.session_state:
    st.session_state.history = []

# Render history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input(
    "Try: 'list available logs' or 'run pipeline for PurchasingExample.csv'"
)

if user_input:
    state = {
        "user_input": user_input,
        "history": st.session_state.history,
        "response": "",
    }

    result = chat_graph.invoke(state)
    st.session_state.history = result["history"]

    with st.chat_message("assistant"):
        st.markdown(result["response"])

    st.rerun()
