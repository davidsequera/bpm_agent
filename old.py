import streamlit as st
from typing import TypedDict, List
import os
import pandas as pd
import subprocess
import time
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI


# --- SETUP YOUR MODEL ---
API_KEY = st.secrets["OPENAI_API_KEY"]

# --- CONFIG ---
BASE_PATH = r"D:\Projects\Mini\AI\BPS\DeclarativeProcessSimulation\data\4.simulation_results"

# --- DEFINE STATE SCHEMA ---
class ChatState(TypedDict):
    history: List[dict]
    user_input: str
    response: str


# --- LLM SETUP ---
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=API_KEY)


# --- TOOL FUNCTIONS ---
def list_files_and_folders(path: str = BASE_PATH) -> str:
    try:
        items = os.listdir(path)
        result = "\n".join(items)
        return f"📂 Contents of `{path}`:\n{result}"
    except Exception as e:
        return f"❌ Error: {e}"


def search_file(keyword: str, path: str = BASE_PATH) -> str:
    matches = []
    for root, _, files in os.walk(path):
        for f in files:
            if keyword.lower() in f.lower():
                matches.append(os.path.join(root, f))
    if matches:
        return "🔍 Found:\n" + "\n".join(matches)
    else:
        return f"No files found containing '{keyword}'."


def show_file_head(filepath: str, n: int = 5) -> str:
    try:
        if filepath.lower().endswith(".csv"):
            df = pd.read_csv(filepath)
            return f"📄 First {n} rows of {os.path.basename(filepath)}:\n{df.head(n).to_markdown(index=False)}"
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = "".join(f.readlines()[:n])
            return f"📄 First {n} lines of {os.path.basename(filepath)}:\n```\n{lines}\n```"
    except Exception as e:
        return f"❌ Error reading file: {e}"


def run_long_subprocess():
    """Runs dg_prediction.py in the deep_generator_2 conda environment with improved error handling."""
    placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = placeholder.text("🚀 Launching simulation in deep_generator_2...")

    # === PATHS ===
    ACTIVATE_BAT = r"C:\ProgramData\miniconda3\Scripts\activate.bat"
    ENV_NAME = "deep_generator_2"
    SCRIPT_PATH = r"D:\Projects\Mini\AI\BPS\DeclarativeProcessSimulation\dg_prediction.py"
    PATH = r"D:\Projects\Mini\AI\BPS\DeclarativeProcessSimulation"

    # Build command with UTF-8 encoding
    cmd = f'cmd /c "set PYTHONIOENCODING=utf-8 && "{ACTIVATE_BAT}" {ENV_NAME} && python "{SCRIPT_PATH}" {PATH}"'

    # Start subprocess
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
    except Exception as e:
        progress_bar.empty()
        placeholder.empty()
        return f"❌ Failed to start subprocess: {str(e)}"

    total_time = 600  # 10 minutes
    update_interval = 1  # seconds

    for elapsed in range(total_time):
        if process.poll() is not None:  # process finished early
            break
        progress_bar.progress((elapsed + 1) / total_time)
        status_text.text(f"⏳ Running simulation... {elapsed + 1}/{total_time} seconds")
        time.sleep(update_interval)

    # Collect output once it finishes
    try:
        stdout, stderr = process.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        progress_bar.empty()
        placeholder.empty()
        return "❌ Simulation timed out and was killed."

    progress_bar.progress(1.0)
    
    # Check return code
    return_code = process.returncode
    
    if return_code != 0:
        # Process failed
        status_text.text(f"❌ Simulation failed with exit code {return_code}")
        
        error_msg = f"**Simulation Failed (Exit Code: {return_code})**\n\n"
        
        if stderr.strip():
            error_msg += "**Error Output:**\n```\n{}\n```\n\n".format(stderr.strip())
        
        if stdout.strip():
            error_msg += "**Standard Output:**\n```\n{}\n```".format(stdout.strip())
        
        return error_msg
    else:
        # Process succeeded
        status_text.text("✅ Simulation completed successfully!")
        
        success_msg = "✅ **Simulation finished successfully!**\n\n"
        
        if stdout.strip():
            success_msg += "**Output:**\n```\n{}\n```\n\n".format(stdout.strip())
        
        if stderr.strip():
            success_msg += "**Warnings/Info:**\n```\n{}\n```".format(stderr.strip())
        
        return success_msg


# --- MAIN CHATBOT NODE ---
def chatbot_node(state: ChatState):
    user_input = state["user_input"].lower()
    response_text = ""

    # --- TOOL ROUTING ---
    if "list files" in user_input or "show folders" in user_input:
        response_text = list_files_and_folders()

    elif "search" in user_input and "file" in user_input:
        parts = user_input.split()
        keyword = parts[-1] if len(parts) > 2 else ""
        response_text = search_file(keyword)

    elif "head" in user_input or "preview" in user_input:
        tokens = user_input.split()
        filepath = next((t for t in tokens if ":" in t or "\\" in t or "/" in t), "")
        response_text = show_file_head(filepath)

    elif "run simulation" in user_input or "start process" in user_input:
        response_text = run_long_subprocess()

    else:
        messages = [{"role": "system", "content": "You are a helpful assistant with file access tools."}]
        messages.extend(state["history"])
        messages.append({"role": "user", "content": state["user_input"]})
        llm_response = model.invoke(messages)
        response_text = llm_response.content

    # Update history
    new_history = state["history"] + [
        {"role": "user", "content": state["user_input"]},
        {"role": "assistant", "content": response_text},
    ]
    return {"response": response_text, "history": new_history}


# --- BUILD LANGGRAPH ---
graph = StateGraph(ChatState)
graph.add_node("chatbot", chatbot_node)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)
chat_graph = graph.compile()


# --- STREAMLIT UI ---
st.title("💬 LangGraph File Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

# Render past messages
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Say something...")

if user_input:
    state = {
        "user_input": user_input,
        "history": st.session_state.history,
        "response": "",
    }
    result = chat_graph.invoke(state)
    st.session_state.history = result["history"]

    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(result["response"])

    st.rerun()