import streamlit as st
from typing import TypedDict, List
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import json

# --- SETUP YOUR MODEL ---
API_KEY = st.secrets["OPENAI_API_KEY"]

# --- DEFINE STATE SCHEMA ---
class ChatState(TypedDict):
    history: List[dict]
    user_input: str
    response: str

# --- LLM SETUP ---
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=API_KEY)

# --- USER INFO ---
USER_NAME = "David"
USER_BIRTHDAY = datetime(2003, 4, 5)

# --- TOOL FUNCTIONS ---
def add(a: float, b: float) -> str:
    """Add two numbers"""
    result = a + b
    return f"➕ {a} + {b} = **{result}**"

def subtract(a: float, b: float) -> str:
    """Subtract b from a"""
    result = a - b
    return f"➖ {a} - {b} = **{result}**"

def multiply(a: float, b: float) -> str:
    """Multiply two numbers"""
    result = a * b
    return f"✖️ {a} × {b} = **{result}**"

def divide(a: float, b: float) -> str:
    """Divide a by b"""
    if b == 0:
        return "❌ Error: Cannot divide by zero!"
    result = a / b
    return f"➗ {a} ÷ {b} = **{result}**"

def get_birthday() -> str:
    """Get David's birthday and age information"""
    today = datetime.now()
    age = today.year - USER_BIRTHDAY.year
    
    # Adjust age if birthday hasn't occurred this year yet
    if (today.month, today.day) < (USER_BIRTHDAY.month, USER_BIRTHDAY.day):
        age -= 1
    
    # Calculate next birthday
    next_birthday = datetime(today.year, USER_BIRTHDAY.month, USER_BIRTHDAY.day)
    if next_birthday < today:
        next_birthday = datetime(today.year + 1, USER_BIRTHDAY.month, USER_BIRTHDAY.day)
    
    days_until = (next_birthday - today).days
    
    birthday_str = USER_BIRTHDAY.strftime("%B %d, %Y")
    
    return f"""🎂 **{USER_NAME}'s Birthday Information:**
- **Birthday:** {birthday_str}
- **Current Age:** {age} years old
- **Next Birthday:** {next_birthday.strftime("%B %d, %Y")}
- **Days Until Next Birthday:** {days_until} days"""

def extract_numbers(text: str) -> List[float]:
    """Extract numbers from text"""
    import re
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return [float(n) for n in numbers]

# --- TOOL DEFINITIONS FOR LLM ---
tools_definitions = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers together",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "subtract",
            "description": "Subtract the second number from the first number",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number to subtract"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers together",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "divide",
            "description": "Divide the first number by the second number",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "Numerator"},
                    "b": {"type": "number", "description": "Denominator"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_birthday",
            "description": f"Get {USER_NAME}'s birthday information including date, current age, and days until next birthday",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# --- TOOL EXECUTION MAPPER ---
TOOL_MAP = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
    "get_birthday": get_birthday
}

# --- MAIN CHATBOT NODE ---
def chatbot_node(state: ChatState):
    messages = [
        {
            "role": "system", 
            "content": f"You are a helpful calculator assistant for {USER_NAME}. You have access to math tools (add, subtract, multiply, divide) and can provide birthday information. Always use the appropriate tool when the user asks for calculations or birthday info."
        }
    ]
    messages.extend(state["history"])
    messages.append({"role": "user", "content": state["user_input"]})
    
    # Call LLM with tools
    llm_response = model.bind_tools(tools_definitions).invoke(messages)
    
    response_text = ""
    tool_results = []
    
    # Check if LLM wants to use tools
    if hasattr(llm_response, 'tool_calls') and llm_response.tool_calls:
        for tool_call in llm_response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            if tool_name in TOOL_MAP:
                try:
                    result = TOOL_MAP[tool_name](**tool_args)
                    tool_results.append(result)
                except Exception as e:
                    tool_results.append(f"❌ Error executing {tool_name}: {str(e)}")
        
        response_text = "\n\n".join(tool_results)
    else:
        # No tool call, just return the content
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
st.title("🧮 Calculator Chatbot for David")
st.caption("Ask me to add, subtract, multiply, divide numbers, or tell you about your birthday!")

# Example queries
with st.expander("💡 Example queries"):
    st.markdown("""
    - Add 15 and 27
    - What's 100 minus 43?
    - Multiply 8 by 7
    - Divide 144 by 12
    - When is my birthday?
    - How old am I?
    - When's my next birthday?
    """)

if "history" not in st.session_state:
    st.session_state.history = []

# Render past messages
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me anything about math or your birthday...")

if user_input:
    state = {
        "user_input": user_input,
        "history": st.session_state.history,
        "response": "",
    }
    
    with st.spinner("🤔 Thinking..."):
        result = chat_graph.invoke(state)
    
    st.session_state.history = result["history"]

    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(result["response"])

    st.rerun()