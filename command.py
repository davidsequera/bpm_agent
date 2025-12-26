# === PATHS ===
ACTIVATE_BAT = r"C:\ProgramData\miniconda3\Scripts\activate.bat"
ENV_NAME = "deep_generator_2"
SCRIPT_PATH = r"D:\Projects\Mini\AI\BPS\DeclarativeProcessSimulation\dg_prediction.py"
PATH = r"D:\Projects\Mini\AI\BPS\DeclarativeProcessSimulation"

# Build command with UTF-8 encoding
cmd = f'cmd /c "set PYTHONIOENCODING=utf-8 && "{ACTIVATE_BAT}" {ENV_NAME} && python "{SCRIPT_PATH}" {PATH}"'

print(f"Running command: {cmd}")