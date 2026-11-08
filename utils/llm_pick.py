from langchain_ollama import ChatOllama


def llm_pick(level: str):
    level_lower = level.lower()

    if level_lower == "low":
        temperature = 0.0
    elif level_lower == "medium":
        temperature = 0.5
    elif level_lower == "high":
        temperature = 0.8
    else:
        raise ValueError("Invalid level. Choose from 'low', 'medium', or 'high'.")

    # Initialized with the lightweight, fast llama3.2:3b model
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=temperature,
    )

    return llm
