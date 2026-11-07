from langchain_ollama import ChatOllama


def llm_pick(level: str):
    if level.lower() == "low":
        temperature = 0.0
    elif level.lower() == "medium":
        temperature = 0.5
    elif level.lower() == "high":
        temperature = 0.8
    else:
        raise ValueError("Invalid level")

    # Point directly to Qwen 3.5 4B running locally
    llm = ChatOllama(model="qwen3.5:4b", temperature=temperature)

    return llm
