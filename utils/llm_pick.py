from langchain_ollama import ChatOllama


def llm_pick(level: str):
    level_lower = level.lower()

    if level_lower == "low":
        temperature = 0.0
        enable_think = False
    elif level_lower == "medium":
        temperature = 0.5
        enable_think = False
    elif level_lower == "high":
        temperature = 0.8
        enable_think = True  # Reasoning enabled only for high mode
    else:
        raise ValueError("Invalid level. Choose from 'low', 'medium', or 'high'.")

    # Pass the 'think' parameter via model_kwargs
    llm = ChatOllama(
        model="qwen3.5:4b",
        temperature=temperature,
        model_kwargs={"think": enable_think},
    )

    return llm


# Test low mode (fast, no reasoning)
llm_low = llm_pick("low")
print("--- Low Mode ---")
print(llm_low.invoke("What is 2 + 2?").content)

# Test high mode (includes reasoning process)
llm_high = llm_pick("high")
print("\n--- High Mode ---")
print(llm_high.invoke("Explain why 9.11 is greater than 9.8 mathematically.").content)
