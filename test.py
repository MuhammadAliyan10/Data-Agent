from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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",
)

# Use streaming to prevent 504 gateway timeouts on heavy models
stream = client.chat.completions.create(
    model="deepseek-ai/deepseek-v4.1-flash",
    messages=[{"role": "user", "content": "Which number is larger, 9.11 or 9.8?"}],
    temperature=1,
    top_p=1,
    max_tokens=4096,
    stream=True,  # <--- MUST BE TRUE
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
