from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.etl_tools import ETLTools
from utils.llm_pick import llm_pick
from models.schema import ETLAgentSchema


@tool
def extract_load_tool(url: str, output_folder: str, format: str) -> str:
    """Extracts JSON data from a URL and saves it locally in the specified format."""
    etl_tools = ETLTools()
    return etl_tools.extract_load(url, output_folder, format)


@tool
def transform_load_tool(
    user_question: str, input_file_path: str, output_folder: str, output_format: str
) -> str:
    """Transforms data using pandas based on user question and saves the output file."""
    etl_tools = ETLTools()
    context = etl_tools.transform_load_context(input_file_path)
    llm = llm_pick("high")
    prompt = f"""
    You are a Python data analyst who uses pandas to analyze data.
    Provide ONLY the executable pandas code to perform the ETL operation as per the user's question.
    Do not provide any explanation or comments.
    Create the pandas dataframe from the file: {input_file_path}
    Transform and save the data at: {output_folder} with the format: {output_format}
    User Question: {user_question}
    Data Context: {context}
    """
    response = llm.invoke(prompt).content

    # Robust markdown code block cleaner
    cleaned_code = response.strip()
    if "```python" in cleaned_code:
        cleaned_code = cleaned_code.split("```python")[1].split("```")[0].strip()
    elif "```" in cleaned_code:
        cleaned_code = cleaned_code.split("```")[1].split("```")[0].strip()

    results = etl_tools.execute_code(cleaned_code)
    return f"The data is transformed and saved at {output_folder} in {output_format} format with results: {results}"


tools = [extract_load_tool, transform_load_tool]
llm = llm_pick("high")
llm_bind = llm.bind_tools(tools)


def llm_node(state: ETLAgentSchema):
    messages = state.messages
    system_prompt = SystemMessage(
        content=(
            "You are a Python Data Analyst with access to ETL tools. "
            "Analyze the user's request, invoke the correct tool(s) to extract/transform data, "
            "and once completed, inform the user and conclude."
        )
    )

    # Pass system prompt + message history to the bound model so tool-calling works
    response = llm_bind.invoke([system_prompt] + messages)
    state.messages = messages + [response]
    return state


def tool_node(state: ETLAgentSchema):
    tool_calls = state.messages[-1].tool_calls
    tool_results = []
    tools_by_name = {tool.name: tool for tool in tools}

    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])

        tool_results.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )

    state.messages = state.messages + tool_results
    return state


etl_analyst_graph = StateGraph(ETLAgentSchema)

etl_analyst_graph.add_node("llm_node", llm_node)
etl_analyst_graph.add_node("tool_node", tool_node)

etl_analyst_graph.add_edge(START, "llm_node")


def is_tool_call(state: ETLAgentSchema):
    last_message = state.messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return "END"


etl_analyst_graph.add_conditional_edges(
    "llm_node", is_tool_call, {"tool_node": "tool_node", "END": END}
)

# FIXED: Corrected node name from "tool_name" to "tool_node" and added loopback edge
etl_analyst_graph.add_edge("tool_node", "llm_node")

etl_analyst = etl_analyst_graph.compile()

# --- TEST INVOCATION BLOCK ---
if __name__ == "__main__":
    test_state = {
        "messages": [
            HumanMessage(
                content="Fetch pokemon data from https://pokeapi.co/api/v2/pokemon/ and save it as a csv in the data folder."
            )
        ]
    }

    print("Running ETL Agent Test...")
    final_state = etl_analyst.invoke(test_state)
    print("\n=== Final Agent Response ===")
    print(final_state["messages"][-1].content)
