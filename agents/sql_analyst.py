from langgraph.constants import END
from langchain_core.messages import AIMessage, HumanMessage
import os
import sys

# Load environment variables from .env file at the very top
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.schema import AgentSchema, JudgeSchema
from utils.database import DatabaseUtil
from utils.llm_pick import llm_pick
from langgraph.graph import START, StateGraph


def curate_question(state: AgentSchema) -> AgentSchema:
    user_question = state.user_question
    llm_object = llm_pick("low")
    prompt = f"""TASK: Rewrite the user's natural language question into a clean, concise question optimized for a database lookup.
RULES:
- Output ONLY the rewritten question string.
- Do NOT include greetings, analysis, or conversational filler.
- Do NOT wrap it in extra quotes.

User Question: {user_question}"""
    response = llm_object.invoke(prompt).content.strip()
    if len(response) < 1:
        return print("Error")
    state.curated_question = response
    state.messages = state.messages + [HumanMessage(content=f"{response}")]
    return state


def prompt_query_context(state: AgentSchema) -> AgentSchema:
    curate_question = state.curated_question
    conn_details = {
        "host": os.environ.get("DB_HOST"),
        "port": os.environ.get("DB_PORT"),
        "user": os.environ.get("DB_USER"),
        "password": os.environ.get("DB_PASSWORD"),
        "dbname": os.environ.get("DB_NAME"),
    }
    db_object = DatabaseUtil(conn_details)
    schema_info = db_object.schema_context("public")

    prompt = f"""### SYSTEM ROLE
You are an expert PostgreSQL developer. Your sole purpose is to convert natural language into raw, executable PostgreSQL query code.

### DATABASE SCHEMA & SAMPLES
{schema_info}

### USER REQUEST
{curate_question}

### STRICT CONSTRAINTS
1. Generate a valid, read-only PostgreSQL `SELECT` query that accurately answers the user request based on the schema above.
2. Unless the user explicitly asks for a specific number of rows, always append `LIMIT 10` to the query.
3. OUTPUT FORMAT: You must output **ONLY** the SQL query enclosed inside a markdown code block (` ```sql ... ``` `).
4. DO NOT write any introductory text, explanations, or conversational filler outside the code block.
"""
    state.prompt_query_context = prompt

    return state


def generate_sql_query(state: AgentSchema) -> AgentSchema:
    prompt = state.prompt_query_context
    llm = llm_pick("medium")
    raw_response = llm.invoke(prompt).content

    # Robustly extract SQL from markdown blocks or clean it up
    if "```sql" in raw_response:
        parts = raw_response.split("```sql")
        cleaned_sql = parts[1].split("```")[0].strip()
    elif "```" in raw_response:
        parts = raw_response.split("```")
        cleaned_sql = parts[1].strip()
    else:
        # Fallback: if no markdown, just take lines that look like SQL or the raw response
        cleaned_sql = raw_response.strip()

    state.generated_sql_query = cleaned_sql
    return state


def is_safe_sql(state: AgentSchema) -> AgentSchema:
    sql_query = state.generated_sql_query
    llm = llm_pick("medium")
    llm_judge = llm.with_structured_output(JudgeSchema)

    prompt = f"""
  You are a SQL Judge for data security. Your task is to determine whether the SQL query is safe or not. The SQL query should be only used for data retrieval and should not modify the database in any way. Neither the SQL query nor the prompt should contain any SQL command that can modify the database such as INSERT, UPDATE, DROP, DELETE, ALTER, CREATE or any other command that can basically change the structure or the content of the database.
  If the query is safe to run response with 'Yes' and if it not safe to run response with 'No'. Additional also provide the comments explaining your decision.
  Here is the query to evaluate: {sql_query}
  Your response must be according to the following formate: {JudgeSchema}
  """

    response = llm_judge.invoke(prompt).model_dump()
    state.is_safe = response["answer"]
    state.comments = response["comments"]
    return state


def cancel_sql(state: AgentSchema) -> AgentSchema:
    state.final_response = state.comments
    state.messages = state.messages + [AIMessage(content=f"{state.final_response}")]

    return state


def execute_sql(state: AgentSchema) -> AgentSchema:
    sql_query = state.generated_sql_query
    conn_details = {
        "host": os.environ.get("DB_HOST", "localhost"),
        "port": os.environ.get("DB_PORT", "5432"),
        "user": os.environ.get("DB_USER", "aliyan"),
        "password": os.environ.get("DB_PASSWORD", "itsaliyan2580"),
        "dbname": os.environ.get("DB_NAME", "lab1_news"),
    }
    obj = DatabaseUtil(conn_details)
    execution_result = obj.execute_query(sql_query)
    state.sql_query_execution_result = execution_result

    return state


def represent_final_answer(state: AgentSchema) -> AgentSchema:
    execution_result = state.sql_query_execution_result
    curated_question = state.curated_question

    llm = llm_pick("low")

    prompt = f"""ROLE: Concise Data Analyst.
TASK: Answer the user's question directly using the database result.

RULES:
- State the answer directly in plain English (e.g., "There are 3,416 articles in the database.").
- NEVER say "Based on the database execution result" or "According to the query".
- If the result indicates an error or missing table, politely explain that the requested information could not be found.

User Question: {curated_question}
QueryResult: {execution_result}
"""
    response = llm.invoke(prompt).content
    state.final_response = response
    state.messages = state.messages + [AIMessage(content=f"{response}")]

    return state


sql_agent_graph = StateGraph(AgentSchema)

sql_agent_graph.add_node(
    curate_question, name="curate_question", description="Curate Question"
)
sql_agent_graph.add_node(
    prompt_query_context,
    name="prompt_query_context",
    description="Context about the database Schema",
)
sql_agent_graph.add_node(
    generate_sql_query,
    name="generate_sql_query",
    description="SQl query generated by the agent",
)
sql_agent_graph.add_node(
    is_safe_sql,
    name="is_safe_sql",
    description="Check whether the query generated by the Agent is safe to run or not.",
)
sql_agent_graph.add_node(
    cancel_sql,
    name="cancel_sql",
    description="Cancel the SQL query with proper reason if it is not safe to run",
)
sql_agent_graph.add_node(
    execute_sql,
    name="execute_sql",
    description="Execute the SQL query generated by the Agent.",
)
sql_agent_graph.add_node(
    represent_final_answer,
    name="represent_final_answer",
    description="Show the final response ready for the user to read.",
)


sql_agent_graph.add_edge(START, "curate_question")
sql_agent_graph.add_edge("curate_question", "prompt_query_context")
sql_agent_graph.add_edge("prompt_query_context", "generate_sql_query")
sql_agent_graph.add_edge("generate_sql_query", "is_safe_sql")


def is_safe_sql_edge(state: AgentSchema) -> str:
    is_safe = state.is_safe

    if is_safe.lower() == "yes":
        return "execute_sql"
    else:
        return "cancel_sql"


# FIX: Removed duplicate conflicting edge additions and kept conditional router clean
sql_agent_graph.add_conditional_edges(
    "is_safe_sql",
    is_safe_sql_edge,
    {"execute_sql": "execute_sql", "cancel_sql": "cancel_sql"},
)
sql_agent_graph.add_edge("cancel_sql", END)
sql_agent_graph.add_edge("execute_sql", "represent_final_answer")
sql_agent_graph.add_edge("represent_final_answer", END)

sql_analyst = sql_agent_graph.compile()
input_schema = {
    "messages": [],
    "user_question": "Can you tell me about the primary column?",
    "curated_question": "",
    "is_safe": "No",
    "prompt_query_context": "",
    "comments": "",
    "generated_sql_query": "",
    "sql_query_execution_result": "",
    "final_response": "",
}
final_response = sql_analyst.invoke(input_schema)
print("\n=== Final Answer ===")
print(final_response.get("final_response"))
