from langchain_core.messages import AIMessage
import os
import sys
from langchain_core.messages import HumanMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.schema import AgentSchema, JudgeSchema
from utils.database import DatabaseUtil
from utils.llm_pick import llm_pick


def curate_question(state: AgentSchema) -> AgentSchema:
    user_question = state.user_question
    llm_object = llm_pick("low")
    response = llm_object.invoke(
        f"I want you to analyze the following user question, understand it and then curate it. : {user_question}"
    ).content
    if len(response) < 1:
        return print("Error")
    state.curated_question = response
    state.messages = state.messages + [HumanMessage(content=f"{response}")]
    return state


def prompt_query_context(state: AgentSchema) -> AgentSchema:
    curate_question = state.curated_question
    conn_details = {
        "host": os.environ["host"],
        "port": os.environ["port"],
        "user": os.environ["user"],
        "password": os.environ["password"],
        "dbname": os.environ["dbname"],
    }
    db_object = DatabaseUtil(conn_details)
    schema_info = db_object.schema_context("public")

    prompt = f"""You are an SQL analyst agent. Your task is to convert the user's natural language query into Postgres SQL query that can executed into the database.
    You are provided with the user's original query and the schema details of the database, including tables name , columns name, data types and sample data for each table so you can understand the structure of the database and generate an accurate SQL query.
    Unless user explicitly ask for specific number of rows, always limit the output the 10 rows.
    Note - Just generate the SQL query without any explanation or additional text because this query will be executed directly on the database. So the output should be SQL ready to be executed without any modifications.

    User original Query : {curate_question}
    Database schema details: {schema_info}
    """
    state.prompt_query_context = prompt

    return state


def generate_sql_query(state: AgentSchema) -> AgentSchema:
    prompt = state.prompt_query_context
    llm = llm_pick("medium")
    generated_sql_query = llm.invoke(prompt).content
    state.generated_sql_query = generated_sql_query
    return state


def is_safe_sql(state: AgentSchema) -> AgentSchema:
    sql_query = state.generated_sql_query
    llm = llm_pick("medium")
    llm_judge = llm.with_structured_output(JudgeSchema)

    prompt = """
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
        "host": os.environ["host"],
        "port": os.environ["port"],
        "user": os.environ["user"],
        "password": os.environ["password"],
        "dbname": os.environ["dbname"],
    }
    obj = DatabaseUtil(conn_details)
    execution_result = obj.execute_sql(sql_query)
    state.sql_query_execution_result = execution_result

    return state


def representation_node(state: AgentSchema) -> AgentSchema:
    execution_result = state.sql_query_execution_result
    curated_question = state.curated_question

    llm = llm_pick("low")

    prompt = f"""
You are an SQL analyst agent. Your task is to provide the final answer to the user based on the execution result of the SQL query and the user original question. The final answer should be concise, clear and directly address the user's query. Avoid including any SQL code or technical details in the final answer. The final answer should be in the user-friendly formate that is easy to understand.\n
Here is the execution result: {execution_result}
Here is the user question: {curated_question}
  """
    response = llm.invoke(prompt).content
    state.final_response = response
    state.messages = state.messages + [AIMessage(content=f"{response}")]

    return state
