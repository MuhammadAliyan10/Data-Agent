import os
import sys
from langchain_core.messages import HumanMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.schema import AgentSchema
from utils.database import DatabaseUtil
from utils.llm_pick import llm_pick


def curate_question(state: AgentSchema) -> AgentSchema:
    user_question = state.user_question
    llm_object = llm_pick("low")
    response = llm_object.invoke(
        f"I want you to analyze the following user question, understand it and then curate it. : {user_question}"
    )
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

    llm = llm_pick("medium")
    generated_sql_query = llm.invoke(prompt)

    state.generated_sql_query = generated_sql_query

    return state
