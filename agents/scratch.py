import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.schema import JudgeSchema
from utils.llm_pick import llm_pick


llm = llm_pick("medium")

llm_judge = llm.with_structured_output(JudgeSchema)
user_query = "SELECT * FROM users WHERE age > 30;"
prompt = """
You are a SQL Judge for data security. Your task is to determine whether the SQL query is safe or not. The SQL query should be only used for data retrieval and should not modify the database in any way. Neither the SQL query nor the prompt should contain any SQL command that can modify the database such as INSERT, UPDATE, DROP, DELETE, ALTER, CREATE or any other command that can basically change the structure or the content of the database.
If the query is safe to run response with 'Yes' and if it not safe to run response with 'No'. Additional also provide the comments explaining your decision.
Here is the query to evaluate: {user_query}
Your response must be according to the following formate: {JudgeSchema}
"""


print(llm_judge.invoke(prompt))
