import psycopg2


class DatabaseUtil:
    def __init__(self, db_config):
        self.db_config = db_config
        try:
            self.connection = psycopg2.connect(**db_config)
        except Exception as e:
            print("Error while connecting to database", e)
            self.connection = None

    def schema_context(self, schema_name):
        cursor = None
        schema_info_context = ""
        try:
            connection = self.connection
            if not connection:
                raise Exception("No active database connection.")

            cursor = connection.cursor()
            schema_info_context = f"Database Schema {schema_name}\n"

            # FIX: Use a proper tuple with a trailing comma (schema_name,)
            cursor.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = %s;",
                (schema_name,),
            )
            tables_list = cursor.fetchall()

            for table in tables_list:
                table_name = table[0]
                schema_info_context += f"\nTable: {table_name}\n"

                # FIX: Removed stray comma after data_type
                cursor.execute(
                    "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = %s AND table_name = %s;",
                    (schema_name, table_name),
                )
                column_list = cursor.fetchall()

                for column in column_list:
                    column_name, data_type = column
                    schema_info_context += (
                        f" Column: {column_name}, Data Type = {data_type}\n"
                    )

                # FIX: Removed '=' sign from LIMIT 5
                try:
                    cursor.execute(f"SELECT * FROM {schema_name}.{table_name} LIMIT 5")
                    sample_data = cursor.fetchall()
                    schema_info_context += " Sample Data:\n"
                    for data in sample_data:
                        schema_info_context += f" {data}\n"
                except Exception as sample_err:
                    schema_info_context += (
                        f" (Could not fetch sample data: {sample_err})\n"
                    )

        except Exception as e:
            print("An error occurred during schema inspection:", e)
            return None
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

        return schema_info_context

    def execute_query(self, query):
        try:
            connection = self.connection
            cursor = connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            connection.commit()
            return str(result)
        except Exception as e:
            print("Error", e)
            return None
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()


obj = DatabaseUtil(
    {
        "host": "localhost",
        "port": 5432,
        "user": "aliyan",
        "password": "itsaliyan2580",
        "dbname": "lab1_news",
    }
)

result = obj.schema_context("public")

if result:
    with open("db_result.txt", "w") as f:
        f.write(result)
    print("Schema context successfully saved to db_result.txt!")
else:
    print("Failed to generate schema context.")
