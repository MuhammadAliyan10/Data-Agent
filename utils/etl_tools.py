import os
import requests
import pandas as pd


class ETLTools:
    def __init__(self):
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

    def extract_load(self, url: str, output_path: str, format: str) -> str:
        output_folder = os.path.join(self.project_root, output_path)

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            filename = os.path.join(output_folder, f"extracted_data.{format}")
            os.makedirs(output_folder, exist_ok=True)

            df = pd.json_normalize(data["results"])
            if format == "csv":
                df.to_csv(filename, index=False)
            elif format == "json":
                df.to_json(filename, orient="records", lines=True)
            elif format == "parquet":
                df.to_parquet(filename, index=False)
            else:
                return f"Unsupported Format: {format}"
            return f"Data stored in the file successfully: {filename}"
        except requests.RequestException as e:
            return f"Extraction Error: {str(e)}"

    def transform_load_context(self, file_path: str):
        # Ensure path resolution is safe whether relative or absolute
        full_path = (
            file_path
            if os.path.isabs(file_path)
            else os.path.join(self.project_root, file_path)
        )

        file_extension = os.path.splitext(full_path)[1].lower()
        if file_extension == ".csv":
            df = pd.read_csv(full_path)
        elif file_extension == ".json":
            df = pd.read_json(full_path, lines=True)
        elif file_extension == ".parquet":
            df = pd.read_parquet(full_path)
        else:
            return "Invalid format"

        return str(df.head(5))

    def execute_code(self, code: str):
        try:
            # Safe local execution scope with pandas pre-imported
            local_vars = {"pd": pd, "os": os}
            exec(code, {}, local_vars)
            return "Code Ran Successfully"
        except Exception as e:
            return f"Execution Error: {str(e)}"
