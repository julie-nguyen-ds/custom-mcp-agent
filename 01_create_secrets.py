# Databricks notebook source
# MAGIC %md
# MAGIC # Create Secrets
# MAGIC This notebook is to create secrets for secure authentication to the various tools we'll be using (eg. MCP server, or other API endpoint within Databricks), to be used for deployment.
# MAGIC
# MAGIC Documentations:
# MAGIC - You can create your personal token by following this documentation: [Create Token](https://docs.databricks.com/aws/en/dev-tools/auth/pat)
# MAGIC - You can create your Databricks service principal by following this documentation: [Create Service Principal](https://docs.databricks.com/aws/en/admin/users-groups/manage-service-principals)
# MAGIC
# MAGIC

# COMMAND ----------

import requests

# Replace with your actual Databricks values.
DATABRICKS_HOST = "" # eg. https://adb-35643334286923.3.azuredatabricks.net
TOKEN = "" # fill in your Databricks token here 

# Create your Databricks Service Principal for MCP authentication and fill the info here
SP_SECRET = ""
SP_CLIENT_ID = ""

# Create scope
scope_name = "mcp-scope"
create_scope_url = f"{DATABRICKS_HOST}/api/2.0/secrets/scopes/create"
headers = {"Authorization": f"Bearer {TOKEN}"}
data_scope = {"scope": scope_name, "initial_manage_principal": "users"}  # "users" grants everyone manage access (optional)

resp = requests.post(create_scope_url, headers=headers, json=data_scope)
print(resp.status_code, resp.text)  # 200 + {} = success

# Add secrets to the scope
add_secret_url = f"{DATABRICKS_HOST}/api/2.0/secrets/put"

for key, value in [("DATABRICKS_HOST", DATABRICKS_HOST), 
                   ("DATABRICKS_CLIENT_ID", SP_CLIENT_ID), 
                   ("DATABRICKS_CLIENT_SECRET", SP_SECRET), 
                   ("DATABRICKS_TOKEN", TOKEN)]:
    data_secret = {"scope": scope_name, "key": key, "string_value": value}
    resp = requests.post(add_secret_url, headers=headers, json=data_secret)
    print(resp.status_code, resp.text)  # 200 + {} = success