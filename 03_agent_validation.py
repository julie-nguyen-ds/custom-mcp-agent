# Databricks notebook source
# MAGIC %md
# MAGIC # Agent Validation with MLFlow Evaluate
# MAGIC This notebook demonstrates how to validate AI agent workflows using the MLFlow Evaluate framework. The approach enables comprehensive assessment of agent responses against ground truth, leveraging MLFlow to trace the agent's reasoning, create custom metrics and tool usage metrics for overall validation processes.

# COMMAND ----------

# MAGIC %pip install openai databricks-agents==1.4.0 openpyxl databricks-feature-engineering==0.12.1 mlflow==3.3.2
# MAGIC %restart_python

# COMMAND ----------

import pandas as pd

data = {
    "inputs": [
        "Can you summarize the latest news articles?",
        "Can you help identify trending topics from latest news?",
        "Extract insights from the latest news articles.",
        "Provide sentiment analysis on news articles."
    ],
    "targets": [
        "The news agent can quickly condense lengthy articles into concise summaries, saving time and improving information consumption.",
        "By aggregating and analyzing article content, the news agent can detect frequently mentioned topics and highlight emerging trends.",
        "The news agent can extract key facts, summarize main points, identify sentiment, and highlight entities such as people, organizations, and locations.",
        "News agent should be able to analyze the sentiment of articles exactly, and mention whether the news is 'risk-positive' or 'risk-negative'."
    ]
}

eval_data = pd.DataFrame(data)
display(eval_data)

# COMMAND ----------

# MAGIC %md
# MAGIC Formatting the data to match for LLM agent calling.

# COMMAND ----------

import pandas as pd

# Read the Excel sheet using pandas
df = eval_data.rename(columns={"inputs": "inputs", "targets": "expectations"})
df["inputs"] = [{"question": value} for value in df["inputs"]]
df["expectations"] = [
    {
        "expected_response": str(value["expectations"])
    }
    for index, value in df.iterrows()
]
df_validation = df[["inputs", "expectations"]]
display(df_validation)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Create Custom Metrics
# MAGIC We create a custom metrics for partially correct guideline. We can edit the prompt to define what's correct, and what would not be.

# COMMAND ----------

from mlflow.genai.scorers import scorer
from mlflow.genai.judges import meets_guidelines

import json
from typing import Dict, Any

partially_correct = "The response must follow the expected guidelines. If it has the main correct element and does not contradict the expected response, consider it to pass. If it fulfill find any of the expected element, then it does not pass."

@scorer
def partially_correct_response(inputs: Dict[Any, Any], outputs: Dict[Any, Any], expectations: dict[str, Any]):
    feedbacks = []

    response = outputs
    expected_response = expectations["expected_response"]

    feedbacks.append(
        meets_guidelines(
            name="Correctness",
            guidelines=partially_correct,
            context={"response": response, "expected_response": expected_response},
        )
    )

    return feedbacks

# COMMAND ----------

# MAGIC %md
# MAGIC We also create an example metric for correct tool usage 

# COMMAND ----------

from mlflow.genai.scorers import scorer
from mlflow.genai.judges import meets_guidelines

import json
from typing import Dict, Any

right_tool_usage = """The response must use the correct tool for the response. 
                    The expected tool is mentioned in the expectations after the text "Answer expected from:" which mentions the data source of the tool. 
                    Match the closest tools used to the expectation. 
                   """

@scorer
def right_tool_usage_response(inputs: Dict[Any, Any], outputs: Dict[Any, Any], expectations: dict[str, Any]):
    feedbacks = []
    response = outputs
    expected_response = expectations["expected_tool"]

    feedbacks.append(
        meets_guidelines(
            name="Right tool usage",
            guidelines=right_tool_usage,
            context={"response": response, "expected_response": expected_response},
        )
    )

    return feedbacks

# COMMAND ----------

import mlflow.genai
from mlflow.genai.scorers import Guidelines, Safety, RetrievalGroundedness, RelevanceToQuery, Safety, Correctness, RetrievalSufficiency

# Define evaluation scorers
scorers = [
    partially_correct_response,
    RelevanceToQuery(),
    RetrievalSufficiency()
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Prepare the predict function

# COMMAND ----------

from openai import OpenAI
import os
import time

ENDPOINT = "" # eg. https://adb-12345.3.azuredatabricks.net/serving-endpoints/mcp_agent_model/invocations
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="" # eg. https://adb-12345.3.azuredatabricks.net/serving-endpoints
)

@mlflow.trace    
def query(question: str) -> str:
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "input": [{"role": "user", "content": question}]
    }
    response = requests.post(
        ENDPOINT,
        headers=headers,
        json=data
    )
    result = response.json()
    text_result = result["output"][-1]["content"][0]["text"]
    
    return text_result

# COMMAND ----------

import mlflow
import mlflow.genai
import os 
os.environ['MLFLOW_GENAI_EVAL_MAX_WORKERS'] = '2'

results = mlflow.genai.evaluate(
    data=df_validation,
    predict_fn=query,
    scorers=scorers
)