# Databricks notebook source
import os
from uuid import uuid4

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import PostgresAPI
from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config
from arxiv_curator.memory import LakebaseMemory

spark = SparkSession.builder.getOrCreate()
cfg = load_config("arxiv_config.yml", get_env(spark))

w = WorkspaceClient()
pg_api = PostgresAPI(w.api_client)

project_id = cfg.lakebase_project_id

scope_name = "arxiv-agent-scope"
os.environ["LAKEBASE_SP_CLIENT_ID"] = dbutils.secrets.get(scope_name, "client_id")
os.environ["LAKEBASE_SP_CLIENT_SECRET"] = dbutils.secrets.get(scope_name, "client_secret")


w = WorkspaceClient()
os.environ["LAKEBASE_SP_HOST"] = w.config.host

# COMMAND ----------

project = pg_api.get_project(name=f"projects/{project_id}")

memory = LakebaseMemory(
    project_id=project_id,
)

# COMMAND ----------

# Create a test session
session_id = f"test-session-{uuid4()}"

# Save some messages
test_messages = [
    {"role": "user", "content": "What are recent papers on transformers?"},
    {"role": "assistant", "content": "Here are some recent papers on transformer architectures..."},
    {"role": "user", "content": "Tell me more about the first one"},
]

memory.save_messages(session_id, test_messages)
logger.info(f"✓ Saved {len(test_messages)} messages to session: {session_id}")

# COMMAND ----------

# Load messages back
loaded_messages = memory.load_messages(session_id)
