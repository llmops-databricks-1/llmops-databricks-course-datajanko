# Databricks notebook source
"""
Setup Lakebase for Learning Buddy

This notebook provisions the Lakebase project and creates the session_messages
table with its index. It is idempotent and safe to re-run.

Parameters:
  config_path: Path to configuration file (default: learning_buddy_config.yml)
  env: Environment name (dev, acc, prd)
"""

import urllib.parse

import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import (
    PostgresAPI,
    Project,
    ProjectDefaultEndpointSettings,
    ProjectSpec,
)
from google.protobuf.duration_pb2 import Duration
from loguru import logger
from pyspark.sql import SparkSession

from commons.config import load_config

# COMMAND ----------

dbutils.widgets.text("config_path", "learning_buddy_config.yml", "Config File Path")  # noqa F821
dbutils.widgets.text("env", "dev", "Environment")  # noqa F821

config_path = dbutils.widgets.get("config_path")  # noqa F821
env = dbutils.widgets.get("env")  # noqa F821

spark = SparkSession.builder.getOrCreate()
cfg = load_config(config_path, env=env)

logger.info(f"Environment: {env}")
logger.info(f"Lakebase project ID: {cfg.lakebase_project_id}")

# COMMAND ----------

# Create or retrieve the Lakebase project
w = WorkspaceClient()
pg_api = PostgresAPI(w.api_client)

project_id = cfg.lakebase_project_id

try:
    project = pg_api.get_project(name=f"projects/{project_id}")
    logger.info(f"Found existing project: {project_id}")
except Exception:
    logger.info(f"Creating new project: {project_id}")
    project = pg_api.create_project(
        project_id=project_id,
        project=Project(
            spec=ProjectSpec(
                display_name=project_id,
                budget_policy_id=cfg.usage_policy_id,
                default_endpoint_settings=ProjectDefaultEndpointSettings(
                    autoscaling_limit_min_cu=1,
                    autoscaling_limit_max_cu=4,
                    suspend_timeout_duration=Duration(seconds=300),
                ),
            ),
        ),
    ).wait()
    logger.info(f"Created project: {project_id}")

# COMMAND ----------

# Build connection string
default_branch = next(iter(pg_api.list_branches(parent=project.name)))
endpoint = next(iter(pg_api.list_endpoints(parent=default_branch.name)))
host = endpoint.status.hosts.host

user = w.current_user.me()
pg_credential = pg_api.generate_database_credential(endpoint=endpoint.name)
username = urllib.parse.quote_plus(user.user_name)
conn_string = f"postgresql://{username}:{pg_credential.token}@{host}:5432/databricks_postgres?sslmode=require"

logger.info(f"Connecting to Lakebase at: {host}")

# COMMAND ----------

# Create session_messages table and index
with psycopg.connect(conn_string) as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_messages (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            message_data JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_messages_session_id
        ON session_messages(session_id)
    """)
    logger.info("session_messages table and index are ready")

# COMMAND ----------

logger.info("Lakebase setup complete.")
