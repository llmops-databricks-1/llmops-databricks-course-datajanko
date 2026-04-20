# Databricks notebook source
# MAGIC %md
# MAGIC # 4.5 Learning Buddy Agent — Production
# MAGIC
# MAGIC This notebook brings the Learning Buddy to production level by adding:
# MAGIC
# MAGIC | Feature | 3.7 notebook | This notebook |
# MAGIC |---------|-------------|---------------|
# MAGIC | MLflow tracing | ✗ | ✓ Full span tree (AGENT → CHAIN → LLM / TOOL / RETRIEVER) |
# MAGIC | Session memory | ✗ | ✓ Lakebase (PostgreSQL) — multi-turn conversations |
# MAGIC | Evaluation suite | ✗ | ✓ 5 scorers (3 Guidelines + 2 code) |
# MAGIC | MLflow model registration | ✗ | ✓ Unity Catalog with resources |
# MAGIC
# MAGIC **Supported courses:**
# MAGIC - `mit_18_100a` — Real Analysis (MIT 18.100A, English)
# MAGIC - `bielefeld_a1` — Analysis I (Universität Bielefeld, German)

# COMMAND ----------

# Cell: Setup

import os
import random
from datetime import datetime
from uuid import uuid4

import mlflow
from dotenv import load_dotenv
from loguru import logger
from mlflow.types.responses import ResponsesAgentRequest
from pyspark.sql import SparkSession

from commons.config import get_env
from learning_buddy.config import LearningBuddyProjectConfig

# Local (non-Databricks) setup: load .env and set MLflow tracking URI
if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = LearningBuddyProjectConfig.load("learning_buddy_config.yml", env)

mlflow.set_experiment(cfg.experiment_name)

logger.info(f"✓ Environment: {env}")
logger.info(f"✓ Catalog: {cfg.catalog}.{cfg.schema}")
logger.info(f"✓ LLM endpoint: {cfg.llm_endpoint}")
logger.info(f"✓ Experiment: {cfg.experiment_name}")

# COMMAND ----------

# Cell: Agent initialisation

from learning_buddy.agent import LearningBuddyAgent

agent = LearningBuddyAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=cfg.system_prompt,
    catalog=cfg.catalog,
    schema=cfg.schema,
    vector_search_endpoint=cfg.vector_search_endpoint,
    embedding_endpoint=cfg.embedding_endpoint,
    usage_policy_id=cfg.usage_policy_id,
    lakebase_project_id=cfg.lakebase_project_id,
)

logger.info("✓ LearningBuddyAgent initialised")
logger.info(f"  Tools: {list(agent._tools_dict.keys())}")
logger.info(f"  Memory: {'enabled' if agent.memory else 'disabled'}")
logger.info(f"  VS index: {agent._vs_manager.index_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Architecture Overview
# MAGIC
# MAGIC ```
# MAGIC User Request
# MAGIC     ↓
# MAGIC ┌──────────────────────────────────────────────────────┐
# MAGIC │  @mlflow.trace(AGENT)  predict_stream()              │
# MAGIC │    │                                                  │
# MAGIC │    ├─ @mlflow.trace(RETRIEVER)  load_memory()        │
# MAGIC │    │  → Lakebase: load previous session messages     │
# MAGIC │    │                                                  │
# MAGIC │    ├─ @mlflow.trace(CHAIN)  call_and_run_tools()     │
# MAGIC │    │    ├─ mlflow.start_span(LLM)  call_llm()        │
# MAGIC │    │    │  → streaming OpenAI call                   │
# MAGIC │    │    │                                             │
# MAGIC │    │    ├─ @mlflow.trace(TOOL)  execute_tool()       │
# MAGIC │    │    │  → get_problems_by_week()                  │
# MAGIC │    │    │    or search_lectures()                    │
# MAGIC │    │    │    or search_homework()                    │
# MAGIC │    │    │  each calls LearningBuddyVectorSearchManager│
# MAGIC │    │    │                                             │
# MAGIC │    │    └─ Loop until model stops (max 10 iterations) │
# MAGIC │    │                                                  │
# MAGIC │    └─ @mlflow.trace(CHAIN)  save_memory()            │
# MAGIC │       → Lakebase: persist new messages               │
# MAGIC └──────────────────────────────────────────────────────┘
# MAGIC     ↓
# MAGIC ResponsesAgentResponse + complete MLflow trace
# MAGIC ```

# COMMAND ----------

# Cell: Single-turn test

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "What are the exercises of week 3 for Real Analysis?"}],
    custom_inputs={"session_id": session_id, "request_id": request_id},
)

logger.info(f"Session ID: {session_id}")
logger.info(f"Request ID: {request_id}")

response = agent.predict(request)
logger.info("=" * 80)
logger.info(response.output[-1].content if response.output else "(no output)")
logger.info("✓ Check MLflow UI for the trace.")

# COMMAND ----------

# Cell: Multi-turn memory demo
# Turn 1 and Turn 2 share the same session_id.
# Turn 2 references "that problem" — the agent resolves this from memory.

conv_session = f"s-conv-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{random.randint(100000, 999999)}"

# Turn 1
turn1 = ResponsesAgentRequest(
    input=[{"role": "user", "content": "Show me problem 2 of week 3 in the Real Analysis course."}],
    custom_inputs={"session_id": conv_session, "request_id": f"req-t1-{uuid4().hex[:8]}"},
)
response1 = agent.predict(turn1)
logger.info("Turn 1 — agent response:")
logger.info(response1.output[-1].content if response1.output else "(no output)")

# Turn 2 — relies on memory: "that problem" should resolve via context
turn2 = ResponsesAgentRequest(
    input=[{"role": "user", "content": "Which lecture sections cover the concepts needed to solve that problem?"}],
    custom_inputs={"session_id": conv_session, "request_id": f"req-t2-{uuid4().hex[:8]}"},
)
response2 = agent.predict(turn2)
logger.info("Turn 2 — agent response (memory-driven):")
logger.info(response2.output[-1].content if response2.output else "(no output)")

logger.info(f"✓ Multi-turn session: {conv_session}")

# COMMAND ----------

# Cell: German query

german_session = f"s-de-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

german_request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "Was sind die Aufgaben der 2. Übung in Analysis I?"}],
    custom_inputs={"session_id": german_session, "request_id": f"req-de-{uuid4().hex[:8]}"},
)
german_response = agent.predict(german_request)
logger.info("German query response:")
logger.info(german_response.output[-1].content if german_response.output else "(no output)")

# COMMAND ----------

# Cell: Trace inspection

session_traces = mlflow.search_traces(
    filter_string=f"request_metadata.`mlflow.trace.session` = '{conv_session}'",
    order_by=["timestamp_ms ASC"],
)

logger.info(f"Traces for session {conv_session}:")
if len(session_traces) > 0:
    scalar_cols = [c for c in session_traces.columns if c not in ["request", "response", "spans", "inputs", "outputs"]]
    display(session_traces[scalar_cols])
else:
    logger.info("No traces found yet. The traces may take a few seconds to appear.")

# COMMAND ----------

# Cell: Evaluation

from learning_buddy.evaluation import evaluate_agent

eval_results = evaluate_agent(cfg=cfg, eval_inputs_path="../learning_buddy_eval_inputs.txt")

# Cast all columns to string before display to avoid Arrow type inference errors
# on mixed-type columns (e.g. assessments containing strings like "yes"/"no")
eval_df = eval_results.tables["eval_results"].astype(str)
display(eval_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Log & Register (Optional)
# MAGIC
# MAGIC Run the cell below to log and register the agent to Unity Catalog.
# MAGIC This creates a versioned model entry with the declared Databricks resources
# MAGIC (LLM endpoint, Vector Search index, embedding endpoint) so that
# MAGIC model serving can automatically bind the correct credentials.

# COMMAND ----------

# Cell: Model log/register (optional — comment out if not needed)

import os

from learning_buddy.agent import log_register_agent

git_sha = os.getenv("GIT_SHA", "local")
run_id = os.getenv("MLFLOW_RUN_ID", f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
model_name = f"{cfg.catalog}.{cfg.schema}.learning_buddy_agent"
agent_code_path = "../learning_buddy_agent.py"

registered = log_register_agent(
    cfg=cfg,
    git_sha=git_sha,
    run_id=run_id,
    agent_code_path=agent_code_path,
    model_name=model_name,
    evaluation_metrics=eval_results.metrics if "eval_results" in dir() else None,
)

logger.info(f"✓ Registered: {model_name} version {registered.version}")
