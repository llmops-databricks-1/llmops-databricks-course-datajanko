# Databricks notebook source
import mlflow
from databricks.sdk.runtime import dbutils

from learning_buddy.agent import log_register_agent
from learning_buddy.config import LearningBuddyProjectConfig
from learning_buddy.evaluation import evaluate_agent

# COMMAND ----------

env = dbutils.widgets.get("env")
git_sha = dbutils.widgets.get("git_sha")
run_id = dbutils.widgets.get("run_id")

cfg = LearningBuddyProjectConfig.load(config_path="learning_buddy_config.yml", env=env)

mlflow.set_experiment(cfg.experiment_name)

model_name = f"{cfg.catalog}.{cfg.schema}.learning_buddy_agent"

# COMMAND ----------

# Run evaluation
results = evaluate_agent(cfg, eval_inputs_path="learning_buddy_eval_inputs.txt")

# COMMAND ----------

# Log and register model
registered_model = log_register_agent(
    cfg=cfg,
    git_sha=git_sha,
    run_id=run_id,
    agent_code_path="learning_buddy_agent.py",
    model_name=model_name,
    evaluation_metrics=results.metrics,
)
