import mlflow
from mlflow.models import ModelConfig

from commons.config import ProjectConfig
from learning_buddy.agent import LearningBuddyAgent

config = ModelConfig(
    development_config={
        "catalog": "mlops_dev",
        "schema": "learning_buddy",
        "system_prompt": "placeholder",
        "llm_endpoint": "databricks-gpt-oss-120b",
        "lakebase_project_id": "learning-buddy-lakebase",
    }
)

cfg = ProjectConfig(
    catalog=config.get("catalog"),
    schema=config.get("schema"),
    volume="learning_materials",
    llm_endpoint=config.get("llm_endpoint"),
    embedding_endpoint="databricks-bge-large-en",
    warehouse_id="",
    vector_search_endpoint="learning-buddy-vs",
    usage_policy_id="",
    lakebase_project_id=config.get("lakebase_project_id"),
    experiment_name="/Shared/llmops-learning-buddy",
    system_prompt=config.get("system_prompt"),
)

agent = LearningBuddyAgent(
    config=cfg,
    system_prompt=config.get("system_prompt"),
    lakebase_project_id=config.get("lakebase_project_id"),
)
mlflow.models.set_model(agent)
