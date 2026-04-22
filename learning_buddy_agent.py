import mlflow
from mlflow.models import ModelConfig

from learning_buddy.agent import SYSTEM_PROMPT, LearningBuddyAgent

config = ModelConfig(
    development_config={
        "catalog": "mlops_dev",
        "schema": "jankoch8",
        "system_prompt": SYSTEM_PROMPT,
        "llm_endpoint": "databricks-llama-4-maverick",
        "embedding_endpoint": "databricks-gte-large-en",
        "vector_search_endpoint": "llmops_course_vs_endpoint",
        "usage_policy_id": "a97cfff0-17de-4bb0-8774-671911359995",
        "lakebase_project_id": "learning-buddy",
    }
)

agent = LearningBuddyAgent(
    llm_endpoint=config.get("llm_endpoint"),
    system_prompt=config.get("system_prompt"),
    catalog=config.get("catalog"),
    schema=config.get("schema"),
    vector_search_endpoint=config.get("vector_search_endpoint"),
    embedding_endpoint=config.get("embedding_endpoint"),
    usage_policy_id=config.get("usage_policy_id"),
    lakebase_project_id=config.get("lakebase_project_id"),
)
mlflow.models.set_model(agent)
