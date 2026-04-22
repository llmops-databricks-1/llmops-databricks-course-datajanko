# Databricks notebook source

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.iam import AccessControlRequest, PermissionLevel

from learning_buddy.config import LearningBuddyProjectConfig

cfg = LearningBuddyProjectConfig.from_yaml("../learning_buddy_config.yml")
w = WorkspaceClient()

# COMMAND ----------

spn_app_id = dbutils.secrets.get("dev_SPN", "client_id")

# COMMAND ----------

# Grant CAN_RUN on the Genie space
w.permissions.update(
    request_object_type="genie",
    request_object_id=cfg.genie_space_id,
    access_control_list=[
        AccessControlRequest(
            service_principal_name=spn_app_id,
            permission_level=PermissionLevel.CAN_RUN,
        )
    ],
)

# COMMAND ----------

# Grant CAN_USE on the Vector Search endpoint
vs_endpoint = w.vector_search_endpoints.get_endpoint(cfg.vector_search_endpoint)
w.permissions.update(
    request_object_type="vector-search-endpoints",
    request_object_id=vs_endpoint.id,
    access_control_list=[
        AccessControlRequest(
            service_principal_name=spn_app_id,
            permission_level=PermissionLevel.CAN_USE,
        )
    ],
)

# COMMAND ----------

# Grant CAN_USE on the SQL warehouse
w.permissions.update(
    request_object_type="warehouses",
    request_object_id=cfg.warehouse_id,
    access_control_list=[
        AccessControlRequest(
            service_principal_name=spn_app_id,
            permission_level=PermissionLevel.CAN_USE,
        )
    ],
)

# COMMAND ----------

# Grant CAN_QUERY on the learning buddy serving endpoint
endpoint_name = "learning-buddy-endpoint"
serving_endpoint = w.serving_endpoints.get(endpoint_name)
w.permissions.update(
    request_object_type="serving-endpoints",
    request_object_id=serving_endpoint.id,
    access_control_list=[
        AccessControlRequest(
            service_principal_name=spn_app_id,
            permission_level=PermissionLevel.CAN_QUERY,
        )
    ],
)
