"""Learning Buddy Agent — production-grade ResponsesAgent with MLflow tracing and Lakebase memory."""

import json
import os
import urllib.parse
import warnings
from collections.abc import Callable, Generator
from datetime import datetime
from typing import Any
from uuid import uuid4

import backoff
import mlflow
import nest_asyncio
import openai
import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import PostgresAPI
from loguru import logger
from mlflow import MlflowClient
from mlflow.entities import SpanType
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from psycopg_pool import ConnectionPool
from pydantic import BaseModel

from learning_buddy.config import LearningBuddyProjectConfig
from learning_buddy.vector_search import LearningBuddyVectorSearchManager

# ---------------------------------------------------------------------------
# Inlined from commons.mcp
# ---------------------------------------------------------------------------


class ToolInfo(BaseModel):
    """Tool information for agent integration."""

    name: str
    spec: dict
    exec_fn: Callable

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# Inlined from commons.memory
# ---------------------------------------------------------------------------


class LakebaseMemory:
    """Handles session message persistence using Lakebase (PostgreSQL)."""

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self._pool: ConnectionPool | None = None

    def _get_connection_string(self) -> str:
        client_id = os.environ.get("LAKEBASE_SP_CLIENT_ID")
        client_secret = os.environ.get("LAKEBASE_SP_CLIENT_SECRET")
        host = os.environ.get("LAKEBASE_SP_HOST")

        w = WorkspaceClient(host=host, client_id=client_id, client_secret=client_secret) if client_id and client_secret and host else WorkspaceClient()

        pg_api = PostgresAPI(w.api_client)

        if client_id:
            username = client_id
        else:
            user = w.current_user.me()
            username = urllib.parse.quote_plus(user.user_name)

        project_parent = f"projects/{self.project_id}"
        default_branch = next(iter(pg_api.list_branches(parent=project_parent)))
        endpoint = next(iter(pg_api.list_endpoints(parent=default_branch.name)))
        host = endpoint.status.hosts.host
        pg_credential = pg_api.generate_database_credential(endpoint=endpoint.name)

        return f"postgresql://{username}:{pg_credential.token}@{host}:5432/databricks_postgres?sslmode=require"

    def _get_pool(self) -> ConnectionPool:
        if self._pool is None:
            self._pool = ConnectionPool(conninfo=self._get_connection_string(), min_size=1, max_size=5)
        return self._pool

    def _reset_pool(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        try:
            with self._get_pool().connection() as conn:
                result = conn.execute(
                    "SELECT message_data FROM session_messages WHERE session_id = %s ORDER BY created_at ASC",
                    (session_id,),
                ).fetchall()
                return [row[0] for row in result]
        except psycopg.OperationalError:
            self._reset_pool()
            raise
        except Exception as e:
            logger.warning(f"Failed to load session messages: {e}")
            return []

    def save_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        try:
            with self._get_pool().connection() as conn:
                for msg in messages:
                    conn.execute(
                        "INSERT INTO session_messages (session_id, message_data) VALUES (%s, %s)",
                        (session_id, json.dumps(msg)),
                    )
        except psycopg.OperationalError:
            self._reset_pool()
            raise
        except Exception as e:
            logger.warning(f"Failed to save session messages: {e}")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Learning Buddy — an AI assistant that helps students with math courses.

## Available Courses
- **mit_18_100a**: Real Analysis (MIT 18.100A, English), homework sets 1–6
- **bielefeld_a1**: Analysis I (Universität Bielefeld, German), homework sets 0–6

## Available Tools
- **get_problems_by_week**: Fetch the full text of a homework set for a given week.
  Use when the user asks "what are the exercises of week N", "show me problem set N",
  or "show me problem N of week M". The text is a continuous concatenation of chunks —
  read through it to identify individual problems; chunk boundaries do not align with
  problem boundaries.
- **search_lectures**: Search lecture notes for concepts, theorems, definitions, proofs.
  Use when the user asks about a concept or needs lecture material relevant to an exercise.
- **search_homework**: Search exercises by topic or keyword across all sets.
  Use when the week is unknown and the user wants exercises related to a concept.

## Instructions
1. Always use the tools to retrieve material before answering. Do not answer from memory alone.
2. When the user specifies a week/set number, use get_problems_by_week and read the full
   returned text to identify the specific problem they are asking about.
3. When the user asks about a concept without specifying a week, use search_lectures or search_homework.
4. For cross-course questions, call tools for each course separately and synthesize the results.
5. Always cite the source title from search results when referencing specific content.
6. Respond in the **same language** as the user's question (German → German, English → English).
7. If results are empty or irrelevant, say so honestly and suggest rephrasing.
"""

# ---------------------------------------------------------------------------
# Tool specs (module-level constants)
# ---------------------------------------------------------------------------

GET_PROBLEMS_BY_WEEK_SPEC = {
    "type": "function",
    "function": {
        "name": "get_problems_by_week",
        "description": (
            "Fetch the full text of a homework set for a given week. "
            "Use this when the user asks 'what are the exercises of week N', "
            "'show me problem set N', or 'show me problem N of week M'. "
            "Returns the concatenated content of all chunks — read through the "
            "text to identify individual problems, as chunk boundaries do not "
            "align with problem boundaries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "course": {
                    "type": "string",
                    "enum": ["mit_18_100a", "bielefeld_a1"],
                    "description": "'mit_18_100a' = Real Analysis (English), 'bielefeld_a1' = Analysis I (German)",
                },
                "week": {
                    "type": "integer",
                    "description": "Homework set / week number (e.g. 3 for week 3 / problem set 3)",
                },
            },
            "required": ["course", "week"],
        },
    },
}

SEARCH_LECTURES_SPEC = {
    "type": "function",
    "function": {
        "name": "search_lectures",
        "description": (
            "Search lecture notes for mathematical concepts, theorems, definitions, and proofs. "
            "Use this when the user asks about a concept, wants a theorem reference, or needs "
            "lecture material relevant to an exercise."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query, e.g. 'definition of Cauchy sequence'",
                },
                "course": {
                    "type": "string",
                    "enum": ["mit_18_100a", "bielefeld_a1"],
                    "description": "Optional course filter. 'mit_18_100a' = Real Analysis (English), 'bielefeld_a1' = Analysis I (German)",
                },
            },
            "required": ["query"],
        },
    },
}

SEARCH_HOMEWORK_SPEC = {
    "type": "function",
    "function": {
        "name": "search_homework",
        "description": (
            "Search homework and exercise sheets by topic or keyword. "
            "Use this to find exercises related to a concept when the week is unknown. "
            "To retrieve all problems from a specific week, use get_problems_by_week instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query, e.g. 'epsilon delta continuity'",
                },
                "course": {
                    "type": "string",
                    "enum": ["mit_18_100a", "bielefeld_a1"],
                    "description": "Optional course filter. 'mit_18_100a' = Real Analysis (English), 'bielefeld_a1' = Analysis I (German)",
                },
            },
            "required": ["query"],
        },
    },
}


class LearningBuddyAgent(ResponsesAgent):
    """Production ResponsesAgent for the Learning Buddy use case.

    Uses native Python tools wrapping LearningBuddyVectorSearchManager —
    no MCP, because the custom filtering logic (material_id, document_type)
    is not available through the generic MCP Vector Search tool.
    """

    def __init__(
        self,
        llm_endpoint: str,
        system_prompt: str,
        catalog: str,
        schema: str,
        vector_search_endpoint: str,
        embedding_endpoint: str,
        usage_policy_id: str,
        lakebase_project_id: str | None = None,
    ) -> None:
        nest_asyncio.apply()

        self.system_prompt = system_prompt
        self.llm_endpoint = llm_endpoint
        self.workspace_client = WorkspaceClient()
        self.model_serving_client = self.workspace_client.serving_endpoints.get_open_ai_client()

        self._vs_manager = LearningBuddyVectorSearchManager(
            catalog=catalog,
            schema=schema,
            vector_search_endpoint=vector_search_endpoint,
            embedding_endpoint=embedding_endpoint,
            usage_policy_id=usage_policy_id,
        )

        self.memory: LakebaseMemory | None = None
        if lakebase_project_id:
            self.memory = LakebaseMemory(project_id=lakebase_project_id)

        self._tools_dict = {tool.name: tool for tool in self._build_tools()}

    # ------------------------------------------------------------------
    # Tool helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_vector_search_results(results: dict) -> list[dict]:
        """Parse vector search results from manifest/array format to list of dicts."""
        columns = [col["name"] for col in results.get("manifest", {}).get("columns", [])]
        data_array = results.get("result", {}).get("data_array", [])
        return [dict(zip(columns, row, strict=False)) for row in data_array]

    def _get_problems_by_week(self, course: str, week: int) -> str:
        """Fetch the full text of a homework set by retrieving all its chunks."""
        material_id = f"{course}_hw{week}"
        raw = self._vs_manager.search(
            "exercise problem",
            num_results=20,
            filters={"material_id": material_id},
        )
        chunks = self._parse_vector_search_results(raw)

        if not chunks:
            return json.dumps(
                {
                    "error": (
                        f"No content found for course='{course}', week={week}. "
                        f"Expected material_id='{material_id}'. "
                        "Check that the homework set has been ingested."
                    )
                },
                ensure_ascii=False,
            )

        title = chunks[0].get("title", material_id)
        full_text = "\n\n".join(c.get("text") or "" for c in chunks)
        return json.dumps(
            {
                "course": course,
                "week": week,
                "material_id": material_id,
                "title": title,
                "text": full_text,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _search_lectures(self, query: str, course: str | None = None) -> str:
        """Search lecture notes for concepts, theorems, and definitions."""
        filters: dict = {"document_type": "lecture"}
        if course:
            filters["course"] = course

        raw = self._vs_manager.search(query, num_results=5, filters=filters)
        chunks = self._parse_vector_search_results(raw)

        results = [
            {
                "chunk_id": c.get("chunk_id"),
                "material_id": c.get("material_id"),
                "title": c.get("title"),
                "course": c.get("course"),
                "language": c.get("language"),
                "excerpt": (c.get("text") or "")[:300],
            }
            for c in chunks
        ]
        return json.dumps(results, ensure_ascii=False, indent=2)

    def _search_homework(self, query: str, course: str | None = None) -> str:
        """Search homework and exercise sheets by topic or keyword."""
        filters: dict = {"document_type": "homework"}
        if course:
            filters["course"] = course

        raw = self._vs_manager.search(query, num_results=5, filters=filters)
        chunks = self._parse_vector_search_results(raw)

        results = [
            {
                "chunk_id": c.get("chunk_id"),
                "material_id": c.get("material_id"),
                "title": c.get("title"),
                "course": c.get("course"),
                "language": c.get("language"),
                "excerpt": (c.get("text") or "")[:300],
            }
            for c in chunks
        ]
        return json.dumps(results, ensure_ascii=False, indent=2)

    def _build_tools(self) -> list[ToolInfo]:
        """Build and return the list of ToolInfo objects for this agent."""
        return [
            ToolInfo(
                name="get_problems_by_week",
                spec=GET_PROBLEMS_BY_WEEK_SPEC,
                exec_fn=self._get_problems_by_week,
            ),
            ToolInfo(
                name="search_lectures",
                spec=SEARCH_LECTURES_SPEC,
                exec_fn=self._search_lectures,
            ),
            ToolInfo(
                name="search_homework",
                spec=SEARCH_HOMEWORK_SPEC,
                exec_fn=self._search_homework,
            ),
        ]

    # ------------------------------------------------------------------
    # Agent methods (mirrored from ArxivAgent)
    # ------------------------------------------------------------------

    def get_tool_specs(self) -> list[dict]:
        """Returns tool specifications in the format OpenAI expects."""
        return [tool_info.spec for tool_info in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> object:
        """Executes the specified tool with the given arguments."""
        return self._tools_dict[tool_name].exec_fn(**args)

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def call_llm(
        self,
        messages: list[dict[str, Any]],
    ) -> Generator[dict[str, Any], None, None]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
            stream = self.model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=to_chat_completions_input(messages),
                tools=self.get_tool_specs(),
                stream=True,
            )
            with mlflow.start_span(name="call_llm", span_type=SpanType.LLM) as span:
                last_chunk: dict[str, Any] = {}
                for chunk in stream:
                    chunk_dict = chunk.to_dict()
                    last_chunk = chunk_dict
                    yield chunk_dict
                llm_request_id = stream.response.headers.get("x-request-id")
                outputs: dict[str, Any] = {
                    "model": last_chunk.get("model"),
                    "usage": last_chunk.get("usage"),
                }
                if llm_request_id:
                    outputs["llm_request_id"] = llm_request_id
                span.set_outputs(outputs)

    def handle_tool_call(self, tool_call: dict[str, Any], messages: list[dict[str, Any]]) -> ResponsesAgentStreamEvent:
        """Execute tool calls, add them to the running message history, and return a stream event."""
        args = json.loads(tool_call["arguments"])
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))

        tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result)
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)

    @mlflow.trace(span_type=SpanType.RETRIEVER, name="memory_load")
    def load_memory(self, session_id: str) -> list[dict[str, Any]]:
        """Load previous messages from Lakebase memory."""
        if self.memory:
            return self.memory.load_messages(session_id)
        return []

    @mlflow.trace(span_type=SpanType.CHAIN, name="memory_save")
    def save_memory(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Save new messages to Lakebase memory."""
        self.memory.save_messages(session_id, messages)

    def _extract_output_items(
        self,
        events: list[ResponsesAgentStreamEvent],
    ) -> list[dict[str, Any]]:
        """Extract and serialize output items from stream events."""
        items = []
        for e in events:
            if e.type != "response.output_item.done":
                continue
            item = e.item if isinstance(e.item, dict) else e.item.model_dump()
            if item.get("type") == "message":
                items.append(item)
        return items

    def _run_tool_loop(
        self,
        messages: list[dict[str, Any]],
        max_iter: int = 10,
    ) -> list[ResponsesAgentStreamEvent]:
        """Run the LLM <-> tool loop until the model stops or max_iter."""
        events: list[ResponsesAgentStreamEvent] = []
        for _ in range(max_iter):
            last_msg = messages[-1]
            if last_msg.get("role") == "assistant":
                break
            elif last_msg.get("type") == "function_call":
                events.append(self.handle_tool_call(last_msg, messages))
            else:
                events.extend(
                    output_to_responses_items_stream(
                        chunks=self.call_llm(messages),
                        aggregator=messages,
                    ),
                )
        else:
            events.append(
                ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_text_output_item(
                        "Max iterations reached. Stopping.",
                        str(uuid4()),
                    ),
                ),
            )
        return events

    @mlflow.trace(span_type=SpanType.CHAIN)
    def call_and_run_tools(
        self,
        request_input: list[dict[str, Any]],
        previous_messages: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> list[ResponsesAgentStreamEvent]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        if previous_messages:
            messages.extend(previous_messages)
        messages.extend(request_input)

        mlflow.update_current_trace(
            tags={
                "git_sha": os.getenv("GIT_SHA", "local"),
                "model_serving_endpoint_name": os.getenv("MODEL_SERVING_ENDPOINT_NAME", "local"),
                "model_version": os.getenv("MODEL_VERSION", "local"),
            },
            metadata=({"mlflow.trace.session": session_id} if session_id else {}),
            client_request_id=request_id,
        )

        events = self._run_tool_loop(messages)

        if session_id and self.memory:
            self.save_memory(
                session_id,
                request_input + self._extract_output_items(events),
            )
        return events

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        events = list(self.predict_stream(request))
        return ResponsesAgentResponse(
            output=self._extract_output_items(events),
            custom_outputs=request.custom_inputs,
        )

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        custom = request.custom_inputs or {}
        session_id = custom.get("session_id")
        request_id = custom.get("request_id")

        previous_messages = self.load_memory(session_id) if session_id and self.memory else []

        request_input = [i.model_dump() for i in request.input]
        events = self.call_and_run_tools(
            request_input=request_input,
            previous_messages=previous_messages,
            request_id=request_id,
            session_id=session_id,
        )
        yield from events


def log_register_agent(
    cfg: LearningBuddyProjectConfig,
    git_sha: str,
    run_id: str,
    agent_code_path: str,
    model_name: str,
    evaluation_metrics: dict | None = None,
) -> mlflow.entities.model_registry.RegisteredModel:
    """Log and register the LearningBuddyAgent to Unity Catalog.

    Args:
        cfg: Project configuration containing catalog, schema, and other settings.
        git_sha: Git commit SHA for tracking.
        run_id: Run identifier for tracking.
        agent_code_path: Path to the agent Python file.
        model_name: Model path in Unity Catalog (e.g. catalog.schema.model_name).
        evaluation_metrics: Optional evaluation metrics to log.

    Returns:
        RegisteredModel object from Unity Catalog.
    """
    resources = [
        DatabricksServingEndpoint(endpoint_name=cfg.llm_endpoint),
        DatabricksVectorSearchIndex(index_name=f"{cfg.catalog}.{cfg.schema}.learning_buddy_index"),
        DatabricksServingEndpoint(endpoint_name=cfg.embedding_endpoint),
    ]

    model_config = {
        "catalog": cfg.catalog,
        "schema": cfg.schema,
        "system_prompt": cfg.system_prompt,
        "llm_endpoint": cfg.llm_endpoint,
        "embedding_endpoint": cfg.embedding_endpoint,
        "vector_search_endpoint": cfg.vector_search_endpoint,
        "usage_policy_id": cfg.usage_policy_id,
        "lakebase_project_id": cfg.lakebase_project_id,
    }

    test_request = {"input": [{"role": "user", "content": "What are the exercises of week 3 for Real Analysis?"}]}

    mlflow.set_experiment(cfg.experiment_name)
    ts = datetime.now().strftime("%Y-%m-%d")

    with mlflow.start_run(
        run_name=f"learning-buddy-agent-{ts}",
        tags={"git_sha": git_sha, "run_id": run_id},
    ):
        # Explicitly list pip requirements to prevent MLflow from auto-inferring
        # 'learning-buddy' (not on PyPI). The package source is captured via
        # infer_code_paths=True, which bundles the local .py files into the artifact.
        pip_requirements = [
            "cffi==2.0.0",
            "cloudpickle==3.1.2",
            "numpy==2.4.3",
            "pandas==2.2.3",
            "pyarrow==23.0.1",
            "databricks-sdk==0.101.0",
            "pydantic==2.12.5",
            "loguru==0.7.3",
            "python-dotenv==1.2.2",
            "databricks-vectorsearch==0.66",
            "openai==2.29.0",
            "databricks-mcp==0.9.0",
            "backoff==2.2.1",
            "mlflow==3.10.1",
            "nest-asyncio==1.6.0",
            "arxiv==2.4.1",
            "databricks-agents==1.9.3",
            "psycopg==3.3.3",
            "psycopg-pool==3.3.0",
            "psycopg[binary]==3.3.3",
        ]
        model_info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=agent_code_path,
            pip_requirements=pip_requirements,
            resources=resources,
            input_example=test_request,
            model_config=model_config,
        )
        if evaluation_metrics:
            mlflow.log_metrics(evaluation_metrics)

    logger.info(f"Registering model: {model_name}")
    registered_model = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=model_name,
        env_pack="databricks_model_serving",
        tags={"git_sha": git_sha, "run_id": run_id},
    )
    logger.info(f"Registered version: {registered_model.version}")

    client = MlflowClient()
    logger.info("Setting alias 'latest-model'")
    client.set_registered_model_alias(
        name=model_name,
        alias="latest-model",
        version=registered_model.version,
    )
    return registered_model
