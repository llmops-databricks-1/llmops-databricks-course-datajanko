# Databricks notebook source
"""
3.7 Learning Buddy Agent

Combines patterns from 3.1 (custom tools) and 3.2 (MCP/tool registry) to build
an agentic learning buddy for math courses.

Supported courses:
- mit_18_100a  — Real Analysis (English, MIT 18.100A)
- bielefeld_a1 — Analysis I (German, Universität Bielefeld)

The agent can:
- Fetch all problems from a specific homework set (week)
- Fetch a single problem by its number within a homework set
- Search lecture notes for concepts, theorems, and definitions
- Search homework exercises by topic or keyword
- Cross-search both courses and synthesize results

Queries NOT supported (require full-corpus aggregation, not similarity search):
- "What are the most referenced topics in Analysis I?" → use Genie Space instead
"""

# COMMAND ----------

# Cell 1: Setup & Config

import json

from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from arxiv_curator.mcp import ToolInfo
from commons.config import get_env, load_config

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("learning_buddy_config.yml", env)

w = WorkspaceClient()

logger.info(f"✓ Environment: {env}")
logger.info(f"✓ Catalog: {cfg.catalog}.{cfg.schema}")
logger.info(f"✓ LLM endpoint: {cfg.llm_endpoint}")
logger.info(f"✓ Vector search endpoint: {cfg.vector_search_endpoint}")

# COMMAND ----------

# Cell 2: Vector Search Manager

from learning_buddy.vector_search import LearningBuddyVectorSearchManager

vs_manager = LearningBuddyVectorSearchManager(config=cfg)

logger.info(f"✓ Vector Search Index: {vs_manager.index_name}")

# COMMAND ----------

# Cell 3: Helper — parse vector search results


def parse_vector_search_results(results) -> list[dict]:
    """Parse vector search results from manifest/array format to list of dicts."""
    columns = [col["name"] for col in results.get("manifest", {}).get("columns", [])]
    data_array = results.get("result", {}).get("data_array", [])
    return [dict(zip(columns, row)) for row in data_array]


# COMMAND ----------

# Cell 4: Custom Tools


def get_problems_by_week(course: str, week: int) -> str:
    """Fetch the full text of a homework set by retrieving all its chunks.

    Filters the vector index by material_id (e.g. 'mit_18_100a_hw3') to get
    every chunk belonging to the set, then concatenates them in order. Because
    the chunking pipeline splits on token boundaries rather than problem
    boundaries, a single problem may span multiple chunks. The returned text
    is therefore the continuous homework sheet content — the LLM should read
    through it to identify individual problems.

    Args:
        course: Course identifier — 'mit_18_100a' or 'bielefeld_a1'
        week: Homework set / week number (integer, e.g. 3)

    Returns:
        JSON string with 'title', 'material_id', and 'text' (full concatenated
        content of all chunks), or an error if the set has not been ingested.
    """
    material_id = f"{course}_hw{week}"
    raw = vs_manager.search(
        "exercise problem",
        num_results=20,
        filters={"material_id": material_id},
    )
    chunks = parse_vector_search_results(raw)

    if not chunks:
        return json.dumps(
            {
                "error": f"No content found for course='{course}', week={week}. "
                f"Expected material_id='{material_id}'. "
                "Check that the homework set has been ingested."
            },
            ensure_ascii=False,
        )

    title = chunks[0].get("title", material_id)
    full_text = "\n\n".join(c.get("text") or "" for c in chunks)
    return json.dumps(
        {"course": course, "week": week, "material_id": material_id, "title": title, "text": full_text},
        ensure_ascii=False,
        indent=2,
    )


def search_lectures(query: str, course: str | None = None) -> str:
    """Search lecture notes for concepts, theorems, and definitions.

    Args:
        query: Natural language search query (e.g. "definition of Cauchy sequence")
        course: Optional course filter — 'mit_18_100a' or 'bielefeld_a1'

    Returns:
        JSON string with matching lecture chunks
    """
    filters: dict = {"document_type": "lecture"}
    if course:
        filters["course"] = course

    raw = vs_manager.search(query, num_results=5, filters=filters)
    chunks = parse_vector_search_results(raw)

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


def search_homework(query: str, course: str | None = None) -> str:
    """Search homework and exercise sheets by topic or keyword.

    Use this to find exercises related to a concept when you don't know the
    exact week/set. To retrieve a specific week's problems, use
    get_problems_by_week instead.

    Args:
        query: Natural language search query (e.g. "epsilon delta continuity")
        course: Optional course filter — 'mit_18_100a' or 'bielefeld_a1'

    Returns:
        JSON string with matching homework chunks
    """
    filters: dict = {"document_type": "homework"}
    if course:
        filters["course"] = course

    raw = vs_manager.search(query, num_results=5, filters=filters)
    chunks = parse_vector_search_results(raw)

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


# Manual smoke tests
# get_problems_by_week("mit_18_100a", 3)
# search_lectures("cauchy sequence", course="mit_18_100a")

# COMMAND ----------

# Cell 5: Tool specs + ToolInfo objects

get_problems_by_week_spec = {
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

search_lectures_spec = {
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

search_homework_spec = {
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

get_problems_by_week_tool = ToolInfo(
    name="get_problems_by_week",
    spec=get_problems_by_week_spec,
    exec_fn=get_problems_by_week,
)
search_lectures_tool = ToolInfo(name="search_lectures", spec=search_lectures_spec, exec_fn=search_lectures)
search_homework_tool = ToolInfo(name="search_homework", spec=search_homework_spec, exec_fn=search_homework)

logger.info("✓ Tools defined: get_problems_by_week, search_lectures, search_homework")

# COMMAND ----------

# Cell 6: ToolRegistry + SimpleAgent (pattern from 3.1)

from typing import Any


class ToolRegistry:
    """Registry for managing agent tools."""

    def __init__(self):
        self._tools: dict[str, ToolInfo] = {}

    def register(self, tool: ToolInfo) -> None:
        self._tools[tool.name] = tool
        logger.info(f"✓ Registered tool: {tool.name}")

    def get_tool(self, name: str) -> ToolInfo:
        if name not in self._tools:
            raise ValueError(f"Tool not found: {name}")
        return self._tools[name]

    def get_all_specs(self) -> list[dict]:
        return [tool.spec for tool in self._tools.values()]

    def execute(self, name: str, args: dict) -> Any:
        tool = self.get_tool(name)
        return tool.exec_fn(**args)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_all_tools(self) -> list[ToolInfo]:
        return list(self._tools.values())


class SimpleAgent:
    """Agentic loop: calls LLM, executes tool calls, and iterates until a final answer."""

    def __init__(self, llm_endpoint: str, system_prompt: str, tools: list[ToolInfo]):
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self._tools_dict = {tool.name: tool for tool in tools}
        self._client = OpenAI(
            api_key=w.tokens.create(lifetime_seconds=1200).token_value,
            base_url=f"{w.config.host}/serving-endpoints",
        )

    def get_tool_specs(self) -> list[dict]:
        return [tool.spec for tool in self._tools_dict.values()]

    def execute_tool(self, tool_name: str, args: dict) -> str:
        if tool_name not in self._tools_dict:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self._tools_dict[tool_name].exec_fn(**args)

    def chat(self, user_message: str, max_iterations: int = 10) -> str:
        """Run the agentic loop for a single user message."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        for _ in range(max_iterations):
            response = self._client.chat.completions.create(
                model=self.llm_endpoint,
                messages=messages,
                tools=self.get_tool_specs() if self._tools_dict else None,
            )

            assistant_message = response.choices[0].message

            if assistant_message.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                            }
                            for tc in assistant_message.tool_calls
                        ],
                    }
                )

                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    logger.info(f"→ Tool call: {tool_name}({tool_args})")

                    try:
                        result = self.execute_tool(tool_name, tool_args)
                    except Exception as e:
                        result = f"Error: {e}"

                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})
            else:
                return assistant_message.content

        return "Max iterations reached."


# COMMAND ----------

# Cell 7: System Prompt

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

# COMMAND ----------

# Cell 8: Agent instantiation + test queries

registry = ToolRegistry()
registry.register(get_problems_by_week_tool)
registry.register(search_lectures_tool)
registry.register(search_homework_tool)

agent = SimpleAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=SYSTEM_PROMPT,
    tools=registry.get_all_tools(),
)

logger.info("✓ Learning Buddy Agent ready")
logger.info(f"  Tools: {registry.list_tools()}")

# COMMAND ----------

# Query 1: All problems for a specific week
logger.info("=" * 80)
logger.info("Query 1: All problems for week 3")
response = agent.chat("What are the exercises of Week 3 for the Real Analysis course?")
logger.info(f"\n{response}")

# COMMAND ----------

# Query 2: Specific problem from within a week (agent reads full text and extracts it)
logger.info("=" * 80)
logger.info("Query 2: Specific problem within a week")
response = agent.chat("Show me problem 2 of week 3 in the Real Analysis course.")
logger.info(f"\n{response}")

# COMMAND ----------

# Query 3: Lecture references for a specific problem
logger.info("=" * 80)
logger.info("Query 3: Lecture reference for a specific problem")
response = agent.chat("Which topics from the lecture can help solve problem 2 of week 3 in the Real Analysis course?")
logger.info(f"\n{response}")

# COMMAND ----------

# Query 4: Cross-course concept lookup
logger.info("=" * 80)
logger.info("Query 4: Cross-course concept lookup")
response = agent.chat("What are concepts from Analysis I that are also used in Real Analysis?")
logger.info(f"\n{response}")

# COMMAND ----------

# Query 5: German query — problems for a specific week
logger.info("=" * 80)
logger.info("Query 5: German — problems for week 2")
response = agent.chat("Was sind die Aufgaben der 2. Übung in Analysis I?")
logger.info(f"\n{response}")

# COMMAND ----------

# Query 6: Direct definition lookup
logger.info("=" * 80)
logger.info("Query 6: Direct definition lookup")
response = agent.chat("Provide a reference to the definition of a Cauchy sequence.")
logger.info(f"\n{response}")
