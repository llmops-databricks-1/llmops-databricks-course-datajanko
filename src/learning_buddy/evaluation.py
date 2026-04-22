"""Evaluation module for the Learning Buddy agent."""

from typing import Literal

import mlflow
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import Guidelines

from commons.config import ProjectConfig
from learning_buddy.agent import LearningBuddyAgent

# ---------------------------------------------------------------------------
# Guidelines scorers (LLM-as-judge)
# ---------------------------------------------------------------------------

responds_in_user_language = Guidelines(
    name="responds_in_user_language",
    guidelines=[
        "The response must be written in the same language as the user's question.",
        "If the user asks in German, the entire response must be in German.",
        "If the user asks in English, the entire response must be in English.",
        "Switching languages mid-response is not acceptable.",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)

uses_retrieved_content = Guidelines(
    name="uses_retrieved_content",
    guidelines=[
        "The response must be grounded in retrieved course material, not in general knowledge alone.",
        "The response should reference specific content such as a homework title, lecture section,"
        " problem number, or material identifier found in the search results.",
        "Responses that are entirely generic (e.g. explaining a concept from memory without any"
        " reference to the retrieved material) do not meet this requirement.",
        "Exception: if the retrieval returned no results, the response may acknowledge this and suggest rephrasing — that is acceptable.",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)

stays_in_course_scope = Guidelines(
    name="stays_in_course_scope",
    guidelines=[
        "The response must stay on-topic for the supported courses: Real Analysis (MIT 18.100A) and Analysis I (Universität Bielefeld).",
        "The response must not answer questions about subjects unrelated to these courses (e.g. history, cooking, geography, programming).",
        "If the user asks about an out-of-scope topic, the response should politely redirect to the supported courses without providing the off-topic answer.",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)


# ---------------------------------------------------------------------------
# Custom code scorers
# ---------------------------------------------------------------------------


def _extract_text(outputs: object) -> str:
    """Normalize scorer inputs to a plain string."""
    if isinstance(outputs, str):
        return outputs
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, dict):
            return first.get("text") or first.get("content") or str(first)
        return str(first)
    return str(outputs)


@mlflow.genai.scorer
def cites_sources(outputs: object) -> bool:
    """Check that the response references a source from the retrieved material.

    Looks for patterns that indicate a title or material identifier was cited:
    course IDs, document type words, or explicit homework / lecture references.
    """
    text = _extract_text(outputs).lower()
    indicators = [
        # Course identifiers
        "mit_18_100a",
        "bielefeld_a1",
        # Material type words (EN + DE)
        "homework",
        "problem set",
        "lecture",
        "exercise",
        "übungsblatt",
        "vorlesung",
        "aufgabe",
        # Structural citation patterns
        "week ",
        "woche ",
        "problem ",
        "aufgabe ",
        "section ",
        "abschnitt ",
    ]
    return any(ind in text for ind in indicators)


@mlflow.genai.scorer
def response_not_too_long(outputs: object) -> float:
    """Score based on response length.

    Returns 1.0 for responses ≤ 200 words, decays linearly to 0.0 at 600 words.
    Encourages focused, concise answers rather than walls of text.
    """
    text = _extract_text(outputs)
    word_count = len(text.split())
    if word_count <= 200:
        return 1.0
    if word_count >= 600:
        return 0.0
    return 1.0 - (word_count - 200) / 400.0


# ---------------------------------------------------------------------------
# Judges (LLM-as-judge with scored / categorical output)
# ---------------------------------------------------------------------------

_JUDGE_MODEL = "databricks:/databricks-gpt-oss-120b"

relevance_judge = make_judge(
    name="answer_relevance",
    instructions=(
        "Evaluate how relevant the response in {{ outputs }} is to the homework or lecture "
        "question in {{ inputs }}.\n"
        "Score from 1 to 5:\n"
        "1 - Completely off-topic or does not address the question at all\n"
        "2 - Tangentially related but missing the core of the question\n"
        "3 - Partially answers the question but lacks key details\n"
        "4 - Clearly relevant and addresses the main question\n"
        "5 - Directly and completely answers the question with appropriate course context"
    ),
    model=_JUDGE_MODEL,
    feedback_value_type=int,
)

source_quality_judge = make_judge(
    name="source_citation_quality",
    instructions=(
        "Evaluate how well the response in {{ outputs }} cites and contextualises sources "
        "from the course material in response to the question in {{ inputs }}.\n"
        "Score from 1 to 5:\n"
        "1 - No sources or material references at all\n"
        "2 - Vague mention of sources without useful detail\n"
        "3 - References course material but without week, problem, or section specifics\n"
        "4 - Cites specific material (e.g. week number, problem set, lecture section)\n"
        "5 - Precise citations that directly help the student locate the relevant material"
    ),
    model=_JUDGE_MODEL,
    feedback_value_type=int,
)

language_appropriateness_judge = make_judge(
    name="language_appropriateness",
    instructions=(
        "Determine whether the language of the response in {{ outputs }} matches the language "
        "of the user's question in {{ inputs }}. "
        "Classify as 'correct' if the response is in the same language as the question, "
        "'mixed' if the response switches languages mid-answer, "
        "or 'incorrect' if the response is entirely in a different language than the question."
    ),
    model=_JUDGE_MODEL,
    feedback_value_type=Literal["correct", "mixed", "incorrect"],
)


# ---------------------------------------------------------------------------
# evaluate_agent
# ---------------------------------------------------------------------------


def evaluate_agent(
    cfg: ProjectConfig,
    eval_inputs_path: str,
) -> mlflow.models.EvaluationResult:
    """Run evaluation on the LearningBuddyAgent.

    Args:
        cfg: Project configuration.
        eval_inputs_path: Path to a text file with one evaluation question per line.

    Returns:
        MLflow EvaluationResult with scores for all scorers.
    """
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

    with open(eval_inputs_path) as f:
        eval_data = [{"inputs": {"question": line.strip()}} for line in f if line.strip()]

    def predict_fn(question: str) -> str:
        from mlflow.types.responses import ResponsesAgentRequest

        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": question}],
        )
        result = agent.predict(request)
        items = result.output
        if items:
            last = items[-1]
            content = last.get("content", "") if isinstance(last, dict) else getattr(last, "content", str(last))
            if isinstance(content, list) and content:
                return content[0].get("text", str(content[0]))
            return str(content)
        return ""

    mlflow.set_experiment(cfg.experiment_name)

    return mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=eval_data,
        scorers=[
            responds_in_user_language,
            uses_retrieved_content,
            stays_in_course_scope,
            cites_sources,
            response_not_too_long,
            relevance_judge,
            source_quality_judge,
            language_appropriateness_judge,
        ],
    )


def load_eval_data(eval_inputs_path: str) -> list[dict]:
    """Load evaluation data from a file.

    Args:
        eval_inputs_path: Path to file with one question per line.

    Returns:
        List of evaluation data dicts in the format [{"inputs": {"question": ...}}].
    """
    with open(eval_inputs_path) as f:
        return [{"inputs": {"question": line.strip()}} for line in f if line.strip()]
