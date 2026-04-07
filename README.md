<h1 align="center">
LLMOps Course on Databricks
</h1>

## Practical information
- Weekly lectures on Wednesdays 16:00-18:00 CET.
- Weekly Q&A on Mondays 16:00-17:00 CET.
- Code for the lecture is shared before the lecture.
- Presentation and lecture materials are shared right after the lecture.
- Video of the lecture is uploaded within 24 hours after the lecture.

- Every week we set up a deliverable, and you implement it with your own dataset.
- To submit the deliverable, create a feature branch in that repository, and a PR to main branch. The code can be merged after we review & approve & CI pipeline runs successfully.
- The deliverables can be submitted with a delay (for example, lecture 1 & 2 together), but we expect you to finish all assignments for the course before the demo day.


## Set up your environment
In this course, we use serverless environment 4, which uses Python 3.12.
In our examples, we use UV. Check out the documentation on how to install it: https://docs.astral.sh/uv/getting-started/installation/

To create a new environment and create a lockfile, run:

```
uv sync --extra dev
```

## Use Cases

### ArXiv Curator

Ingests AI/ML paper metadata from the arXiv API (CS.AI and CS.LG categories) and processes it into a searchable vector index.

**Pipeline:**
- `1.3_arxiv_data_ingestion.py` — fetch paper metadata from arXiv API into `arxiv_papers`
- `2.2_pdf_parsing_ai_parse.py` — download PDFs and parse with `ai_parse_document`
- `2.4_embeddings_vector_search.py` — create `arxiv_index` vector search index

**Agent tools:** `search_papers` — semantic + year-filtered search over AI/ML papers

**Scheduled job:** `resources/process_data.yml` — daily at 06:00 Amsterdam time

---

### Learning Buddy

A homework assistant for two mathematics courses. Given lecture notes and homework sets, the agent retrieves relevant references to help solve exercises. Extends the ArXiv Curator patterns with several additions.

**Data Sources:**

1. **Analysis 1 (University of Bielefeld, German)**
   - Lecture notes and homework sets 0–6
2. **Real Analysis (MIT, English)**
   - Lecture notes and homework assignments 1–12

**Pipeline:**
- `1.5_learning_buddy_ingestion.py` — sync course catalog YAML into `learning_materials`
- `2.5_learning_buddy_data_processing.py` — download, parse, chunk PDFs into `learning_materials_chunks`
- `2.6_learning_buddy_vector_search.py` — create `learning_buddy_index` vector search index
- `3.7_learning_buddy_agent.py` — interactive agent with custom + MCP tools

**Scheduled job:** `resources/learning_buddy_data_processing_job.yml` — daily at 07:00 Amsterdam time

**Key additions over ArXiv Curator:**

| Area | Detail |
|---|---|
| YAML course registry | `learning_buddy_courses.yml` declares all materials declaratively; adding a course requires no code change |
| Richer chunk metadata | `course`, `document_type` (`lecture`/`homework`), and `language` columns enable filtered retrieval |
| Bilingual corpus | English (MIT) and German (Bielefeld) documents in the same index; agent responds in the user's query language |
| Three specialized agent tools | `get_problems_by_week` (direct fetch of all chunks for a homework set), `search_lectures` (semantic search filtered to lecture content), `search_homework` (semantic search filtered to homework content) |
| Combined MCP + custom tools | MCP-provided Vector Search and Genie tools are registered alongside custom Python tools in one `ToolRegistry` |
| `LearningBuddyProjectConfig` | Extends `ProjectConfig` with a `courses_path` field pointing to the YAML registry |
| Reset utility job | `resources/reset_learning_buddy_tables_job.yml` drops all three learning buddy tables to support schema migrations |
