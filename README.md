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

### ArXiv Curator (Lecture 1.3)

The arxiv_curator package demonstrates data ingestion from the arXiv API. It fetches metadata for papers in CS.AI and CS.LG categories and stores them in a Delta table for downstream processing.

**Notebook:** `notebooks/1.3_arxiv_data_ingestion.py`
**Table:** `{catalog}.{schema}.arxiv_papers`

### Learning Buddy (Lecture 1.5)

The learning_buddy package builds a metadata repository of lecture notes and homework assignments for two mathematics courses. It provides references to educational materials that can later be enriched with PDF content, chunking, and embeddings for RAG-based retrieval.

**Data Sources:**

1. **Analysis 1 (University of Bielefeld, German)**
   - Lecture: https://www.math.uni-bielefeld.de/~grigor/a1lect.pdf
   - Course page: https://www.math.uni-bielefeld.de/~grigor/a1ws2024-25.htm
   - **v1 Scope:** Homework sets 0–6 (7 sets)
   - Homework URL pattern: https://www.math.uni-bielefeld.de/~grigor/a1bX.pdf (X = 0 to 14)

2. **Real Analysis (MIT, English)**
   - Lecture: https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/mit18_100af20_lec_full2.pdf
   - Course page: https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/pages/lecture-notes-and-readings/
   - **v1 Scope:** All 12 assignments (1–12)
   - Homework URL pattern: https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/mit18_100af20_hwX.pdf (X = 1 to 12)

**Notebook:** `notebooks/1.5_learning_buddy_ingestion.py`
**Table:** `{catalog}.{schema}.learning_materials`
