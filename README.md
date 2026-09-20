# SafeSight AI

**Safety Intelligence Platform --- Incident Precursor Detector**

SafeSight AI is an AI-assisted workplace safety analysis platform built
for the **Safety Report Analysis Agent (Incident Precursor Detector)**
challenge. It helps safety officers turn large volumes of free-text and
historical incident reports into structured risk information, recurring
patterns, emerging signals, and traceable evidence for investigation.

> SafeSight surfaces historical and changing safety patterns to support
> human review. It does not predict that a future incident will occur.

## Problem

Organizations can collect thousands of safety observations, near-misses,
and incident reports. Manually reviewing every report makes it difficult
to identify recurring hazards and early warning patterns quickly.

SafeSight AI brings report analysis, historical pattern detection,
evidence retrieval, and human review into one workflow.

## Key Features

-   **AI report parsing** --- extracts hazards, risk factors, location
    context, equipment, people at risk, and potential consequences from
    free text.
-   **Risk classification** --- assigns LOW, MEDIUM, or HIGH risk with
    reasoning and a recommended action.
-   **Human risk override** --- safety officers can change an AI risk
    label and log a reason; recent corrections are reused as few-shot
    feedback in later classifications.
-   **Emerging precursor detection** --- compares recent and preceding
    report activity to surface changing incident categories.
-   **Interactive safety dashboard** --- filters historical analysis by
    state and year.
-   **Risk Hotspot Matrix** --- compares incident frequency with serious
    historical outcomes.
-   **Safety Action Center** --- turns evidence into prioritized areas
    for human investigation.
-   **Safety Advisor** --- natural-language questions are routed by
    LangGraph to deterministic Python tools.
-   **RAG evidence retrieval** --- retrieves related historical
    incidents from a Chroma vector store with traceable report IDs.
-   **Investigation export** --- downloads the current filtered safety
    context as a CSV summary.
-   **Graceful error handling** --- historical analytics and local
    retrieval remain usable when an external model request is
    unavailable.

## Architecture

``` text
Historical Safety Reports
        |
        +--> Pandas Pattern Analysis --> Emerging Precursors
        |                              --> Dashboard / Hotspots
        |
New Free-Text Report
        |
        v
LangGraph Coordinator
        |
        +--> Report Parsing Agent
        +--> Risk Scoring Agent <---- Human Override Feedback
        +--> Pattern Analysis
        +--> RAG Evidence Retrieval
        |
        v
Structured Safety Review

Safety Officer Question
        |
        v
LangGraph Tool Router
        |
        +--> Similar Incident Search
        +--> Safety Pattern Analysis
        +--> Emerging Precursor Tool
        +--> Location Risk Tool
        |
        v
Traceable Historical Evidence
```

## RAG Pipeline

The retrieval pipeline includes:

1.  Historical report loader
2.  `RecursiveCharacterTextSplitter` with 500-character chunks and
    75-character overlap
3.  `sentence-transformers/all-MiniLM-L6-v2` embeddings
4.  Chroma vector store
5.  Similarity retriever with lightweight reranking
6.  Grounded evidence displayed with historical report IDs

The development vector store was built from 1,000 reports and produced
approximately 1,204 chunks.

## Tool Calling

SafeSight includes four LangChain-decorated tools with real Python
logic:

-   `search_similar_incidents`
-   `analyze_safety_patterns`
-   `get_emerging_precursors`
-   `get_location_risk`

LangGraph coordinates incident analysis and safety-officer question
routing.

## Dataset

The project uses an OSHA-style severe-injury report dataset covering
**January 2015 to November 2025**.

After cleaning:

-   **105,991** reports
-   **86,513** hospitalizations
-   **27,976** amputations
-   **35** loss-of-eye records
-   **81.62%** hospitalization rate in the selected historical dataset

The dataset is used for historical safety intelligence and evaluation
context. Severe-injury records are not presented as pure near-miss data.

## Technology Stack

-   Python 3.12
-   Streamlit
-   Pandas / NumPy
-   Plotly
-   LangGraph
-   LangChain
-   Google Gemini
-   ChromaDB
-   Hugging Face Sentence Transformers
-   Pydantic

## Project Structure

``` text
SafeSight_AI/
├── app.py
├── agents/
│   ├── report_parser.py
│   ├── risk_agent.py
│   ├── pattern_agent.py
│   ├── batch_processor.py
│   ├── pipeline.py
│   └── coordinator.py
├── tools/
│   ├── safety_tools.py
│   ├── tool_router.py
│   └── test_tools.py
├── rag/
│   ├── vector_store.py
│   └── safety_advisor.py
├── feedback/
│   └── feedback_manager.py
├── data/
├── vectorstore/
├── notebooks/
└── requirements.txt
```

## Running Locally

### 1. Create and activate a virtual environment

``` powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
```

### 2. Install dependencies

``` powershell
pip install -r requirements.txt
```

### 3. Configure Gemini

Create a `.env` file in the project root:

``` text
GEMINI_API_KEY=your_api_key_here
```

Never commit the `.env` file or API key.

### 4. Run SafeSight

``` powershell
streamlit run app.py
```

## Example Safety Advisor Questions

``` text
What are the risks in Texas?
What incidents are increasing recently?
Show common patterns in the dataset.
Show me incidents involving machinery during cleaning.
```

## Human-in-the-Loop Learning

SafeSight keeps the safety professional in control. When an officer
overrides an AI risk classification, the system records:

-   original risk
-   corrected risk
-   officer reason
-   timestamp

Recent corrections are inserted into subsequent risk-classification
prompts as few-shot examples.

## Responsible Use

SafeSight is a decision-support prototype. Emerging signals, hotspot
views, historical frequencies, and AI-generated recommendations require
professional safety review. They should not be interpreted as forecasts,
guarantees, or substitutes for workplace safety procedures and qualified
judgment.

## Hackathon Highlights

SafeSight demonstrates:

-   custom interactive dashboard UX
-   report parsing and risk-scoring agents
-   LangGraph orchestration
-   decorated deterministic tools
-   complete local RAG pipeline
-   traceable historical evidence
-   cross-report precursor detection
-   human-in-the-loop feedback
-   interactive filtering and hotspot analysis
-   investigation export
-   graceful failure handling
