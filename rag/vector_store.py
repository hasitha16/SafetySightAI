from pathlib import Path

import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "safesight_dev_1000.csv"
)

VECTORSTORE_DIR = (
    PROJECT_ROOT
    / "vectorstore"
    / "safety_reports"
)


# --------------------------------------------------
# 1. DOCUMENT LOADER
# --------------------------------------------------

def load_report_documents(limit=1000):
    """
    Load safety reports from CSV and convert them into
    LangChain Document objects.
    """

    df = pd.read_csv(
        DATA_FILE,
        dtype={"industry_code": str},
        low_memory=False
    )

    df = df.head(limit)

    documents = []

    for _, row in df.iterrows():

        report_text = str(row["report_text"])

        searchable_text = f"""
        Safety report: {report_text}
        Incident type: {row.get("event_type", "Unknown")}
        Injury type: {row.get("injury_nature", "Unknown")}
        Body part: {row.get("body_part", "Unknown")}
        Incident source: {row.get("incident_source", "Unknown")}
        """.strip()

        metadata = {
            "report_id": str(row["report_id"]),
            "event_date": str(row["event_date"]),
            "city": str(row["city"]),
            "state": str(row["state"]),
            "industry_code": str(row["industry_code"]),
            "event_type": str(row.get("event_type", "Unknown")),
            "injury_nature": str(row.get("injury_nature", "Unknown")),
            "incident_source": str(row.get("incident_source", "Unknown"))
        }

        document = Document(
            page_content=searchable_text,
            metadata=metadata
        )

        documents.append(document)

    return documents


# --------------------------------------------------
# 2. TEXT SPLITTER
# --------------------------------------------------

def split_documents(documents):
    """
    Split reports into retrieval-friendly chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75
    )

    return splitter.split_documents(documents)


# --------------------------------------------------
# 3. EMBEDDING MODEL
# --------------------------------------------------

def get_embedding_model():
    """
    Use a local sentence-transformer model.
    No Gemini API calls are required.
    """

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# 4. VECTOR STORE
# --------------------------------------------------

def build_vector_store(limit=1000):
    """
    Build and persist the SafeSight Chroma vector database.
    """

    print("\nLoading safety reports...")

    documents = load_report_documents(limit)

    print(f"Loaded documents: {len(documents)}")

    chunks = split_documents(documents)

    print(f"Created chunks: {len(chunks)}")

    print("\nLoading local embedding model...")

    embeddings = get_embedding_model()

    print("Creating Chroma vector store...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_name="safesight_reports"
    )

    print("\nVector store created successfully.")

    return vector_store


# --------------------------------------------------
# 5. RETRIEVER
# --------------------------------------------------

def load_vector_store():

    embeddings = get_embedding_model()

    return Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
        collection_name="safesight_reports"
    )


def retrieve_similar_reports(query, k=5):
    """
    Retrieve safety reports most relevant to a query.
    """

    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )

    return retriever.invoke(query)


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    build_vector_store(limit=1000)

    print("\n====================================")
    print("       SafeSight RAG Test")
    print("====================================")

    query = (
        "workers injured by machinery "
        "during maintenance"
    )

    print(f"\nQuery: {query}")

    results = retrieve_similar_reports(
        query,
        k=5
    )

    for number, document in enumerate(
        results,
        start=1
    ):

        print(f"\n--- Result {number} ---")

        print(
            f"Report ID: "
            f"{document.metadata.get('report_id')}"
        )

        print(
            f"Location: "
            f"{document.metadata.get('city')}, "
            f"{document.metadata.get('state')}"
        )

        print(
            document.page_content[:500]
        )