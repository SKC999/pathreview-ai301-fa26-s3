"""Regression tests for issue #1: re-ingesting the same source must be skipped."""

import chromadb
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable

from core.models import IngestedSource
from ingestion.pipeline import IngestionPipeline

PROFILE_ID = "8f2c1d4e-0b6a-4c3e-9a51-2d7f0e6b9c10"
README = (
    "# demo-repo\n\nA small demo project for testing ingestion.\n\n"
    "## Install\n\nRun pip install demo.\n\n"
    "## Usage\n\nImport demo and call run().\n"
)


class CountingEmbedder:
    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[float(len(t) % 7), 1.0, 0.5] for t in texts]


@pytest.fixture
def pipeline_parts(request):
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        # Table only: the model declares each index twice, which SQLite rejects.
        conn.execute(CreateTable(IngestedSource.__table__))
    session = Session(engine)
    collection = chromadb.EphemeralClient().get_or_create_collection(
        f"dedup_{request.node.name}"
    )
    embedder = CountingEmbedder()
    pipeline = IngestionPipeline(
        vector_db=collection, db_session=session, embedding_provider=embedder
    )
    yield pipeline, session, collection, embedder
    session.close()


def _records(session):
    return session.execute(select(func.count()).select_from(IngestedSource)).scalar()


def test_reingesting_same_readme_is_skipped(pipeline_parts):
    pipeline, session, collection, embedder = pipeline_parts

    first = pipeline.ingest_readme(PROFILE_ID, "demo-repo", README)
    second = pipeline.ingest_readme(PROFILE_ID, "demo-repo", README)

    assert first.skipped is False
    assert second.skipped is True
    assert second.skip_reason == "Source already ingested"
    assert embedder.calls == 1
    assert collection.count() == first.chunk_count
    assert _records(session) == 1


def test_changed_readme_is_ingested_again(pipeline_parts):
    pipeline, session, _collection, embedder = pipeline_parts

    pipeline.ingest_readme(PROFILE_ID, "demo-repo", README)
    changed = pipeline.ingest_readme(PROFILE_ID, "demo-repo", README + "\n## Notes\n\nNew.\n")

    assert changed.skipped is False
    assert embedder.calls == 2
    assert _records(session) == 2


def test_same_readme_for_another_profile_is_not_skipped(pipeline_parts):
    pipeline, _session, _collection, embedder = pipeline_parts

    pipeline.ingest_readme(PROFILE_ID, "demo-repo", README)
    other = pipeline.ingest_readme(
        "0d9e4b7a-3c21-4f8e-b6a2-91c5e7f3d482", "demo-repo", README
    )

    assert other.skipped is False
    assert embedder.calls == 2
