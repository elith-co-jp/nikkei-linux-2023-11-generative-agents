from __future__ import annotations

import datetime
from copy import deepcopy
from typing import Any, TypeVar

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.pydantic_v1 import Field
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores.base import VectorStore

Time = TypeVar("Time", int, float, datetime.datetime)


class TimeImportanceSimilarityRetriever(BaseRetriever):
    vectorstore: VectorStore
    agent_name: str
    search_kwargs: dict = Field(default_factory=lambda: dict(k=100))
    memory_stream: list[Document] = Field(default_factory=list)
    decay_rate: float = Field(default=0.01)
    weights: dict[str, float] = Field(
        default_factory=lambda: dict(similarity=1.0, time=1.0, importance=1.0)
    )
    k: int = 4
    use_datetime: bool = False
    current_time: int = 0

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self.use_datetime:
            current_time = datetime.datetime.now()
        else:
            current_time = self.current_time

        docs_scores = self._get_combined_score(query, current_time=current_time)
        sorted_docs_scores = sorted(docs_scores, key=lambda x: x[1], reverse=True)
        documents = [doc for doc, _ in sorted_docs_scores[: self.k]]

        # update last_access_time
        for document in documents:
            document.metadata["last_access_time"] = current_time
        return documents

    def _get_combined_score(
        self, query: str, current_time: Time
    ) -> list[tuple[Document, float]]:
        """時間・重要度・類似度を合算したスコアを計算する

        Args:
            query (str): クエリテキスト
            current_time (Time): 現在時刻

        Returns:
            list(tuple[Document, float]): ドキュメントとスコアのリスト
        """
        docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query, **self.search_kwargs
        )
        docs_total_scores = []
        for doc, similarity in docs_and_scores:
            time_passed = self._get_time_passed(
                current_time, doc.metadata["last_access_time"]
            )
            score_time = (1.0 - self.decay_rate) ** time_passed
            score_importance = doc.metadata["importance"]
            total_score = (
                self.weights["similarity"] * similarity
                + self.weights["time"] * score_time
                + self.weights["importance"] * score_importance
            )
            docs_total_scores.append((doc, total_score))
        return docs_total_scores

    def _get_time_passed(self, current_time: Time, last_access_time: Time) -> Time:
        """時間の差を計算する

        Args:
            current_time (Time): 現在時刻
            last_access_time (Time): 最終アクセス時刻

        Returns:
            Time: 時間の差
        """
        if isinstance(current_time, datetime.datetime):
            hours_passed = (current_time - last_access_time).total_seconds() / 3600
            return hours_passed
        return current_time - last_access_time

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Add documents to vectorstore."""

        if self.use_datetime:
            current_time = datetime.datetime.now()
        else:
            current_time = self.current_time

        docs = deepcopy(documents)
        for doc in docs:
            doc.metadata["last_access_time"] = current_time
            doc.metadata["created_time"] = current_time
            if "importance" not in doc.metadata:
                doc.metadata["importance"] = 0
            self.current_time += 1

        return self.vectorstore.add_documents(docs, **kwargs)
