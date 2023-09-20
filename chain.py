from __future__ import annotations

from json import JSONDecodeError
from typing import Any

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.docstore.document import Document
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.prompts import PromptTemplate
from langchain.prompts.base import BasePromptTemplate
from langchain.pydantic_v1 import Field
from langchain.schema.language_model import BaseLanguageModel
from pydantic import Extra

from retriever import TimeImportanceSimilarityRetriever


class GAChain(Chain):
    prompt: BasePromptTemplate
    observation_prompt: BasePromptTemplate
    importance_chain: LLMChain
    llm: BaseLanguageModel
    retriever: TimeImportanceSimilarityRetriever
    default_importance = 3.0
    parser: RetryWithErrorOutputParser

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> list[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return ["input_text", "speaker"]

    @property
    def output_keys(self) -> list[str]:
        """Will always return text key.

        :meta private:
        """
        return ["text", "input_importance", "output_importance"]

    def memorize(self, observation: str, context: str) -> tuple[Document, float]:
        result = self.importance_chain.run(
            context=context,
            observation=observation,
        )
        prompt_value = self.importance_chain.prompt.format_prompt(
            context=context,
            observation=observation,
        )
        try:
            importance = self.parser.parse_with_prompt(result, prompt_value).rate
        except JSONDecodeError:
            importance = self.default_importance

        doc = Document(page_content=observation, metadata={"importance": importance})
        self.retriever.add_documents([doc])
        return doc, importance

    def get_context(self, query: str) -> str:
        docs = self.retriever.get_relevant_documents(query)
        docs = sorted(docs, key=lambda x: x.metadata["created_time"])
        context = "\n".join([doc.page_content for doc in docs])
        return context

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        observation = self.observation_prompt.format_prompt(**inputs).text
        context = self.get_context(observation)
        _, input_importance = self.memorize(observation, context)
        prompt_value = self.prompt.format_prompt(
            context=context,
            observation=observation,
            name=self.retriever.agent_name,
        )
        response = self.llm.generate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        context += "\n" + observation
        generated_text = response.generations[0][0].text
        observation = self.observation_prompt.format_prompt(
            speaker=self.retriever.agent_name, input_text=generated_text
        ).text
        _, output_importance = self.memorize(observation, context)

        if run_manager:
            run_manager.on_text("Log something about this run")

        return {
            "text": generated_text,
            "input_importance": input_importance,
            "output_importance": output_importance,
        }

    @property
    def _chain_type(self) -> str:
        return "Generative_Agents_Chain"
