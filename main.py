import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from faiss import IndexFlatIP
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from pydantic import BaseModel, Field

from chain import GAChain
from retriever import TimeImportanceSimilarityRetriever

load_dotenv()

IMPORTANCE_TEMPLATE = """\
あなたの名前は Elith bot です。
以下の Observation に示す内容の発言がありました。
この発言に関連する内容を Context に列挙します。
Observation の発言の重要度について 1 から 10 で客観的に評価してください。
目安として、1 は歯を磨く、ベッドを整えるなどの日常的な行為、10 は別れ、大学合格などの非常に重要な出来事とします。

Context: 
```{context}```
Observation: 
```{observation}```

出力は以下のJSONスキーマに準拠したJSONインスタンスとしてフォーマットしてください。

例として、スキーマ{{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}に対して、オブジェクト{{"foo": ["bar", "baz"]}}はスキーマの適切にフォーマットされたインスタンスです。しかし、オブジェクト{{"properties": {{"foo": ["bar", "baz"]}}}}は適切にフォーマットされていません。

今回用いるスキーマは以下の通りです。
{{"properties": {{"rate": {{"description": "重要度を表す 1 から 10 の整数", "title": "Rate", "type": "integer"}}}}, "required": ["rate"]}}

出力:
"""

OBSERVATION_TEMPLATE = "{speaker}: {input_text}"

TEMPLATE = """\
あなたの名前は {name} です。

Context として、関連する発言を示します。
Context: 
```{context}```

Context を考慮して以下の発言に対して、返答してください。
```{observation}```

{name}:
"""


def create_demo(chain: GAChain):
    def respond(speaker, message, chat_history, history):
        result = chain(dict(input_text=message, speaker=speaker))
        bot_message = result["text"]
        input_importance = result["input_importance"]
        output_importance = result["output_importance"]
        chat_history.append((message, bot_message))
        history["name"].append(speaker)
        history["importance"].append(input_importance)
        history["text"].append(message)
        history["name"].append(chain.retriever.agent_name)
        history["importance"].append(output_importance)
        history["text"].append(bot_message)

        return "", chat_history, history

    def create_df(history):
        df = pd.DataFrame(history)
        return df

    with gr.Blocks() as demo:
        name = gr.Text(label="あなたの名前", interactive=True)
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])
        history = gr.State(value={"name": [], "importance": [], "text": []})

        msg.submit(respond, [name, msg, chatbot, history], [msg, chatbot, history])
        with gr.Accordion(label="チャット履歴", open=False):
            button = gr.Button("履歴を取得")
            df = gr.DataFrame()
        button.click(create_df, inputs=history, outputs=df)

    demo.launch()


def main():
    llm_0 = ChatOpenAI(temperature=0)
    llm = ChatOpenAI()

    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    index = IndexFlatIP(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    retriever = TimeImportanceSimilarityRetriever(
        vectorstore=vectorstore,
        decay_rate=0.005,
        k=5,
        weights=dict(importance=0.1, similarity=1, time=1),
        agent_name="Elith bot",
    )

    importance_prompt = PromptTemplate(
        template=IMPORTANCE_TEMPLATE, input_variables=["context", "observation"]
    )
    importance_chain = LLMChain(llm=llm_0, prompt=importance_prompt)
    observation_prompt = PromptTemplate(
        template=OBSERVATION_TEMPLATE, input_variables=["speaker", "input_text"]
    )
    prompt = PromptTemplate(
        template=TEMPLATE, input_variables=["name", "context", "observation"]
    )

    class ImportanceRate(BaseModel):
        rate: int = Field(description="重要度を表す 1 から 10 の整数")

    parser = PydanticOutputParser(pydantic_object=ImportanceRate)
    retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=llm_0)

    chain = GAChain(
        prompt=prompt,
        observation_prompt=observation_prompt,
        retriever=retriever,
        llm=llm,
        importance_chain=importance_chain,
        parser=retry_parser,
    )

    create_demo(chain)


if __name__ == "__main__":
    main()
