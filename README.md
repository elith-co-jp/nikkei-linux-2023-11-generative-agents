# 日経Linux2023年11月号LLMによる生成エージェント特集サポートページ
## 概要
2023年10月6日に日経BPから出版される**日経Linux2023年11月号**のLLM特集の「LLMによる生成エージェントの世界」特集のサポートページです。 

<img src="https://m.media-amazon.com/images/I/81AR8Bmk27L._SL1390_.jpg" alt="日経Linux 2023年 11 月号" width="600" height="450" />


## 学習コード
### 動作環境
- デモはGradioを用いて行うことができます。
- ローカルのPCで動作確認済みです。
- LangChainとOpenAIのAPIを利用しているため、APIキーを用意する必要があります。
 
### OpenAIのAPIキーの認識方法
- リポジトリの直下に `.env`ファイルを作成し利用してください。

```
OPENAI_API_KEY="<自身のAPIキー>"
```

### 記憶方法（メモリストリーム)
- 重要度・関連性・最新性の3つの観点で生成エージェントが検索スコアを算出し、それによってメッセージを作成します。

### 利用方法
- `pyproject.toml`は以下のようなライブラリとバージョンで構成されています。

```pyproject.toml 
[tool.poetry]
name = "generative_agents"
version = "0.1.0"
description = ""
authors = []
readme = "README.md"

[tool.poetry.dependencies]
python = ">=3.9,<3.12"
langchain = "^0.0.295"
openai = "^0.28.0"
notebook = "^7.0.3"
python-dotenv = "^1.0.0"
tiktoken = "^0.5.1"
faiss-cpu = "^1.7.4"
gradio = "^3.44.4"
```

- 自身でライブラリをインストールするか`pip`コマンドを利用してインストールしてください

- `main.py`を動作させることでデモ画面が立ち上がります。

### ライセンス
- GNU Affero General Public License v3.0ライセンスです。

## 参考文献・注釈
記事内で参照した参考文献です。記事内では `＊3` と記載があれば、以下の `＊3` の参考文献や注釈を参照しています。

- ＊1 https://arxiv.org/abs/2304.03442
- ＊2 ザ・シムズ（The Sims）」は仮想の人々の日常生活を制御・管理するライフシミュレーションゲームで、プレイヤーはキャラクターの家、仕事、関係を築いていきます。その豊富なカスタマイズと拡張パックで非常に人気があります。
- ＊3 https://github.com/langchain-ai/langchain
- ＊4 https://www.inworld.ai/   
- ＊5 https://reverie.herokuapp.com/arXiv_Demo/



