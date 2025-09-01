import os
import argparse
import textwrap
from datetime import datetime

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI


def wrap(text: str, width: int = 88) -> str:
    return "\n".join(textwrap.fill(line, width=width) for line in text.splitlines())


def hr(width: int = 90, ch: str = "─") -> str:
    return ch * width


def panel(title: str, body: str, width: int = 90) -> str:
    return f" {title} ".center(width, "─") + "\n" + wrap(body, width) + "\n" + hr(width)


def snippet(text: str, max_len: int = 400) -> str:
    t = " ".join(text.split())
    return (t[: max_len - 1] + "…") if len(t) > max_len else t


def load_vectordb(index_dir: str, embed_model: str):
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_chain(retriever, model: str, temperature: float, max_tokens: int | None):
    prompt = ChatPromptTemplate.from_template(
        "Answer ONLY from the CONTEXT. If insufficient, say \"I don't know.\""
        "\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:"
    )
    llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens, timeout=60)
    return (
        RunnableParallel({"docs": retriever, "question": RunnablePassthrough()})
        | (lambda x: {"context": "\n\n---\n\n".join(d.page_content for d in x["docs"]), "question": x["question"]})
        | prompt
        | llm
        | StrOutputParser()
    )


def main():
    load_dotenv("key.env")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    ap = argparse.ArgumentParser(description="Query FAISS RAG (LangChain + OpenAI)")
    ap.add_argument("question", type=str, help="Your question")
    ap.add_argument("--index", default="artifacts/faiss_langchain", help="Index folder")
    ap.add_argument("--embed", default="sentence-transformers/all-MiniLM-L6-v2", help="HF embedding model")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="OpenAI model")
    ap.add_argument("--k", type=int, default=4, help="Top-k documents")
    ap.add_argument("--width", type=int, default=90, help="Console wrap width")
    ap.add_argument("--show-context", action="store_true", help="Print retrieved context preview")
    ap.add_argument("--temp", type=float, default=0.0, help="LLM temperature")
    ap.add_argument("--max-tokens", type=int, default=None, help="LLM max tokens")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Put it in key.env or your env.")

    vectordb = load_vectordb(args.index, args.embed)
    retriever = vectordb.as_retriever(search_kwargs={"k": args.k})
    chain = build_chain(retriever, args.model, args.temp, args.max_tokens)

    answer = chain.invoke(args.question).strip()
    docs_scores = vectordb.similarity_search_with_score(args.question, k=args.k)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"Query Time: {ts} | Model: {args.model} | Embeddings: {args.embed} | k={args.k}"

    lines = []
    for rank, (doc, score) in enumerate(docs_scores, start=1):
        src = doc.metadata.get("source", "unknown")
        extra = []
        if "page" in doc.metadata:
            extra.append(f"page {doc.metadata['page']}")
        if "file_path" in doc.metadata:
            extra.append(os.path.basename(doc.metadata["file_path"]))
        hint = f" ({', '.join(extra)})" if extra else ""
        lines.append(f"{rank}. {os.path.basename(src)}{hint}\n   score: {score:.4f}\n   text:  {snippet(doc.page_content)}")
    sources_block = "\n".join(lines) if lines else "No sources found."

    if args.show_context:
        ctx_docs = retriever.get_relevant_documents(args.question)
        ctx_preview = "\n\n---\n\n".join(snippet(d.page_content, 400) for d in ctx_docs)
        print(panel("CONTEXT (preview)", ctx_preview, width=args.width))

    print(panel("RAG QUERY", f"{header}\n\nQ: {args.question}", width=args.width))
    print(panel("ANSWER", answer if answer else "I don't know.", width=args.width))
    print(panel("SOURCES (ranked by similarity)", sources_block, width=args.width))


if __name__ == "__main__":
    main()
