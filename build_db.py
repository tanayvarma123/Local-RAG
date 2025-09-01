import os
from pathlib import Path
import argparse

from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


def load_docs(dir_path: Path):
    docs = []
    for p in dir_path.glob("*.docx"):
        docs.extend(Docx2txtLoader(str(p)).load())
    if not docs:
        raise FileNotFoundError(f"No .docx files found in: {dir_path}")
    return docs


def main():
    load_dotenv("key.env")  # safe even if OPENAI key isn't required here

    ap = argparse.ArgumentParser(description="Build FAISS index from .docx")
    ap.add_argument("--data", default="data", help="Folder with .docx files")
    ap.add_argument("--out", default="artifacts/faiss_langchain", help="Index output folder")
    ap.add_argument("--embed", default="sentence-transformers/all-MiniLM-L6-v2", help="HF embedding model")
    ap.add_argument("--chunk", type=int, default=1000, help="Chunk size (chars)")
    ap.add_argument("--overlap", type=int, default=100, help="Chunk overlap (chars)")
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_docs(data_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk,
        chunk_overlap=args.overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=args.embed,
        encode_kwargs={"normalize_embeddings": True},
    )

    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(str(out_dir))

    print(f"✓ Indexed {len(chunks)} chunks → {out_dir}")


if __name__ == "__main__":
    main()
