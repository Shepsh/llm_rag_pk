import os
import json
import gzip
import logging
from pathlib import Path
from typing import List, Tuple
import pytz
from telegram.ext import JobQueue, Application, CommandHandler, MessageHandler, filters, ContextTypes

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ApplicationBuilder

# --- LangChain / Chroma ---
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_openai import OpenAIEmbeddings
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# ================== ЛОГИ ==================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("rag-bot")

# ================== ГЛОБАЛЫ ==================
RETRIEVER: MultiVectorRetriever | None = None

# ================== УТИЛИТЫ ==================
def chunk_text(s: str, limit: int = 4000) -> List[str]:
    """Telegram ограничивает ~4096 символов: режем по абзацам/строкам, чтобы не ломать формат."""
    if len(s) <= limit:
        return [s]
    parts, buf = [], []
    size = 0
    for line in s.split("\n"):
        if size + len(line) + 1 > limit:
            parts.append("\n".join(buf))
            buf, size = [line], len(line) + 1
        else:
            buf.append(line)
            size += len(line) + 1
    if buf:
        parts.append("\n".join(buf))
    return parts

def looks_like_html_table(s: str) -> bool:
    sl = (s or "").lower()
    return "<table" in sl and "</table>" in sl

# ================== ДЕСЕРИАЛИЗАЦИЯ ==================
def deserialize_mvretriever(
    persist_dir: str | Path,
    docstore_jsonl_gz: str | Path,
    meta_json: str | Path,
) -> MultiVectorRetriever:
    persist_dir = Path(persist_dir)
    docstore_jsonl_gz = Path(docstore_jsonl_gz)
    meta_json = Path(meta_json)

    with meta_json.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    id_key = meta.get("id_key", "doc_id")
    collection_name = meta["collection_name"]
    embedding_model = meta["embedding_model"]

    emb = OpenAIEmbeddings(model=embedding_model)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=emb,
        persist_directory=str(persist_dir),
    )

    store = InMemoryStore()
    pairs = []
    with gzip.open(docstore_jsonl_gz, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            d = Document(page_content=obj["page_content"], metadata=obj.get("metadata", {}))
            pairs.append((obj["doc_id"], d))
    if pairs:
        store.mset(pairs)

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    logger.info("Retriever restored: collection=%s, embed=%s, docs=%d",
                collection_name, embedding_model, len(pairs))
    return retriever

# ================== ПОИСК + СУММАРИЗАЦИЯ ==================
def search_summaries(vectorstore: Chroma, query: str, k: int = 6) -> List[Document]:
    return vectorstore.similarity_search(query, k=k)

def resolve_originals(mvr: MultiVectorRetriever, summaries: List[Document]) -> List[Document]:
    ids = [d.metadata.get("doc_id") for d in summaries if d.metadata.get("doc_id")]
    originals = mvr.docstore.mget(ids)
    return [o for o in originals if o is not None]

def format_text_block(doc: Document, max_len: int = 1200) -> str:
    src  = doc.metadata.get("source", "")
    page = doc.metadata.get("page", "?")
    kind = doc.metadata.get("type", "text")
    content = doc.page_content or ""
    if looks_like_html_table(content):
        # упрощённо: для краткости оставим как есть; можно распарсить bs4.get_text()
        pass
    if len(content) > max_len:
        content = content[:max_len] + " …"
    return f"[источник: {src} | стр. {page} | тип: {kind}]\n{content}"

def summarize_over_originals(
    originals: List[Document],
    user_query: str,
    model_name: str = "gpt-4o-mini",
    max_text_blocks: int = 6,
    max_images: int = 0,         # в Telegram тут оставим только текст (избежим отправки картинок)
    max_tokens: int = 650,
) -> Tuple[str, List[Tuple[str, int]]]:
    texts = [d for d in originals if (d.metadata or {}).get("type") in {"text", "table"}]
    imgs  = [d for d in originals if (d.metadata or {}).get("type") == "img"]

    blocks = [format_text_block(d) for d in texts[:max_text_blocks]]
    text_context = "\n\n---\n\n".join(blocks) if blocks else "(нет текста/таблиц)"

    header = (
        "Ты аналитик. Ответь по-русски кратко и структурированно.\n"
        "Используй ТОЛЬКО данные из приведённого контекста.\n"
        "Если приводишь цифры — используй значения из источников. "
        "В конце добавь список [источник, стр.]."
    )
    parts = [
        {"type": "text",
         "text": header + "\n\nЗапрос пользователя: " + user_query + "\n\nКонтекст:\n" + text_context}
    ]
    from langfuse.langchain import CallbackHandler

    langfuse_handler = CallbackHandler()
    chat = ChatOpenAI(
            model=model_name,
            temperature=0,
            max_tokens=max_tokens,
            callbacks = [langfuse_handler]
                        )
    msg = chat.invoke([HumanMessage(content=parts)])
    answer = msg.content

    cites = []
    for d in texts[:max_text_blocks] + imgs[:max_images]:
        cites.append((d.metadata.get("source", ""), int(d.metadata.get("page", 0) or 0)))
    # uniq + sort
    seen = set()
    uniq = []
    for c in sorted(cites, key=lambda x: x[1]):
        if c not in seen and c[0]:
            uniq.append(c); seen.add(c)

    if uniq:
        refs = "\nИсточники: " + "; ".join([f"[{src}, стр. {pg}]" for src, pg in uniq if pg])
        answer = answer + "\n\n" + refs

    return answer, uniq

def query_rag_system(query: str) -> str:
    global RETRIEVER
    if RETRIEVER is None:
        return "Индекс не загружен. Проверьте конфигурацию пути к индексу."
    # 1) поиск по summary в векторном хранилище
    summaries = search_summaries(RETRIEVER.vectorstore, query, k=8)
    if not summaries:
        return "По этому запросу ничего не нашлось."

    # 2) подтянуть оригиналы
    originals = resolve_originals(RETRIEVER, summaries)
    if not originals:
        return "Нашёл похожие сводки, но не смог загрузить оригиналы."

    # 3) суммаризировать по оригиналам
    answer, _ = summarize_over_originals(
        originals, user_query=query, model_name=os.getenv("OPENAI_RAG_MODEL", "gpt-4o-mini")
    )
    return answer

# ================== TELEGRAM ХЕНДЛЕРЫ ==================
load_dotenv()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Welcome to the RAG system! Ask me anything.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    user_id = update.message.from_user.id if update.message.from_user else None

    langfuse = context.application.langfuse  # если в main() сохранили в app

    try:
        with langfuse.start_as_current_observation(
            as_type="span",
            name="telegram-rag-query",
            trace_context={"user_id": user_id, "input": {"user_message": user_message}}
        ) as span:
            response = query_rag_system(user_message)

            span.update_trace(output={"answer": response, "status": "SUCCESS"})
            span.end()

            for part in chunk_text(response):
                await update.message.reply_text(part)
    except Exception as e:
        logger.exception("Error handling message")
        with langfuse.start_as_current_observation(
            as_type="span",
            name="telegram-rag-query-error",
            trace_context={"user_id": user_id, "input": {"user_message": user_message}}
        ) as span:
            span.end(output={"error": str(e), "status": "ERROR"})
        await update.message.reply_text("Sorry, something went wrong.")


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Update %s caused error %s", update, context.error)

def main() -> None:
    try:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            raise ValueError("No token found in environment variables")
        tz = pytz.timezone("Europe/Amsterdam")  # или pytz.UTC
        #jq = JobQueue(timezone=tz)

        # --- пути к индексу (Windows/Linux/Colab-стиль — всё ок) ---
        # Пропишите в .env или окружении:
        # RAG_PERSIST_DIR=/path/to/rag_index/mm_rag_cj_blog
        # RAG_DOCSTORE_PATH=/path/to/rag_index/docstore.jsonl.gz
        # RAG_META_PATH=/path/to/rag_index/meta.json
        persist_dir = os.getenv("RAG_PERSIST_DIR")
        docstore_path = os.getenv("RAG_DOCSTORE_PATH")
        meta_path = os.getenv("RAG_META_PATH")

        if not (persist_dir and docstore_path and meta_path):
            raise ValueError("Set RAG_PERSIST_DIR, RAG_DOCSTORE_PATH, RAG_META_PATH in env/.env")
        from langfuse import get_client

        langfuse = get_client()

        # Проверка подключения
        if langfuse.auth_check():
            print("Langfuse client is authenticated!")
        else:
            print("Authentication failed!")

        # Ключ OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is missing")

        # --- поднимаем ретривер один раз на старте ---
        global RETRIEVER
        RETRIEVER = deserialize_mvretriever(persist_dir, docstore_path, meta_path)

        # --- Telegram bot ---
        app = (
            ApplicationBuilder()
            .token(token)
            .build()
        )


        app.langfuse = langfuse
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        app.add_error_handler(error_handler)

        logger.info("Starting bot…")
        app.run_polling(drop_pending_updates=True)

    except Exception as e:
        logger.exception("Bot startup failed")

if __name__ == "__main__":
    main()
