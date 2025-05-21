import os
import io
import uuid
import logging
import requests

from fastapi import FastAPI, Request
from telegram import Bot, Update
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Filters, CallbackContext, Defaults
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from pdfminer.high_level import extract_text

# Логирование
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ENV vars
QDRANT_URL    = os.getenv("QDRANT_URL")
QDRANT_API_KEY= os.getenv("QDRANT_API_KEY")
HF_TOKEN      = os.getenv("HF_API_TOKEN")
TG_TOKEN      = os.getenv("TELEGRAM_BOT_TOKEN")
COLLECTION    = os.getenv("QDRANT_COLLECTION", "documents")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 256))
TOP_K         = int(os.getenv("TOP_K", 3))
HF_MODEL      = os.getenv("HF_MODEL", "google/flan-t5-small")

for v in [QDRANT_URL, QDRANT_API_KEY, HF_TOKEN, TG_TOKEN]:
    if not v:
        log.error("Missing one of ENV vars")
        raise RuntimeError("Missing ENV vars")

app = FastAPI()
bot = Bot(token=TG_TOKEN, defaults=Defaults(parse_mode="HTML"))
dispatcher = Dispatcher(bot, None, workers=0, use_context=True)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text) and text[end] != " ":
            end = text.rfind(" ", start, end) or end
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]

def ingest_handler(update: Update, context: CallbackContext):
    doc = update.message.document
    if doc.mime_type != "application/pdf":
        return update.message.reply_text("Пожалуйста, отправьте PDF.")
    f = io.BytesIO()
    bot.get_file(doc.file_id).download(out=f)
    text = extract_text(f)
    if not text.strip():
        return update.message.reply_text("Не удалось извлечь текст из PDF.")
    doc_id = str(uuid.uuid4())
    chunks = chunk_text(text)
    embs = embedder.encode(chunks).tolist()
    points = [
        PointStruct(id=None, vector=embs[i],
                    payload={"doc_id": doc_id, "chunk_index": i, "text": chunks[i]})
        for i in range(len(chunks))
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)
    update.message.reply_text(f"✅ Залито {len(chunks)} чанков (ID={doc_id}).")

def query_handler(update: Update, context: CallbackContext):
    query = update.message.text
    q_emb = embedder.encode([query])[0].tolist()
    hits = qdrant.search(collection_name=COLLECTION, query_vector=q_emb, limit=TOP_K)
    if not hits:
        return update.message.reply_text("Пока нет данных для поиска.")
    context_chunks = [h.payload["text"] for h in hits]
    prompt = (
        "Answer the question based only on the following context:\n"
        f"{'\\n'.join(context_chunks)}\n\n"
        f"Question: {query}\nAnswer:"
    )
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    body = {"inputs": prompt, "parameters": {"max_new_tokens": 128}}
    resp = requests.post(f"https://api-inference.huggingface.co/models/{HF_MODEL}",
                         headers=headers, json=body)
    try:
        ans = resp.json()[0]["generated_text"].strip()
    except Exception:
        log.error("HF API error: %s", resp.text)
        return update.message.reply_text("Ошибка при генерации.")
    used = "\n\n".join(
        f"<b>Chunk {i+1} (score={h.score:.3f}):</b>\n{h.payload['text']}"
        for i, h in enumerate(hits)
    )
    reply = f"🤖 <b>Ответ:</b>\n{ans}\n\n📄 <b>Использованные чанки:</b>\n{used}"
    update.message.reply_text(reply)

dispatcher.add_handler(CommandHandler("start", lambda u,c: u.message.reply_text(
    "Привет! Отправь PDF или задай вопрос.")))
dispatcher.add_handler(MessageHandler(Filters.document.mime_type("application/pdf"), ingest_handler))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, query_handler))

@app.post("/")
async def webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, bot)
    dispatcher.process_update(update)
    return {"ok": True}
