#!/usr/bin/env python3

from src.pdf_rag_pipeline import PDFRAGPipeline
from src.llm_pipeline import LLMPipeline
from src.telegram_rag_bot import TelegramRAGBot

def main():
    # Пути к файлам (обновите при необходимости)
    pdf_path = 'data/2 apple_10k.pdf'
    chunks_path = 'data/apple_chunks.pkl'
    stats_path = 'data/bot_stats.json'
    qa_log_path = 'data/bot_qa_log.json'

    # Создаем retrieval-пайплайн
    retrieval_pipeline = PDFRAGPipeline(pdf_path, chunk_size=1000, overlap=50)
    retrieval_pipeline.load_chunks(chunks_path)
    retrieval_pipeline.build_index()

    # Создаем LLM-пайплайн
    llm_pipeline = LLMPipeline(model_name="google/flan-t5-base", max_length=200, do_sample=False)

    # Запускаем Telegram-бота
    token = "7977851701:AAEbQ7K6_OOTos_F_zyLlZYUSJ8TzyJKB6w"
    bot = TelegramRAGBot(token, retrieval_pipeline, llm_pipeline, stats_path, qa_log_path)
    bot.run()

if __name__ == "__main__":
    main()
