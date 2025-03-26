import logging
import json
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler
from telegram import InlineKeyboardMarkup, InlineKeyboardButton

class TelegramRAGBot:
    """
    Класс для Telegram-бота, использующего Retrieval-Augmented Generation (RAG).

    При получении запроса бот:
      - Вызывает retrieval-пайплайн для получения релевантного контекста.
      - Передаёт запрос и извлечённый контекст в LLM для генерации ответа.
      - Отправляет сгенерированный ответ пользователю.
      - Собирает статистику запросов и сохраняет пары "вопрос–ответ" для дальнейшего обучения.
    """
    def __init__(self, token, retrieval_pipeline, llm_pipeline, stats_path, qa_log_path):
        """
        Инициализация бота.

        :param token: Telegram API-токен.
        :param retrieval_pipeline: Объект retrieval-пайплайна (например, экземпляр PDFRAGPipeline).
        :param llm_pipeline: Объект LLM-пайплайна (например, экземпляр LLMPipeline).
        :param stats_path: Путь для сохранения статистики (например, на Google Drive).
        :param qa_log_path: Путь для сохранения QA-лога.
        """
        self.token = token
        self.retrieval_pipeline = retrieval_pipeline
        self.llm_pipeline = llm_pipeline
        self.stats_path = stats_path
        self.qa_log_path = qa_log_path

        # Статистика: число запросов, правильных и неправильных ответов.
        self.stats = {"total": 0, "correct": 0, "incorrect": 0}
        # QA-лог: список записей, каждая запись содержит запрос, retrieval контекст, страницу, сгенерированный ответ и feedback.
        self.qa_log = []

        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.updater = Updater(token, use_context=True)
        self.dp = self.updater.dispatcher

        self.setup_handlers()

    def setup_handlers(self):
        """Настраивает обработчики команд и сообщений."""
        self.dp.add_handler(CommandHandler("start", self.start))
        self.dp.add_handler(CommandHandler("stats", self.stats_command))
        self.dp.add_handler(CommandHandler("savestats", self.save_stats_command))
        self.dp.add_handler(CommandHandler("saveqa", self.save_qa_log_command))
        self.dp.add_handler(MessageHandler(Filters.text & ~Filters.command, self.answer_query))
        self.dp.add_handler(CallbackQueryHandler(self.feedback_callback, pattern='^feedback_'))

    def start(self, update, context):
        """Обработчик команды /start."""
        update.message.reply_text("Привет! Отправь мне вопрос, и я постараюсь ответить с использованием модели RAG.")

    def stats_command(self, update, context):
        """Обрабатывает команду /stats и выводит текущую статистику."""
        stats_text = json.dumps(self.stats, indent=4, ensure_ascii=False)
        update.message.reply_text(f"Текущая статистика:\n{stats_text}")

    def save_stats_command(self, update, context):
        """Команда /savestats сохраняет статистику в файл."""
        self.save_stats(self.stats_path)
        update.message.reply_text(f"Статистика сохранена в {self.stats_path}")

    def save_qa_log_command(self, update, context):
        """Команда /saveqa сохраняет QA-лог в файл."""
        self.save_qa_log(self.qa_log_path)
        update.message.reply_text(f"QA-лог сохранен в {self.qa_log_path}")

    def answer_query(self, update, context):
        """Обрабатывает текстовые сообщения и возвращает сгенерированный ответ."""
        user_query = update.message.text
        try:
            # Получаем retrieval результат (лучший найденный чанк)
            retrieval_result = self.retrieval_pipeline.search_query(user_query, top_k=1)[0]
            context_text = retrieval_result.get("chunk", "")
            retrieved_page = retrieval_result.get("page", "Unknown")

            # Генерируем ответ с использованием LLM
            answer = self.llm_pipeline.generate_answer(user_query, context_text)
            reply_text = f"Ответ (retrieved from page {retrieved_page}):\n{answer}\n\nОцените, верный ли ответ:"

            # Обновляем статистику
            self.stats["total"] += 1

            # Добавляем запись в QA-лог (feedback пока None)
            qa_record = {
                "query": user_query,
                "retrieval_context": context_text,
                "retrieved_page": retrieved_page,
                "generated_answer": answer,
                "feedback": None
            }
            self.qa_log.append(qa_record)

            # Создаем inline-клавиатуру для обратной связи
            keyboard = [
                [
                    InlineKeyboardButton("Верный", callback_data="feedback_yes"),
                    InlineKeyboardButton("Неверный", callback_data="feedback_no")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            update.message.reply_text(reply_text, reply_markup=reply_markup)
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            update.message.reply_text("Произошла ошибка при обработке запроса. Попробуйте позже.")

    def feedback_callback(self, update, context):
        """Обрабатывает нажатия на кнопки обратной связи."""
        query = update.callback_query
        query.answer()  # обязательно вызываем
        feedback = query.data  # "feedback_yes" или "feedback_no"
        if feedback == "feedback_yes":
            self.stats["correct"] += 1
            feedback_value = "yes"
            response = "Спасибо за обратную связь! Рад, что ответ верный."
        else:
            self.stats["incorrect"] += 1
            feedback_value = "no"
            response = "Спасибо за обратную связь! Мы учтём это для улучшения."
        # Обновляем последний QA-запись, если feedback ещё не установлен
        if self.qa_log and self.qa_log[-1]["feedback"] is None:
            self.qa_log[-1]["feedback"] = feedback_value
        # Сохраняем статистику и QA-лог
        self.save_stats(self.stats_path)
        self.save_qa_log(self.qa_log_path)
        query.edit_message_reply_markup(reply_markup=None)
        query.message.reply_text(response)

    def save_stats(self, stats_path):
        """
        Сохраняет статистику (self.stats) в файл в формате JSON.

        :param stats_path: Путь для сохранения файла статистики.
        """
        try:
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=4)
            self.logger.info(f"Статистика сохранена в {stats_path}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении статистики: {e}")

    def load_stats(self, stats_path):
        """
        Загружает статистику из файла (JSON) и записывает в self.stats.

        :param stats_path: Путь к файлу статистики.
        :return: Загруженная статистика.
        """
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                self.stats = json.load(f)
            self.logger.info(f"Статистика загружена из {stats_path}")
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке статистики: {e}")
        return self.stats

    def save_qa_log(self, qa_log_path):
        """
        Сохраняет QA-лог (self.qa_log) в файл в формате JSON.

        :param qa_log_path: Путь для сохранения QA-лога.
        """
        try:
            with open(qa_log_path, "w", encoding="utf-8") as f:
                json.dump(self.qa_log, f, ensure_ascii=False, indent=4)
            self.logger.info(f"QA-лог сохранен в {qa_log_path}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении QA-лога: {e}")

    def load_qa_log(self, qa_log_path):
        """
        Загружает QA-лог из файла (JSON) и сохраняет в self.qa_log.

        :param qa_log_path: Путь к файлу QA-лога.
        :return: Загруженный QA-лог.
        """
        try:
            with open(qa_log_path, "r", encoding="utf-8") as f:
                self.qa_log = json.load(f)
            self.logger.info(f"QA-лог загружен из {qa_log_path}")
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке QA-лога: {e}")
        return self.qa_log

    def run(self):
        """Запускает бота (polling)."""
        self.updater.start_polling()
        self.updater.idle()
