from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_ollama import ChatOllama

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

BASE = Path(__file__).parent
style_file_path = BASE / "data" / "style_guide.yaml"

with open(style_file_path, 'r', encoding='utf-8') as f:
    STYLE = yaml.safe_load(f)

system_prompt = f"""
Ты полезный ассистент поддержки магазина {os.getenv('BRAND_NAME')}. 
Используй следующий тон разговора: {STYLE['tone']['persona']}.
Избегай в ответах: {STYLE['tone']['avoid']}. Ответ должен содержать: {STYLE['tone']['must_include']}.
Если у тебя нет информации для ответа отвечай: {STYLE['fallback']['no_data']}
"""


class JSONLFormatter(logging.Formatter):
    """Кастомный форматтер для JSONL логов"""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        if hasattr(record, 'session_id'):
            log_obj['session_id'] = record.session_id
        if hasattr(record, 'user_input'):
            log_obj['user_input'] = record.user_input
        if hasattr(record, 'bot_response'):
            log_obj['bot_response'] = record.bot_response
        if hasattr(record, 'usage'):
            log_obj['usage'] = record.usage
        if hasattr(record, 'type'):
            log_obj['type'] = record.type

        for key, value in record.__dict__.items():
            if key not in ['args', 'asctime', 'created', 'exc_info', 'exc_text',
                           'filename', 'funcName', 'levelname', 'levelno', 'lineno',
                           'module', 'msecs', 'msg', 'name', 'pathname', 'process',
                           'processName', 'relativeCreated', 'stack_info', 'thread',
                           'threadName', 'message'] and not key.startswith('_'):
                if key not in log_obj:
                    log_obj[key] = value

        return json.dumps(log_obj, ensure_ascii=False)

class Reply(BaseModel):
    answer: str
    actions: List[str]
    tone: str

class CliBot:
    def __init__(self, model_name, system_prompt=system_prompt):
        self.current_dir = Path(__file__)
        Path("logs").mkdir(exist_ok=True)
        full_system_prompt = self.create_full_system_prompt(system_prompt)
        self.chat_model = ChatOpenAI(
            model_name=model_name,
            temperature=0.7,
            request_timeout=15,
        )
        self.structured_model = self.chat_model.with_structured_output(Reply)
        self.orders = self.load_orders()
        self.store = {}
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", full_system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        self.chain = self.prompt | self.chat_model
        self.structured_chain = self.prompt | self.structured_model
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )
        self.model_name = model_name

    def create_full_system_prompt(self, system_prompt):
        faq_context = self.load_faq()
        faq_base = "ЧАСТО ЗАДАВАЕМЫЕ ВОПРОСЫ (FAQ):\n\n"
        for item in faq_context:
            faq_base += f"Вопрос: {item.get('q')}\n"
            faq_base += f"Ответ: {item.get('a')}\n"
            faq_base += "\n"
        faq_base += "Используй эту информацию при ответе на вопросы пользователей."
        return f'{system_prompt}\n\n{faq_base}'

    def load_faq(self):

        faq_path = self.current_dir.parent / 'data' / 'faq.json'
        with open(faq_path, 'r', encoding='utf-8') as f:
            faq_data = json.load(f)
        return faq_data

    def load_orders(self):
        orders_path = self.current_dir.parent / 'data' / 'orders.json'
        with open(orders_path, 'r', encoding='utf-8') as f:
            orders_data = json.load(f)
        return orders_data

    def ask_structured(self, question: str, session_id: str) -> Optional[Reply]:
        history = self.get_session_history(session_id)
        config = {"configurable": {"session_id": session_id}}
        response = self.structured_chain.invoke(
            {"question": question, 'history': history.messages},
            config=config
        )
        history.add_user_message(question)
        history.add_ai_message(response.answer)
        return response


    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def setup_session_logging(self, session_id: str) -> logging.Logger:

        session_logger = logging.getLogger(f'ShoplyBot.session_{session_id}')
        session_logger.setLevel(logging.INFO)
        session_logger.handlers.clear()
        session_logger.propagate = False
        # Обработчик для файла сессии
        session_file = f'logs/session_{session_id}.jsonl'
        session_handler = logging.FileHandler(session_file, encoding='utf-8', mode='a')
        session_handler.setFormatter(JSONLFormatter())
        session_logger.addHandler(session_handler)

        session_logger.info('Session started', extra={
            'type': 'session_start',
            'session_id': session_id,
            'model': self.model_name
        })

        return session_logger

    def __call__(self, session_id):
        logger = self.setup_session_logging(session_id)
        print(
            "Чат-бот запущен! Можете задавать вопросы. \n - Для выхода введите 'выход'.\n - Для очистки контекста введите 'сброс'.\n")

        while True:
            try:
                user_text = input("Вы: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nБот: Завершение работы.")
                break

            if not user_text:
                continue

            msg = user_text.lower()

            if msg in ("выход", "стоп", "конец"):
                print("Бот: До свидания!")
                logger.info('Session ended', extra={
                    'type': 'session_end',
                    'session_id': session_id
                })
                break

            if msg.startswith('/order'):
                if len(msg.split()) < 2:
                    print('Не передан order_id')
                    continue
                order_id = msg.split()[1]
                if order_id in self.orders:
                    print(self.orders[order_id]['status'])
                else:
                    print('Не найден передаваемый order_id')
                continue

            # Получаем структурированный ответ
            response = self.chain_with_history.invoke(
                {"question": user_text},
                {"configurable": {"session_id": session_id}}
            )

            if response:
                bot_reply = response.content
                print('Бот:', bot_reply, "\n")
                logger.info('Chat interaction', extra={
                    'type': 'chat_interaction',
                    'user_input': msg,
                    'bot_response': bot_reply,
                    'session_id': session_id
                })
            else:
                print('Бот: Извините, произошла ошибка при обработке запроса.')

if __name__ == "__main__":
    bot = CliBot(model_name=os.getenv("OPENAI_API_MODEL", "gpt-5"))
    bot("user_1231")
