import os
import uuid
from typing import Optional

from app_lc import CliBot, Reply, system_prompt, STYLE, system_prompt
system_prompt += f""" 
Выдай в ответе answer: {STYLE['format']['fields']['answer']}, tone: {STYLE['format']['fields']['tone']} и actions: {STYLE['format']['fields']['tone']}."""

bot = CliBot(model_name=os.getenv("OPENAI_API_MODEL", "gpt-5"), system_prompt=system_prompt)

def ask(question: str, session_id: Optional[str] = None) -> Reply:
    global bot

    if session_id is None:
        session_id = str(uuid.uuid4())

    if question.startswith('/order'):
        order_id = question.split()[1]
        if order_id in bot.orders:
            order_info = bot.orders[order_id]['status']
            question += f'Ответь по статусу заказа используя информацию: {order_info}'
    try:
        response = bot.ask_structured(question, session_id=session_id)
        return response

    except Exception as e:
        return Reply(
            answer=f"Извините, произошла ошибка: {str(e)}",
            actions=[],
            tone="error"
        )