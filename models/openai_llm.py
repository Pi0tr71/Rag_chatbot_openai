from openai import OpenAI
import os

api_key_def = os.getenv("sk-...")

def openai_chat_completion(prompt: str, model: str = "gpt-3.5-turbo", api_key_usr = api_key_def) -> str:
    print("openai_chat_completion")
    client = OpenAI(api_key = api_key_usr)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content
