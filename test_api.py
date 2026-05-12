from openai import OpenAI

client = OpenAI(
    api_key="sk-da6ab4fb9490463a8c1e32617b0e2a5c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

response = client.chat.completions.create(
    model="qwen-turbo",
    messages=[{"role": "user", "content": "你好"}]
)
print(response.choices[0].message.content)