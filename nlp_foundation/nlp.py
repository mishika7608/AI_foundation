# lanchain kernel -> load api key %dotenv-magic command -> messages(dictionary lsu: to specify promsts as system(direct model-defines purpose, persona), user, assistant , tool)

from langchain_community.chat_models import ChatOllama

llm = ChatOllama(
    model="qwen2:0.5b",
    temperature=0.9,
    num_predict=30
)

print("Streaming response:")
for chunk in llm.stream([
    {"role": "system", "content": "You are a witty, humorous assistant who always responds in a funny way."},
    {"role": "user", "content": "what is life?"}
]):
    print(chunk.content, end="", flush=True)