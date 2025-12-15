# lanchain kernel -> load api key %dotenv-magic command -> messages(dictionary lsu: to specify promsts as system(direct model-defines purpose, persona), user, assistant , tool)
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    FewShotChatMessagePromptTemplate
)

# 1️ Initialize Ollama chat model
chat = ChatOllama(
    model="qwen2:0.5b",
    temperature=0.0,
    num_predict=100
)

# 2️ Templates (same logic as yours)
TEMPLATE_H = "I've recently adopted a {pet}. Could you suggest some {pet} names"
TEMPLATE_AI = "{response}"

message_template_h = HumanMessagePromptTemplate.from_template(TEMPLATE_H)
message_template_ai = AIMessagePromptTemplate.from_template(TEMPLATE_AI)

# 3️ Example template (Human → AI)
example_template = ChatPromptTemplate.from_messages([
    message_template_h,
    message_template_ai
])

# 4️ Few-shot examples
examples = [
    {
        "pet": "dog",
        "response": (
            "Oh, absolutely. Because nothing screams "
            "'I'm a responsible pet owner' like asking a chatbot "
            "to name your furball. How about 'Bark Twain'?"
        )
    }
]

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_template
)

# 5️ Final chat prompt
chat_template = ChatPromptTemplate.from_messages([
    few_shot_prompt,
    message_template_h
])

# 6️ Invoke the prompt
messages = chat_template.format_messages(
    pet="rabbit"
)

response = chat.invoke(messages)

# 7️ Print result
print(response.content)
