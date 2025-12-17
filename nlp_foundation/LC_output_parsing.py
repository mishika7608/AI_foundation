from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
 
llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature=0
)

response = llm.invoke([
    HumanMessage(content="Ecplain LCEL in  one sentence")
])
print(response.content)