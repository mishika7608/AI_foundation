from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime

llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature=0
)

csv_parser = CommaSeparatedListOutputParser()

csv_prompt = ChatPromptTemplate.from_messages([
    ("system", "Return ONLY comma-seperated values."),
    ("human", "List 5 cute bird names")
])

chain = csv_prompt | llm | csv_parser
csv_result = chain.invoke({})
print("Parsed CSV:",csv_result)
print("Type:",type(csv_result))

date_parser = StrOutputParser()

date_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a strict formatter. Return ONLY a date in ISO-8601 format (YYYY-MM-DD).No text."),
    ("human", "When did the first recorded volcano erupt?")
])

chain = date_prompt | llm | date_parser
date_result = chain.invoke({})

parsed_date = datetime.fromisoformat(date_result)
print("Parsed CSV:",parsed_date)
print("Type:",type(parsed_date))