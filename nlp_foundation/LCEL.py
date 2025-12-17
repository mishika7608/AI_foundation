from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
list_instructions = CommaSeparatedListOutputParser().get_format_instructions()
list_instructions
chat_template = ChatPromptTemplate.from_messages([('system', list_instructions),('human','I have recently adopted a {pet}. Could you suggest three {pet} names? \n')])
print(chat_template.messages[0].prompt.template)
chat = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature=0.8
)


list_output_parser = CommaSeparatedListOutputParser()
chat_template_result = chat_template.invoke({'pet':'penguin'})
chat_result = chat.invoke(chat_template_result)
print(list_output_parser.invoke(chat_result))

#  PIPE SYMBOL - links elements in expression language, output will be input for next component

chain = chat_template | chat | list_output_parser

print(chain.invoke({'pet':'dog'}))