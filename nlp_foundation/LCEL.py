from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

chat_template_books = ChatPromptTemplate.from_template('''Suggest three of best intermediate-level {programming language} books.
                                                       Answer only by listing the books.
                                         ''')
chat_template_projects = ChatPromptTemplate.from_template('''Suggest three interesting {programming language} projects under intermediate-level programmers.
                                                       Answer only by listing the projects.''')

chat_template_time = ChatPromptTemplate.from_template(
    '''
    I am an intermediate level programmer.
    Consider the following literature:{books}
    Also, consider the following projects: {projects}
    Roughly, how much time would it take me to complete literature and the projects?
    '''
)

chat = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature=0.8,
    max_tokens=500
)

string_parser = StrOutputParser()

chain_books = chat_template_books | chat | string_parser 
chain_projects = chat_template_projects | chat | string_parser 

chain_parallel = RunnableParallel({'books':chain_books,'prjects':chain_projects})
print(chain_parallel.invoke({'programming language':'Python'}))

chain_time1 = (RunnableParallel({'books':chain_books,
                                'projects':chain_projects})| chat_template_time | chat | string_parser)
chain_time2 = ({'books':chain_books,
                                'projects':chain_projects}| chat_template_time | chat | string_parser)

print(chain_time2.invoke({'programming language': 'python'}))

print(chain_time2.get_graph().print_ascii()) #takes less time than consequent invokes










# RUNNABLE AND RUNNABLE SEQUENCE CLASSES
# Prompt template, chat model, output parser are instances of runnable class
#chatPrompt Template strats with runnable inherits invoke, betch, stream
# Runnable - unit of work that can invoked, batched, streamed, transformed, composed(link together formulating change and change is also runnable) -> has 2 classes runnable and runnable sequence runnable pass through(to pipe chains)
# from dotenv import load_dotenv
# load_dotenv()
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough #passes inputs through without alteration

# RunnablePassthrough().invoke([1,2,3])
# RunnablePassthrough().invoke("HEllo World")

# chat_template_tools = ChatPromptTemplate.from_template('''What are five most important tools a {job title} needs?
#                                          Answer only by listing the tools.
#                                          ''')
# chat_template_strategy = ChatPromptTemplate.from_template('''Considering tools provided, develop a strategy for effectively learning and mastering them: {tools}''')
# chat = ChatGroq(
#     model = "llama-3.1-8b-instant",
#     temperature=0.8,
#     max_tokens=100
# )

# string_parser = StrOutputParser()
# chain_tools = chat_template_tools | chat | string_parser | {'tools':RunnablePassthrough()}
# chain_strategy = chat_template_strategy | chat | string_parser
# print(chain_tools.invoke({'job title': 'data scientist'}))
# print(chain_strategy.invoke({'tools': '''1. Python
# 2. R
# 3. Jupyter Notebook
# 4. Pandas
# 5. NumPy'''}))
# #merge chain_tool and strategy
# chain_combined = chain_tools | chain_strategy
# print(chain_combined.invoke({'job title': 'data scientist'}))
# chain_long = chat_template_tools | chat | string_parser | {'tools':RunnablePassthrough()} | chat_template_strategy | chat | string_parser #another way




# STREAMING
# Generator function - allow functions that behave like iterators, allowing us to loop over output - > give yield statement instead of return statements(standarad function )
# LCEL - piped components to form chains
# from dotenv import load_dotenv
# load_dotenv()
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate

# chat = ChatGroq(
#     model = "llama-3.1-8b-instant",
#     model_kwargs={ "seed": 365 } #produces same output for same input every time
#     temperature=0.8
# )

# chat_template = chat_template = ChatPromptTemplate.from_messages([('human','I have recently adopted a {pet} wich is a {breed}. Could you suggest several training tips?')])

# chain = chat_template | chat

# response = chain.stream({'pet':'dragon', 'breed':'night fury'})
# next(response)
# for i in response:
#     print(i.content, end = '')

# # BATCHING
# from dotenv import load_dotenv
# load_dotenv()
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate

# chat = ChatGroq(
#     model = "llama-3.1-8b-instant",
#     temperature=0.8
# )

# chat_template = chat_template = ChatPromptTemplate.from_messages([('human','I have recently adopted a {pet} wich is a {breed}. Could you suggest several training tips?')])

# chain = chat_template | chat

# # print(chain.invoke({'pet':'dog', 'breed':'pomerian'})) #invoke doesnt allow us to feed multiple inputs at once(more time consuming) -> batch runs invoke in parallel (less time consuming)


# print(chain.batch([{'pet':'dog', 'breed':'pomerian'},
#                    {'pet':'koala','breed':'brown koala'}])) # wall times- batch invoke time is less than both 1 invoke and 2 invoke 
# print(chain.invoke({'pet':'dog', 'breed':'pomerian'}))
# print(chain.invoke({'pet':'koala', 'breed':'brown koala'}))

#PIPING

# from dotenv import load_dotenv
# load_dotenv()
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import CommaSeparatedListOutputParser

# list_instructions = CommaSeparatedListOutputParser().get_format_instructions()
# list_instructions
# chat_template = ChatPromptTemplate.from_messages([('system', list_instructions),('human','I have recently adopted a {pet}. Could you suggest three {pet} names? \n')])
# print(chat_template.messages[0].prompt.template)

# chat = ChatGroq(
#     model = "llama-3.1-8b-instant",
#     temperature=0.8
# )


# list_output_parser = CommaSeparatedListOutputParser()
# chat_template_result = chat_template.invoke({'pet':'penguin'})
# chat_result = chat.invoke(chat_template_result)
# print(list_output_parser.invoke(chat_result))

# #  PIPE SYMBOL - links elements in expression language, output will be input for next component

# chain = chat_template | chat | list_output_parser

# print(chain.invoke({'pet':'dog'}))