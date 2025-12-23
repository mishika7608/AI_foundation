# Langraph - helps build stateful, multi step application powered by LLMs eg: GPT(openai), Claude(Anthropic)
# allows building flows involving decisions, memory, loops with graph based structure 
# Agent that thinks, remembers and acts

# Graph- defines application flow. A structured map of how information moves and how tasks are executed
# Nodes- user defd python functions that perform a specific tasks
# Edges- defines data flow from 1 node to other(python function that decides wi=hich node to go on next)
# Conditional edges - Yes/No response from user. Dynamic and adaptable applications
# State- users input, assisstant's response, coversation history, retrieved docs, tool outputs. Defined by schema(expected fields,datatypes,optional fields) as dictionary
# Super step- multiple nodes aligned together. multiple node can be executed in parallel
# Active node- if it recieves input. determined by edges leading into it

from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import START, END, StateGraph #StateGraph-our defd graph
from typing_extensions import TypedDict  #most used objects for defining schema of graph. Allow us to define dictionaries with explicitly declared keys. Type checkers will flag TypedDict with wnexpected keys. not enforced a runtime
from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import Runnable
from collections.abc import Sequence

class State(TypedDict):
    messages : Sequence[BaseMessage] #Sequence[BaseMessage] is expected datatype to store in messages

state = State(messages = [HumanMessage("Could you suggest me a book by Carolyn Keene?")])
# print(state)
state["messages"][0].pretty_print()

chat = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature=0.8,
    max_tokens=120
)

response  = chat.invoke(state["messages"])
response.pretty_print()

def chatbot(state: State) -> State: #pass parameter of type state and return state
    print(f"\n--------->ENTERING Chatbot: ")
    response = chat.invoke(state["messages"])
    response.pretty_print()

    return State(messages = [response])

chatbot(state)

graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot",END)
graph_compiled = graph.compile() # a runnable object
print(isinstance(graph,Runnable)) #false
print(isinstance(graph_compiled,Runnable)) #true
print(graph_compiled)
