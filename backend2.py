from langgraph.graph import StateGraph,START,END
from langchain_nvidia_ai_endpoints import ChatNVIDIA 
from typing import TypedDict,List,Annotated
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,SystemMessage,BaseMessage
from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

from langgraph.checkpoint.sqlite import  SqliteSaver
import sqlite3
import os

import requests
load_dotenv()
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
search_tool=DuckDuckGoSearchRun(region='us-en')
#custom tool creation
@tool
def calculator(first_num=float,second_num=float,operation=str)->dict:
    '''perform for basic arithmetic operation
    supported operations:add,mul,sub,div
    '''
    try:
        if operation=='add':
            result=first_num + second_num
        if operation=='mul':
            result= first_num * second_num
        if operation=='sub':
            result= first_num - second_num
        if operation=='div':
            if second_num==0:
                return {'error':'division by zero is not allowed'}
            else:
                result=first_num/second_num
        else:
            {'error':'unsupported operation'}
        return {'first_num':first_num,'second_num':second_num,'operation':operation,'result':result}
    except Exception as e:
        return {'error':str(e)}

@tool
def get_stock_price(symbol:str)->dict:
    ''' fetch latest stock price for a given symbol (e.g:'ASPL','TSLA)
    using alpha vantage with API key in the URL.'''
    url=f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&interval=5min&apikey=ALPHAVANTAGE_API_KEY"
    r = requests.get(url)
    data = r.json()
    return data
llm = ChatNVIDIA(
  model="openai/gpt-oss-20b",
  api_key="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", 
  temperature=1,
  top_p=1,
  max_completion_tokens=4096,
)

tools=[search_tool,get_stock_price,calculator]


llm_with_tools=llm.bind_tools(tools)

class ChatState(TypedDict):

    messages:Annotated[List[BaseMessage],add_messages]

def chat_node(state: ChatState):

    messages=state['messages']

    response=llm_with_tools.invoke(messages)

    return {'messages':[response]}
conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)
checkpointer=SqliteSaver(conn=conn)
graph=StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
graph.add_node("tools",ToolNode(tools))
graph.add_edge(START, "chat_node")

graph.add_conditional_edges(
    "chat_node",
    tools_condition
)

graph.add_edge("tools", "chat_node")

workflow=graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)