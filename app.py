import os
import certifi
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
#from langchain_tools import langchain_tools
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor


#LOAD ENVIRONMENT VARIABLES
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
load_dotenv()
#print(os.getenv("TAVILY_API_KEY"))
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

## Create a search tool using TavilySearchResults

search_tool = TavilySearchResults(max_results=5)
result = search_tool.invoke("What is the latest news on AI?")


# Creating LLM

llm = ChatOpenAI(model="gpt-4o", temperature=1, openai_api_key=openai_api_key)

#response = llm.invoke("tell me a joke about AI")

# Creating a prompt with react agent
prompt = hub.pull("hwchase17/react")

#========================
#St Paging

st.set_page_config(
    page_title="LangChain Agent Demo", page_icon=":robot_face:", layout="centered", initial_sidebar_state="expanded")
st.title("LangChain Agent Demo")
st.markdown("This is a demo of a LangChain agent that uses the Tavily Search tool to answer questions")
#=====================




#creating a tools list for the agent
tools = [search_tool]

#creatomg agent and agent executor
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#run the agent executor with a query

response = agent_executor.invoke(
    {
        "input": ("what impact will AI have on the job market in the next 1 years?"),
     }

)

#===============UI INPUT=========================

user_input = st.text_input("Enter your question:",
                           placeholder= "What is the latest news on AI?")

#=================================================

#======================== RUN AGENT =========================

if st.button("Run Agent"):
    if user_input:
        with st.spinner("Running agent..."):
            try:
                response = agent_executor.invoke(
                    {
                        "input": user_input,
                    }
                )
                st.success("Agent completed!")
                st.write("Response:")
                st.write(response["output"])

            except Exception as e:
                st.error(f"Error running agent: {e}")
    else:
        st.warning("Please enter a question before running the agent.")
#==============================================================