from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
# from langchain.openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

model=ChatOllama(
    model="llama3.2:latest",
    temperature=0,
)
server_params=StdioServerParameters(
    command="npx",
    env={
        "FIRECRAWL_API_KEY":os.getenv("FIRECRAWL_API_KEY")
    },
    args=["firecrawl-mcp"]
)

async def main():
    async with stdio_client(server_params) as (read,write):
        async with ClientSession(read,write) as session:
            await session.initialize()
            tools=await load_mcp_tools(session)
            agent=create_react_agent(model,tools)
            messages=[
                {
                    "role":"system",
                    "content":"You are a helpful assistant that can scrape websites,crawl pages and extract data using firecrawl tools"
                }
            ]

            print("Available tools-",*[tools.name for tools in tools])
            print("-"*60)
            while True:
                user_input=input("\nYou: ")
                if user_input == "quit":
                    print("Goodbye")
                    break

                messages.append({
                    "role":"user",
                    "content":user_input
                })

                try:
                    agent_response=await agent.ainvoke({
                        "messages":messages
                    })
                    ai_message=agent_response["messages"][-1].content
                    print("\nAgent: ",ai_message)

                except Exception as e:
                    print("Error:",e)

if __name__=="__main__":
    asyncio.run(main())

