import asyncio
import os
from dotenv import load_dotenv
from pprint import pprint  # <-- THÊM DÒNG NÀY

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

# Load environment variables from .env file
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# Cảnh báo bảo mật: Bạn nên lấy URI từ biến môi trường thay vì viết thẳng vào code
DB_URI = "redis://default:XcL37sTIeY5UqBD9K9FO3qwOa4pwIHt3@redis-16313.c57.us-east-1-4.ec2.redns.redis-cloud.com:16313"

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash") # Đã sửa lại model name

async def main():
    """Main function to run the asynchronous graph operations."""
    async with AsyncRedisSaver.from_conn_string(DB_URI) as checkpointer:
        
        async def call_model(state: MessagesState):
            response = await model.ainvoke(state["messages"])
            return {"messages": [response]}

        builder = StateGraph(MessagesState)
        builder.add_node("call_model", call_model)
        builder.add_edge(START, "call_model")

        graph = builder.compile(checkpointer=checkpointer)

        config = {
            "configurable": {
                "thread_id": "0" # Đảm bảo dùng cùng thread_id
            }
        }

        print("--- First conversation stream (with Google Gemini) ---")
        async for chunk in graph.astream(
            {"messages": [{"role": "user", "content": "hi! I'm 18 years old."}]},
            config,
            stream_mode="values"
        ):
            chunk["messages"][-1].pretty_print()

        # ==========================================================
        # THÊM ĐOẠN CODE NÀY VÀO ĐỂ IN CHECKPOINT
        # ==========================================================
        print("\n" + "="*40)
        print("✅ LẤY CHECKPOINT MỚI NHẤT TỪ REDIS")
        print("="*40)

        # Dùng checkpointer.aget để lấy checkpoint cuối cùng của thread_id "0"
        final_checkpoint = await checkpointer.aget(config)

        # Dùng pprint để in ra cho dễ nhìn
        pprint(final_checkpoint)
        # ==========================================================


if __name__ == "__main__":
    asyncio.run(main())