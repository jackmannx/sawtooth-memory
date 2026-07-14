import asyncio
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from sawtooth_memory import ContextManager, ContextManagerConfig, OllamaConfig
from sawtooth_memory.integrations.langgraph import SawtoothLangGraphAdapter


# Define standard LangGraph state
class State(TypedDict):
    messages: Annotated[list, add_messages]


async def main():
    # Initialize Sawtooth memory components
    config = ContextManagerConfig(ollama=OllamaConfig(model="llama3"))
    cm = ContextManager(
        system_prompt="You are a helpful workflow agent.", config=config
    )

    # 1. Create the adapter, passing in the shared ContextManager
    adapter = SawtoothLangGraphAdapter(context_manager=cm)

    # 2. Define a basic LangGraph node
    async def chat_node(state: State):
        # A. Sync the LangGraph state into Sawtooth (safely deduplicated)
        await adapter.sync_state(state["messages"])

        # B. Retrieve the compiled, compressed, and sanitized prompt
        # This strips orphaned ToolMessages and injects L1.5/L2 memory
        safe_prompt = await adapter.get_compiled_prompt()

        print(
            f"\n[Node Execution] Compiled Prompt contains {len(safe_prompt)} messages."
        )

        # C. Normally you would pass `safe_prompt` to your LLM here:
        # response = await llm.ainvoke(safe_prompt)

        # For this example, we mock the LLM response
        mock_response = AIMessage(content="I have processed your request.")
        return {"messages": [mock_response]}

    # 3. Build and compile the graph
    workflow = StateGraph(State)
    workflow.add_node("agent", chat_node)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    graph = workflow.compile()

    # 4. Run the graph within the ContextManager's async lifecycle
    async with cm:
        print("--- Starting LangGraph Session ---")

        inputs = {"messages": [HumanMessage(content="Initialize deployment sequence.")]}

        async for event in graph.astream(inputs):
            for node, values in event.items():
                print(f"Node '{node}' completed.")
                print(f"Latest output: {values['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
