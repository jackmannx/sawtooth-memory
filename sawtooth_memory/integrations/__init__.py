"""
integrations — Framework bridges for Sawtooth Memory.

Install extras before importing:

    pip install "sawtooth-memory[langgraph]"   # LangGraph + LangChain core
    pip install "sawtooth-memory[langchain]"  # LangChain history adapter only

Public modules:

    sawtooth_memory.integrations.langgraph
        SawtoothLangGraphAdapter, build_sawtooth_graph, AgentState, ...
    sawtooth_memory.integrations.langchain_adapter
        SawtoothChatMessageHistory
"""

__all__: list[str] = []
