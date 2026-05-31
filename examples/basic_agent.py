import asyncio
from sawtooth_memory import ContextManager
from sawtooth_memory.config import SawtoothConfig, OllamaConfig
# Imported for visibility; ContextManager handles the worker lifecycle automatically via 'async with'
from sawtooth_memory.worker import CompressionWorker 

async def main():
    # 1. Initialize configuration (using local Ollama for the example)
    config = SawtoothConfig(
        ollama=OllamaConfig(model="llama3"),
        fallback_truncate=True
    )
    
    system_prompt = "You are a highly capable AI assistant."
    
    # 2. Instantiate the ContextManager
    # The 'async with' block automatically spins up the background CompressionWorker
    # and gracefully shuts it down when the block exits.
    async with ContextManager(system_prompt=system_prompt, config=config) as cm:
        
        print("--- Ingesting Messages ---")
        # Simulate a conversation
        await cm.add_message("user", "Hello! My database connection ID is db_prod_994.")
        await cm.add_message("assistant", "I have noted your connection ID.")
        await cm.add_message("user", "Actually, switch that to db_staging_112.")
        
        # In a real scenario, the CompressionWorker evaluates the sliding window 
        # in the background and extracts entities to the L1.5 Ledger.
        
        # For demonstration, we manually upsert an entity to show L1.5 state behavior
        cm.state.l1_5_entities.upsert({"connection_id": "db_prod_994"})
        cm.state.l1_5_entities.upsert({"connection_id": "db_staging_112"})
        
        print("\n--- Current Memory State ---")
        
        # 3. Inspect L1 Working Memory
        print("\n[L1 Working Memory - Sliding Window]")
        for msg in cm.state.l1_working.messages:
            print(f"[{msg.role.upper()}]: {msg.content}")
            
        # 4. Inspect L1.5 Entity Ledger (Notice the conflict resolution list)
        print("\n[L1.5 Entity Ledger - Exact Extractions]")
        for key, history in cm.state.l1_5_entities.entities.items():
            print(f"Key: '{key}' -> Historical Timeline: {history}")
            
        print("\n[Compiled Prompt Injection]")
        print(cm.state.l1_5_entities.to_json_str())

if __name__ == "__main__":
    asyncio.run(main())