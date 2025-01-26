import os

from llama_stack.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types import Document


def create_library_client(template="ollama"):
    client = LlamaStackAsLibraryClient(template)
    client.initialize()
    return client


def create_http_client():
    from llama_stack_client import LlamaStackClient

    return LlamaStackClient(base_url=f"http://localhost:8321")


client = create_library_client()


urls = ["chat.rst", "llama3.rst", "datasets.rst", "lora_finetune.rst"]
documents = [
    Document(
        document_id=f"num-{i}",
        content=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}",
        mime_type="text/plain",
        metadata={},
    )
    for i, url in enumerate(urls)
]

# Register a vector database
vector_db_id = "test-vector-db"
client.vector_dbs.register(
    vector_db_id=vector_db_id,
    embedding_model="all-MiniLM-L6-v2",
    embedding_dimension=384,
)

# Insert the documents into the vector database
client.tool_runtime.rag_tool.insert(
    documents=documents,
    vector_db_id=vector_db_id,
    chunk_size_in_tokens=512,
)

agent_config = AgentConfig(
    model="Llama3.2-3B-Instruct",
    instructions="You are a helpful assistant",
    # Enable both RAG and tool usage
    toolgroups=[
        {"name": "builtin::rag", "args": {"vector_db_ids": ["my_docs"]}},
        "builtin::code_interpreter",
    ],
    # Configure safety
    input_shields=["llama_guard"],
    output_shields=["llama_guard"],
    # Control the inference loop
    max_infer_iters=5,
    sampling_params={
        "strategy": {"type": "top_p", "temperature": 0.7, "top_p": 0.95},
        "max_tokens": 2048,
    },
)

agent = Agent(client, agent_config)
session_id = agent.create_session("monitored_session")

# Stream the agent's execution steps
response = agent.create_turn(
    messages=[{"role": "user", "content": "Analyze this code"}],
    attachments=documents,
    session_id=session_id,
)

# Monitor each step of execution
for log in EventLogger().log(response):
    if log.event.step_type == "memory_retrieval":
        print("Retrieved context:", log.event.retrieved_context)
    elif log.event.step_type == "inference":
        print("LLM output:", log.event.model_response)
    elif log.event.step_type == "tool_execution":
        print("Tool call:", log.event.tool_call)
        print("Tool response:", log.event.tool_response)
    elif log.event.step_type == "shield_call":
        if log.event.violation:
            print("Safety violation:", log.event.violation)
