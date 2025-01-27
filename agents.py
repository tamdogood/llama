import os

from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types import Document


def create_http_client():
    from llama_stack_client import LlamaStackClient

    return LlamaStackClient(
        base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}"
    )


client = create_http_client()


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
    model=os.environ["INFERENCE_MODEL"],
    instructions="You are a helpful assistant",
    # Enable both RAG and tool usage
    toolgroups=[
        {"name": "builtin::rag", "args": {"vector_db_ids": ["test-vector-db"]}},
        "builtin::code_interpreter",
    ],
    # # Configure safety
    input_shields=["llama_guard"],
    output_shields=["llama_guard"],
    # Control the inference loop
    max_infer_iters=5,
    strategy={"type": "top_p", "temperature": 0.7, "top_p": 0.95},
    enable_session_persistence=True,
)

agent = Agent(client, agent_config)
session_id = agent.create_session("monitored_session")

# Stream the agent's execution steps
response = agent.create_turn(
    messages=[
        {
            "role": "user",
            "content": "Analyze this doc and help me write some code to finetune a model from scratch",
        }
    ],
    documents=documents,
    session_id=session_id,
)

for log in EventLogger().log(response):
    log.print()
