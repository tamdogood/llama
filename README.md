# llama
**Set up**: First we will have to start the distribution 
```bash
git clone git@github.com:meta-llama/llama-stack.git
cd llama-stack
pip install -e .
```

```bash
# export environment variables
export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
export LLAMA_STACK_PORT=8321
```

- For this tutorial, I decided to go with Ollama serving since it's don't require any api key and is one of the easiest way to spin up a serving from your local machine. But llama-stack offers many more providers like AWS and HuggingFace, so if you decide to go with another provider, you can skip this step below
```bash
# Terminal 1 for ollama inference
ollama run llama3.2:3b-instruct-fp16 --keepalive 60m
```
- After you have the ollama serving running, you can check to make sure it's running in the right port by checking out http://localhost:11434/ in your browser ![[Pasted image 20250126152620.png]]
- Once `ollama` is running, you can open a new terminal in parallel and run these commands:
``` bash
# Terminal 2 for llama-stack distro build
llama stack build --template ollama --image-type conda
conda activate llamastack-ollama

llama stack run ./distributions/ollama/run.yaml
```
