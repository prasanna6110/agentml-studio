# Generative AI and Agentic AI

## Large language models

A large language model (LLM) is a transformer neural network trained to predict the next token on very large text corpora. Self-attention lets every token weigh every other token in the context window, which is why transformers capture long-range dependencies better than recurrent networks. After pre-training, models are aligned with instruction tuning and reinforcement learning from human or AI feedback so that they follow instructions helpfully and safely.

## Retrieval-augmented generation (RAG)

Retrieval-augmented generation grounds an LLM's answer in external documents. Documents are split into chunks, each chunk is converted into an embedding vector, and the vectors are stored in a vector database. At query time the question is embedded, the most similar chunks are retrieved and inserted into the prompt as context, and the model answers from that context with citations. RAG reduces hallucination and lets a model use private or up-to-date knowledge without retraining.

## Hybrid search

Dense retrieval with embeddings captures semantic similarity, while sparse keyword retrieval such as BM25 is precise for exact terms, names and codes. Hybrid search runs both and merges the ranked lists, commonly with reciprocal rank fusion (RRF), where each document scores the sum of 1/(k + rank) over the rankings. Hybrid search is usually more robust than either method alone.

## Agentic RAG

Agentic RAG adds decision-making to the retrieval loop. The agent can decompose a complex question into sub-queries, grade whether retrieved passages are relevant, rewrite the query and retrieve again when results are weak, and reflect on its draft answer to check that every claim is grounded in the sources. This self-correction makes answers more reliable than a single retrieve-then-generate pass.

## AI agents and tool use

An AI agent is an LLM that runs in a loop: it reasons about the task, decides to call a tool such as a calculator, search engine, database or code interpreter, observes the tool result, and repeats until it can answer. This pattern is known as ReAct (reason + act). Function calling lets the model emit structured tool calls with JSON arguments that match a declared schema, so the host program can execute them safely.

## Multi-agent systems

In a multi-agent system several specialised agents collaborate, each with its own role, prompt and tools — for example a planner, an analyst, an engineer and a critic. An orchestrator routes work between them. Specialisation keeps each prompt focused, and a critic or reviewer agent that challenges the others' output is an effective way to catch errors, a pattern sometimes called self-reflection or LLM-as-a-judge.

## Agent memory

Short-term memory is the conversation history kept in the context window, usually trimmed to a sliding window. Long-term memory persists facts or past experiences outside the model, often in a vector store, and relevant memories are recalled by semantic similarity and injected into the prompt when a new task starts.
