# Chatbot-LLM
**Vulcan Lab – Entrance Exam Project**

A modular, memory-augmented chatbot system built with LLMs, structured outputs, and a vector database for long-term conversational context.

---

## Overview

This project implements a **stateful LLM chatbot** with:

- **Structured outputs (JSON Schema enforced)**
- **Short-term & long-term memory**
- **Context augmentation via vector search**
- **Scalable multi-user / multi-chat architecture**

The system periodically summarizes conversations and stores them in a vector database (Milvus), enabling retrieval and augmentation for future queries.

---

## Architecture Highlights

- **LLM Provider**: Groq API  
- **Vector Database**: Milvus  
- **Memory Types**:
  - Short-term: current context window
  - Long-term: summarized session memory (vectorized)
- **Core Components**:
  - Query understanding (ambiguity detection, rewriting)
  - Context augmentation
  - Structured summarization
  - Grounded answer generation

## Workflow

![Full workflow](./content/full_workflow.png)


---

## Getting Started

### 1. Prerequisites

- Docker & Docker Compose
- Python 3.10+
- Groq API key

### 2. Start Milvus (Vector Database)

```bash
docker compose up -d
```

### 3. Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the Chatbot

```bash
chmod u+x ./run.sh
./run.sh
```

---

## Key Design Assumptions

- The chatbot enforces **structured LLM outputs** via JSON schema
- The architecture supports **horizontal scalability** through `user_id` and `chat_id`
- Long conversations are supported via **vectorized session memory**

---

## Limitations

- Ambiguous query classification relies solely on the LLM (no dedicated classifier)
- Context input size can grow large and needs further optimization
- Context augmentation is currently prompt-based

---

## Demo

🎥 **Video Demo**: _To be updated_

---

## References

- Groq API Keys: https://console.groq.com/keys  
- Milvus Quickstart: https://milvus.io/docs/quickstart.md  
- RAG Context Refinement Agent: https://devpost.com/software/rag-context-refinement-agent  
- LangChain Groq Integration: https://github.com/langchain-ai/langchain/tree/master/libs/partners/groq  