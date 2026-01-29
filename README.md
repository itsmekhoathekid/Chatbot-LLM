# Chatbot-LLM
Vulcan Lab entrance exam

create an API at https://console.groq.com/keys 

chat-assistant-memory/
├─ README.md
├─ pyproject.toml              # hoặc requirements.txt
├─ .env.example                # nếu cần API key
├─ config/
│  └─ config.yaml
├─ data/
│  └─ conversation_logs/
│     ├─ long_session_trigger.jsonl
│     ├─ ambiguous_queries.jsonl
│     └─ clarification_needed.jsonl
├─ sessions/                   # runtime artifacts (gitignore)
│  └─ (created at runtime)
├─ src/
│  ├─ main.py                  # entrypoint CLI/demo
│  ├─ pipeline/
│  │  ├─ chat_pipeline.py      # orchestration: input→memory→query→prompt
│  │  └─ prompt_builder.py     # builds final_augmented_context/prompt
│  ├─ memory/
│  │  ├─ memory_manager.py     # trigger logic + store/retrieve
│  │  ├─ summarizer.py         # LLM summarization calls
│  │  ├─ token_counter.py      # char/word/token counting
│  │  └─ store_fs.py           # filesystem store (sessions/<sid>/...)
│  ├─ understanding/
│  │  ├─ query_understanding.py # ambiguity detect, rewrite, clarify
│  │  └─ retriever.py           # select needed memory fields/records
│  ├─ schemas/
│  │  ├─ session_summary.py    
│  │  └─ query_understanding.py
│  └─ llm/
│     ├─ client.py             # wrapper OpenAI/Claude/local
│     └─ prompts.py            # prompt templates
└─ tests/
   ├─ test_memory_trigger.py
   ├─ test_schema_validation.py
   └─ test_query_understanding.py


sessions/
└─ sess_20260129_001/    # hoặc uuid: 3f2c.../
   ├─ messages.jsonl / {"idx": 12, "role": "user", "content": "...", "ts": "2026-01-29T16:10:00+07:00"}
   ├─ rolling_summary.json 
        {
            "session_summary": { ... },
            "message_range_summarized": { "from": 0, "to": 42 }
        }
   ├─ episodic_memory.jsonl      # optional (facts/decisions/todos atomic)
   └─ stats.json 
        {
            "approx_tokens_total": 10834,
            "threshold": 10000,
            "last_summarized_to": 42,
            "recent_window_size": 20
        }



chat_id : directory
    - session_summary.json
    - chat_history.json