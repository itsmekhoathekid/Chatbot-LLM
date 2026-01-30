SESSION_SUMMARY_CONTENT_SCHEMA = {
  "name": "session_summary_content",
  "strict": True,
  "schema": {
    "type": "object",
    "additionalProperties": False,   # ✅ root object
    "properties": {
      "user_profile": {
        "type": "object",
        "additionalProperties": False,  # ✅ nested object MUST have this
        "properties": {
          "prefs": {"type": "array", "items": {"type": "string"}},
          "constraints": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["prefs", "constraints"]
      },
      "key_facts": {"type": "array", "items": {"type": "string"}},
      "decisions": {"type": "array", "items": {"type": "string"}},
      "open_questions": {"type": "array", "items": {"type": "string"}},
      "todos": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["user_profile", "key_facts", "decisions", "open_questions", "todos"]
  }
}



# src/llm/schemas.py
AMBIGUOUS_BOOL_SCHEMA = {
  "name": "ambiguous_bool",
  "strict": True,
  "schema": {
    "type": "object",
    "properties": {
      "is_ambiguous": {"type": "boolean"}
    },
    "required": ["is_ambiguous"],
    "additionalProperties": False
  }
}



from pydantic import BaseModel

class AmbiguousResult(BaseModel):
    is_ambiguous: bool