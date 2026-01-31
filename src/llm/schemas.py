SESSION_SUMMARY_CONTENT_SCHEMA = {
  "name": "session_summary_content",
  "strict": True,
  "schema": {
    "type": "object",
    "additionalProperties": False,   
    "properties": {
      "user_profile": {
        "type": "object",
        "additionalProperties": False,  
        "properties": {
          "prefs": {"type":"array","items":{"type":"string"}, "description":"Stable user preferences about response style/format/language."},
          "constraints": {"type":"array","items":{"type":"string"}, "description":"Condition of the user, limitations, restrictions."}
        },
        "required": ["prefs", "constraints"]
      },
      "key_facts": {"type": "array", "items": {"type": "string"}, "description":"Important factual statements from the conversation, what user asked about, what information did you give him"},
      "decisions": {"type": "array", "items": {"type": "string"}, "description":"Explicit decisions or choices made."},
      "open_questions": {"type": "array", "items": {"type": "string"}, "description":"Unresolved questions asked or implied by the user."},
      "todos": {"type": "array", "items": {"type": "string"}, "description":"Actionable tasks explicitly requested or implied."}
    },
    "required": ["user_profile", "key_facts", "decisions", "open_questions", "todos"]
  }
}


CLARIFYING_QUESTIONS_SCHEMA = {
  "name": "clarifying_questions",
  "strict": True,
  "schema": {
    "type": "object",
    "additionalProperties": False,
    "properties": {
      "questions": {
        "type": "array",
        "items": {
          "type": "object",
          "additionalProperties": False,
          "properties": {
            "question": {"type": "string"},
            "type": {
              "type": "string",
              "enum": ["scope", "constraints", "context", "goal", "format"]
            }
          },
          "required": ["question", "type"]
        }
      }
    },
    "required": ["questions"]
  }
}


REWRITTEN_QUERY_SCHEMA = {
  "name": "rewritten_query",
  "strict": True,
  "schema": {
    "type": "object",
    "additionalProperties": False,
    "properties": {
      "rewritten_query": {
        "type": "string",
        "description": "A clearer and more specific version of the original query."
      }
    },
    "required": ["rewritten_query"]
  }
}

# src/llm/schemas.py\

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



REWRITTEN_QUERY_SCHEMA = {
  "name": "rewritten_query",
  "strict": True,
  "schema": {
    "type": "object",
    "additionalProperties": False,
    "properties": {
      "rewritten_query": {
        "type": "string",
        "description": "A clearer and more specific version of the original query."
      }
    },
    "required": ["rewritten_query"]
  }
}