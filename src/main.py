# src/main.py
from dotenv import load_dotenv
from pathlib import Path

# load .env ở root project (chat-assistant-memory/.env)
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

from src.pipeline.chat_pipeline import ChatPipeline
import argparse
import yaml

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Chat with LLM")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    return parser.parse_args()



def main():
    args = parse_args()
    config = load_config(args.config) if args.config else None
    pipe = ChatPipeline(config)
    pipe.infinite_chat()

if __name__ == "__main__":
    main()