python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# pip install -U sentence-transformers

docker compose up -d

python -m src.main --config ./configs/app.yaml