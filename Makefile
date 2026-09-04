train:
python src/train_models.py

evaluate:
python src/evaluate.py

explain:
python src/explainability.py

test:
pytest tests/

docker-up:
docker-compose up

docker-down:
docker-compose down
