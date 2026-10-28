.PHONY: train smoke test run
train:
	cd backend && python -m app.train
smoke:
	cd backend && python -m app.train --smoke
test:
	cd backend && python -m pytest -q
run:
	docker compose up --build
