.PHONY: seed

seed:
	DATABASE_URL=postgresql://postgres@localhost:5432/blueroads_dev python scripts/seed.py
