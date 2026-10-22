import os

dirs = [
    "data/raw",
    "data/processed",
    "src",
    "src/ingest",
    "src/models",
    "src/dispatch",
    "src/metrics",
    "src/utils",
    "tests",
    "plots",
    "web/static",
    "web/templates"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    if d.startswith("src"):
        init_file = os.path.join(d, "__init__.py")
        with open(init_file, "a") as f:
            pass

print("Directory structure created successfully.")
