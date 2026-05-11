## Day 1 — Project Setup

### What I did
- Set up Python project with uv (faster than pip)
- Created modular folder structure: ingestion/, db/, config
- Installed PyMuPDF, LangChain text-splitters, psycopg, boto3

### What I learned
- uv replaces pip + venv + pip-tools — single fast tool
- pyproject.toml is the modern way to declare deps (replaces requirements.txt)
- .env files separate config from code (12-factor app principle)

### Questions an interviewer might ask
Q: "Why uv over pip?"
A: uv is written in Rust, ~10-100x faster, handles virtual environments
   automatically, and uses lockfiles for reproducible installs.

Q: "Why keep secrets in .env?"
A: Never commit secrets to git. .env stays local; production uses
   AWS Secrets Manager or Parameter Store.