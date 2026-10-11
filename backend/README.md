# Backend

See the repository README for full documentation.

    python -m venv .venv && .\.venv\Scripts\Activate.ps1
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
    uvicorn app.main:app --reload --port 8000
    pytest
