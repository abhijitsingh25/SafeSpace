# SafeSpace task runner for Windows PowerShell.
#   .\tasks.ps1 install | api | ui | index | test | safety | lint | format | check

param([Parameter(Position = 0)][string]$Task = "help")

switch ($Task.ToLower()) {
    "help" {
        Write-Host "SafeSpace"
        Write-Host ""
        Write-Host "  install        Install runtime + dev dependencies (no torch)"
        Write-Host "  install-local  Also install Ollama/Streamlit/sentence-transformers"
        Write-Host "  api            Run the API on :8000 (also serves the chat UI)"
        Write-Host "  ui             Run the Streamlit dev UI on :8501"
        Write-Host "  index          Rebuild corpus embeddings"
        Write-Host "  test           Run the test suite"
        Write-Host "  safety         Run the safety eval set only"
        Write-Host "  lint / format  Ruff"
        Write-Host "  check          Lint + format check + tests (what CI runs)"
    }
    "install"       { uv sync --extra dev }
    "install-local" { uv sync --extra dev --extra local }
    "api"           { uv run uvicorn backend.main:app --reload --port 8000 }
    "ui"            { uv run --extra local streamlit run frontend.py }
    "index"         { uv run --extra local python -m scripts.build_index }
    "test"          { uv run pytest -q }
    "safety"        { uv run pytest tests/test_safety.py -v }
    "lint"          { uv run ruff check . }
    "format"        { uv run ruff format . }
    "check" {
        uv run ruff check .
        if ($LASTEXITCODE -ne 0) { exit 1 }
        uv run ruff format --check .
        if ($LASTEXITCODE -ne 0) { exit 1 }
        uv run pytest -q
    }
    default {
        Write-Host "Unknown task '$Task'. Run '.\tasks.ps1 help'."
        exit 1
    }
}
