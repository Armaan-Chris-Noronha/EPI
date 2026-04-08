"""
API sub-package — FastAPI application for HF Spaces deployment.

Exposes the OpenEnv HTTP interface on port 7860:
  GET  /           → health check
  POST /reset      → StepResult  (accepts empty body {})
  POST /step       → StepResult
  GET  /state      → StateResult
  GET  /tasks      → list of task ids
  GET  /health     → {"status": "ok"}
"""
