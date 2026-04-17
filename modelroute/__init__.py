"""MODELROUTE — Local model router / proxy across Ollama, vLLM, and cloud with fallback."""
from modelroute.core import scan, TOOL_NAME, TOOL_VERSION
__all__ = ["scan", "TOOL_NAME", "TOOL_VERSION"]
