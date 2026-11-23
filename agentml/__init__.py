"""AgentML Studio — autonomous multi-agent AutoML + Agentic RAG framework.

Quick start::

    from agentml import AutoMLPipeline, get_llm

    pipeline = AutoMLPipeline(llm=get_llm())          # uses AGENTML_PROVIDER / API keys from .env
    result = pipeline.run("examples/data/customer_churn.csv",
                          goal="Predict which customers will churn")
    print(result.best_model, result.test_metrics)
"""

from agentml.config import Settings, get_settings
from agentml.llm import LLM, Message, get_llm
from agentml.pipeline import AutoMLPipeline, PipelineResult
from agentml.rag import AgenticRAG

__version__ = "1.0.0"

__all__ = [
    "AgenticRAG",
    "AutoMLPipeline",
    "LLM",
    "Message",
    "PipelineResult",
    "Settings",
    "__version__",
    "get_llm",
    "get_settings",
]
