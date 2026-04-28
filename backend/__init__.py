"""
Backend module for AGCT system
Contains: RAG, Agents, Graph, Utils, Core
"""

__version__ = "1.0.0"
__author__ = "R2101-cmd"

def __getattr__(name):
    if name == "AGCTSystem":
        from backend.core.agct_system import AGCTSystem

        return AGCTSystem
    raise AttributeError(name)


__all__ = ["AGCTSystem"]
