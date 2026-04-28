"""Agent exports for the AGCT backend."""


def __getattr__(name):
    if name == "ReasonerAgent":
        from backend.agents.reasoner_agent import ReasonerAgent

        return ReasonerAgent
    if name == "RetrieverAgent":
        from backend.agents.retriever_agent import RetrieverAgent

        return RetrieverAgent
    if name == "RLPersonalizationAgent":
        from backend.agents.rl_personalization import RLPersonalizationAgent

        return RLPersonalizationAgent
    if name == "VerifierAgent":
        from backend.agents.verifier_agent import VerifierAgent

        return VerifierAgent
    raise AttributeError(name)

__all__ = [
    "ReasonerAgent",
    "RetrieverAgent",
    "RLPersonalizationAgent",
    "VerifierAgent",
]
