from .base import Agent
from .classical import BottomFeeder, Chartist, Fundamentalist, Noise
from .jev_agent import JevAgent

CLASSICAL = {
    "fundamentalist": Fundamentalist,
    "bottom_feeder": BottomFeeder,
    "chartist": Chartist,
    "noise": Noise,
}


def make_agent(spec: str, client=None, jev_latency: float | None = None) -> Agent:
    """Build an agent from a spec like 'fundamentalist' or 'jev:hoarder' or 'jev:value+assist'."""
    if spec.startswith("jev"):
        _, _, rest = spec.partition(":")
        personality, _, flag = (rest or "neutral").partition("+")
        if client is None:
            raise ValueError("Jev agents need a client")
        kw = {"use_measured_latency": jev_latency is None}
        if jev_latency is not None:
            kw["latency"] = jev_latency
        return JevAgent(client, personality=personality, assist=flag == "assist", **kw)
    if spec not in CLASSICAL:
        raise ValueError(f"unknown agent {spec!r}; choose from {sorted(CLASSICAL)} or jev:<personality>")
    return CLASSICAL[spec]()


__all__ = ["Agent", "Fundamentalist", "BottomFeeder", "Chartist", "Noise", "JevAgent", "make_agent", "CLASSICAL"]
