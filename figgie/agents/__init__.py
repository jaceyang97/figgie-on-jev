from .base import Agent
from .classical import BottomFeeder, Chartist, Fundamentalist, MarketMaker, Noise
from .jev_agent import JevAgent

# The Figgie paper's four strategies, then the market maker from the wider literature.
CORE = ("fundamentalist", "bottom_feeder", "chartist", "noise")
CLASSICAL = {
    "fundamentalist": Fundamentalist,
    "bottom_feeder": BottomFeeder,
    "chartist": Chartist,
    "noise": Noise,
    "market_maker": MarketMaker,
}


# Context parts a Jev agent can be given. log and summary are observed facts;
# known (cards known to exist) and assist (exact posterior) are derived by code.
CONTEXT_FLAGS = ("log", "summary", "known", "assist")


def make_agent(spec: str, client=None, jev_latency: float | None = None) -> Agent:
    """Build an agent from a spec like 'fundamentalist' or 'jev:hoarder' or 'jev:value+assist' or 'jev:neutral+log+summary'."""
    if spec.startswith("jev"):
        _, _, rest = spec.partition(":")
        personality, *flags = (rest or "neutral").split("+")
        if client is None:
            raise ValueError("Jev agents need a client")
        kw = {"use_measured_latency": jev_latency is None}
        if jev_latency is not None:
            kw["latency"] = jev_latency
        unknown = set(flags) - set(CONTEXT_FLAGS)
        if unknown:
            raise ValueError(f"unknown Jev flag(s) {sorted(unknown)}; choose from {CONTEXT_FLAGS}")
        return JevAgent(client, personality=personality, **{f: f in flags for f in CONTEXT_FLAGS}, **kw)
    if spec not in CLASSICAL:
        raise ValueError(f"unknown agent {spec!r}; choose from {sorted(CLASSICAL)} or jev:<personality>")
    return CLASSICAL[spec]()


__all__ = ["Agent", "JevAgent", "make_agent", "CLASSICAL", "CORE", *(c.__name__ for c in CLASSICAL.values())]
