from .base import EQUAL_SPEED, PAPER_SPEED, SPEEDS, Agent
from .classical import BottomFeeder, Chartist, Fundamentalist, MarketMakerAS, MarketMakerGM, Noise
from .jev_agent import JevAgent

# The Figgie paper's four strategies, then the two market makers from the wider literature.
CORE = ("fundamentalist", "bottom_feeder", "chartist", "noise")
MARKET_MAKERS = ("market_maker_gm", "market_maker_as")
TYPES = CORE + MARKET_MAKERS
CLASSICAL = {
    "fundamentalist": Fundamentalist,
    "bottom_feeder": BottomFeeder,
    "chartist": Chartist,
    "noise": Noise,
    "market_maker_gm": MarketMakerGM,
    "market_maker_as": MarketMakerAS,
}


# Context parts a Jev agent can be given. log and summary are observed facts;
# known (cards known to exist) and assist (card-counting probabilities) are derived by code.
CONTEXT_FLAGS = ("log", "summary", "known", "assist")


def parse_speed(text: str | None, default: tuple[float, float]) -> tuple[float, float]:
    """'equal', 'paper' or '<wake_rate>/<latency>'."""
    if not text:
        return default
    if text in SPEEDS:
        return SPEEDS[text]
    rate, _, lat = text.partition("/")
    return float(rate), float(lat or 0.0)


def make_agent(spec: str, client=None, speed: tuple[float, float] = EQUAL_SPEED, decode: str = "sample",
               meta: dict | None = None) -> Agent:
    """Build an agent from a spec.

    Specs: 'fundamentalist', 'jev:neutral', 'jev:chartist-desc+log+summary', and any of them with a speed
    suffix '@<wake_rate>/<latency>' (e.g. 'fundamentalist@1/200'); without a suffix the agent gets `speed`.
    """
    spec, _, speed_text = spec.partition("@")
    rate, latency = parse_speed(speed_text, speed)
    if spec.startswith("jev"):
        _, _, rest = spec.partition(":")
        personality, *flags = (rest or "neutral").split("+")
        if client is None:
            raise ValueError("Jev agents need a client")
        unknown = set(flags) - set(CONTEXT_FLAGS)
        if unknown:
            raise ValueError(f"unknown Jev flag(s) {sorted(unknown)}; choose from {CONTEXT_FLAGS}")
        return JevAgent(client, personality=personality, decode=decode, meta=meta,
                        wake_rate=rate, latency=latency, **{f: f in flags for f in CONTEXT_FLAGS})
    if spec not in CLASSICAL:
        raise ValueError(f"unknown agent {spec!r}; choose from {sorted(CLASSICAL)} or jev:<personality>")
    return CLASSICAL[spec](wake_rate=rate, latency=latency)


__all__ = ["Agent", "JevAgent", "make_agent", "parse_speed", "CLASSICAL", "CORE", "MARKET_MAKERS", "TYPES",
           "EQUAL_SPEED", "PAPER_SPEED", "SPEEDS", *(c.__name__ for c in CLASSICAL.values())]
