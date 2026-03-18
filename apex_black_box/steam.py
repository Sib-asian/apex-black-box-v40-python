from __future__ import annotations

from .engine import analyze_steam, build_steam_advice


class SteamAnalyzer:
    def __init__(self, quotes: dict):
        # quotes: {"sO": float, "sC": float, "tO": float, "tC": float}
        self.quotes = quotes
        self._result: dict | None = None

    def detect_movement(self) -> dict:
        q = self.quotes
        self._result = analyze_steam(
            float(q.get("sO", 0)),
            float(q.get("sC", 0)),
            float(q.get("tO", q.get("tC", 2.5))),
            float(q.get("tC", 2.5)),
        )
        return self._result

    def generate_advice(self) -> list[dict]:
        if self._result is None:
            self.detect_movement()
        return build_steam_advice(
            self._result,
            float(self.quotes.get("tC", 2.5)),
            float(self.quotes.get("sC", 0)),
        )
