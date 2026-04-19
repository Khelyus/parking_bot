from __future__ import annotations

from dataclasses import dataclass

from parking_bot.types import Availability


@dataclass(slots=True)
class StabilizerSnapshot:
    stable_availability: Availability
    candidate_availability: Availability
    candidate_hits: int


class AvailabilityStabilizer:
    def __init__(self, stable_cycles: int) -> None:
        self.stable_cycles = max(1, stable_cycles)
        self._stable = Availability.UNKNOWN
        self._candidate = Availability.UNKNOWN
        self._candidate_hits = 0

    def ingest(self, availability: Availability) -> tuple[Availability, Availability] | None:
        if availability == Availability.UNKNOWN:
            self._candidate = Availability.UNKNOWN
            self._candidate_hits = 0
            return None

        if availability == self._candidate:
            self._candidate_hits += 1
        else:
            self._candidate = availability
            self._candidate_hits = 1

        if self._candidate_hits >= self.stable_cycles and self._stable != availability:
            previous = self._stable
            self._stable = availability
            return previous, availability
        return None

    def snapshot(self) -> StabilizerSnapshot:
        return StabilizerSnapshot(
            stable_availability=self._stable,
            candidate_availability=self._candidate,
            candidate_hits=self._candidate_hits,
        )
