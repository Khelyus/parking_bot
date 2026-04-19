from parking_bot.state_machine import AvailabilityStabilizer
from parking_bot.types import Availability


def test_stabilizer_changes_state_only_after_enough_frames() -> None:
    stabilizer = AvailabilityStabilizer(stable_cycles=2)

    assert stabilizer.ingest(Availability.FULL) is None
    assert stabilizer.ingest(Availability.FULL) == (Availability.UNKNOWN, Availability.FULL)
    assert stabilizer.ingest(Availability.FREE) is None
    assert stabilizer.ingest(Availability.FREE) == (Availability.FULL, Availability.FREE)


def test_unknown_frame_does_not_force_transition() -> None:
    stabilizer = AvailabilityStabilizer(stable_cycles=2)
    stabilizer.ingest(Availability.FULL)
    stabilizer.ingest(Availability.FULL)
    assert stabilizer.snapshot().stable_availability == Availability.FULL
    assert stabilizer.ingest(Availability.UNKNOWN) is None
    assert stabilizer.snapshot().stable_availability == Availability.FULL
