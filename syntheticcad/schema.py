"""Canonical CAD schema and field-mapping helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


FIELD_DEFINITIONS: dict[str, dict[str, Any]] = {
    "event_id": {
        "label": "Event ID",
        "level": "event",
        "description": "Unique identifier for each call for service.",
        "aliases": [
            "cad event number",
            "event id",
            "event number",
            "incident id",
            "incident number",
            "call id",
            "call number",
            "cad number",
            "case number",
            "id",
        ],
    },
    "call_received_datetime": {
        "label": "Call received date and time",
        "level": "event",
        "description": "When the call first entered dispatch.",
        "aliases": [
            "cad event original time queued",
            "event original time queued",
            "original time queued",
            "time queued",
            "queued",
            "received",
            "call received",
            "call date",
            "call datetime",
            "created",
            "create date",
            "entered",
            "reported",
            "date time",
            "datetime",
        ],
    },
    "location": {
        "label": "Location",
        "level": "event",
        "description": "Address, block, intersection, beat, or other location field.",
        "aliases": [
            "dispatch address",
            "dispatch neighborhood",
            "location",
            "address",
            "incident address",
            "street",
            "block",
            "intersection",
            "place",
        ],
    },
    "latitude": {
        "label": "Latitude",
        "level": "event",
        "description": "Latitude coordinate, if available.",
        "aliases": ["latitude", "lat", "y", "y coordinate"],
    },
    "longitude": {
        "label": "Longitude",
        "level": "event",
        "description": "Longitude coordinate, if available.",
        "aliases": ["longitude", "lon", "lng", "long", "x", "x coordinate"],
    },
    "call_type": {
        "label": "Call type",
        "level": "event",
        "description": "Operational category or nature of the call.",
        "aliases": [
            "call type",
            "final call type",
            "initial call type",
            "event group",
            "incident type",
            "event type",
            "nature",
            "problem",
            "type",
            "description",
        ],
    },
    "priority": {
        "label": "Priority",
        "level": "event",
        "description": "Urgency level assigned by dispatch.",
        "aliases": ["priority", "prio", "urgency", "severity"],
    },
    "disposition": {
        "label": "Disposition",
        "level": "event",
        "description": "How the call was resolved.",
        "aliases": [
            "cad event clearance description",
            "clearance description",
            "disposition",
            "dispo",
            "clearance",
            "clearance type",
            "result",
            "resolution",
        ],
    },
    "unit_id": {
        "label": "Unit ID",
        "level": "unit",
        "description": "Officer, vehicle, or unit identifier assigned to the call.",
        "aliases": [
            "call sign dispatch id",
            "call sign id",
            "dispatch id",
            "unit id",
            "unit",
            "officer",
            "badge",
            "resource",
            "vehicle",
        ],
    },
    "dispatch_time": {
        "label": "Dispatch time",
        "level": "unit",
        "description": "When a unit was assigned or dispatched.",
        "aliases": [
            "call sign dispatch time",
            "call sign dispatched",
            "dispatch time",
            "dispatched",
            "assigned",
            "assigned time",
            "enroute",
            "en route",
        ],
    },
    "arrival_time": {
        "label": "Arrival time",
        "level": "unit",
        "description": "When a unit arrived on scene.",
        "aliases": [
            "call sign at scene time",
            "cad event arrived time",
            "arrival time",
            "arrived",
            "onscene",
            "on scene",
            "at scene",
        ],
    },
    "clearance_time": {
        "label": "Clearance time",
        "level": "unit",
        "description": "When a unit cleared the call.",
        "aliases": [
            "call sign in service time",
            "in service time",
            "in-service time",
            "in service",
            "clearance time",
            "clear time",
            "cleared",
            "closed",
            "close time",
            "completed",
        ],
    },
}


EVENT_LEVEL_FIELDS = [
    field_name
    for field_name, definition in FIELD_DEFINITIONS.items()
    if definition["level"] == "event"
]
UNIT_LEVEL_FIELDS = [
    field_name
    for field_name, definition in FIELD_DEFINITIONS.items()
    if definition["level"] == "unit"
]


@dataclass
class SyntheticCADMapping:
    """User-confirmed mapping from canonical CAD fields to CSV columns."""

    event_id: str | None = None
    call_received_datetime: str | None = None
    location: str | None = None
    latitude: str | None = None
    longitude: str | None = None
    call_type: str | None = None
    priority: str | None = None
    disposition: str | None = None
    unit_id: str | None = None
    dispatch_time: str | None = None
    arrival_time: str | None = None
    clearance_time: str | None = None
    extra_event_fields: list[str] = field(default_factory=list)
    extra_unit_fields: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SyntheticCADMapping":
        fields = payload.get("fields", payload)
        values = {name: fields.get(name) for name in FIELD_DEFINITIONS}
        values["extra_event_fields"] = payload.get("extra_event_fields", [])
        values["extra_unit_fields"] = payload.get("extra_unit_fields", [])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        fields = {name: payload.pop(name) for name in FIELD_DEFINITIONS}
        return {"fields": fields, **payload}

    def canonical_to_column(self) -> dict[str, str]:
        return {
            name: getattr(self, name)
            for name in FIELD_DEFINITIONS
            if getattr(self, name, None)
        }

    def column_to_canonical(self) -> dict[str, str]:
        return {column: name for name, column in self.canonical_to_column().items()}

    def mapped_columns(self) -> list[str]:
        columns = list(self.canonical_to_column().values())
        columns.extend(self.extra_event_fields)
        columns.extend(self.extra_unit_fields)
        return list(dict.fromkeys(column for column in columns if column))


def load_mapping(path: str | Path) -> SyntheticCADMapping:
    with Path(path).open("r", encoding="utf-8") as file:
        return SyntheticCADMapping.from_dict(json.load(file))


def save_mapping(mapping: SyntheticCADMapping, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        json.dump(mapping.to_dict(), file, indent=2)
        file.write("\n")
