from enum import IntEnum


class EventType(IntEnum):
    INCIDENT = 0
    RESPONDER_AVAILABLE = 1
    ALLOCATION = 2
    FAILURE = 3

# class ActionType(IntEnum):
#     INCIDENT_RESP = 0
#     MOVE_TO_STATION = 1
#     REASSIGN = 2
#     DO_NOTHING = 3  # TODO GET RID OF IT?


class RespStatus(IntEnum):
    WAITING = 0
    RESPONDING = 1
    SERVICING = 2
    IN_TRANSIT = 3

