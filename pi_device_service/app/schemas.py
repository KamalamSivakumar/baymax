from enum import Enum
from pydantic import BaseModel


class Mode(str, Enum):
    kids = "kids"
    young_adult = "young_adult"
    adult = "adult"


class DeviceModeRequest(BaseModel):
    mode: Mode


class DeviceActionRequest(BaseModel):
    action: str