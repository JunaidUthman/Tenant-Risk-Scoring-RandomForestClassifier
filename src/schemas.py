from pydantic import BaseModel
from typing import Optional

# Now the API accepts exactly what the model was trained on
class TenantScoreRequest(BaseModel):
    missedPeriods: int
    totalDisputes: int