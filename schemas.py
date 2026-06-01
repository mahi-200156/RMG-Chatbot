from pydantic import BaseModel
from typing import Optional, List

class JobListRequest(BaseModel):
    location: Optional[str] = None
    title:    Optional[str] = None

class EmployeeSearchRequest(BaseModel):
    employee_name: str
    top_k:         int = 5

class JobSearchRequest(BaseModel):
    job_id: str
    top_k:  int = 5

class GeneralSearchRequest(BaseModel):
    query:  str
    top_k:  int = 5