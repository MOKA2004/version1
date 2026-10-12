import json
from pathlib import Path

from fastapi import APIRouter, HTTPException


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "mock_employees.json"
router = APIRouter(prefix="/mock-hubble", tags=["Mock Hubble"])


def load_employees() -> list[dict]:
    try:
        with DATA_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500, detail="mock_employees.json not found"
        ) from error
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=500, detail="mock_employees.json contains invalid JSON"
        ) from error


@router.get("/employee/{login_id}")
def get_mock_employee(login_id: str):
    for employee in load_employees():
        if employee.get("loginId", "").casefold() == login_id.casefold():
            return employee
    raise HTTPException(status_code=404, detail="Employee not found in mock Hubble data")
