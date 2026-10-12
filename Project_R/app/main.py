import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo.errors import PyMongoError

from app.database import employee_directory_collection, employees_collection
from app.employee_service import enrich_employee, save_employee
from app.mock_hubble import router as mock_hubble_router


BASE_DIR = Path(__file__).resolve().parents[1]
app = FastAPI(
    title="Hubble Employee API",
    description="Mock Hubble Employee Integration with MongoDB",
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(mock_hubble_router)


@app.get("/")
def home():
    return {
        "success": True,
        "message": "Hubble Employee API is running",
        "service": "mock-hubble",
    }


@app.post("/api/directory/load")
def load_directory():
    directory_path = BASE_DIR / "data" / "employee_directory.json"
    try:
        with directory_path.open("r", encoding="utf-8") as file:
            people = json.load(file)
        for person in people:
            employee_directory_collection.update_one(
                {"name": person["name"]}, {"$set": person}, upsert=True
            )
        return {
            "success": True,
            "message": "Employee directory loaded successfully",
            "count": len(people),
        }
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500, detail="employee_directory.json not found"
        ) from error
    except PyMongoError as error:
        raise HTTPException(
            status_code=503, detail=f"MongoDB is unavailable: {error}"
        ) from error
    except (KeyError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=500, detail=f"Invalid employee directory data: {error}"
        ) from error


@app.post("/api/employees/sync/{login_id}")
def sync_employee(login_id: str):
    from app.mock_hubble import load_employees

    hubble_employee = next(
        (
            employee
            for employee in load_employees()
            if employee.get("loginId", "").casefold() == login_id.casefold()
        ),
        None,
    )
    if not hubble_employee:
        raise HTTPException(
            status_code=404,
            detail=f"Employee '{login_id}' not found in Hubble mock data",
        )

    try:
        employee = enrich_employee(hubble_employee)
        saved_employee = save_employee(employee)
    except PyMongoError as error:
        raise HTTPException(
            status_code=503, detail=f"MongoDB is unavailable: {error}"
        ) from error

    return {
        "success": True,
        "message": "Employee synchronized successfully",
        "data": saved_employee,
    }


@app.get("/api/employees/{login_id}")
def get_employee(login_id: str):
    try:
        employee = employees_collection.find_one(
            {"loginId": login_id}, {"_id": 0}
        )
    except PyMongoError as error:
        raise HTTPException(
            status_code=503, detail=f"MongoDB is unavailable: {error}"
        ) from error

    if not employee:
        raise HTTPException(
            status_code=404,
            detail=f"Employee '{login_id}' not found in MongoDB; sync it first",
        )
    return {"success": True, "data": employee}
