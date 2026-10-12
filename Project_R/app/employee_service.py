from app.database import employees_collection, employee_directory_collection


def normalize_name(name: str | None) -> str:
    return " ".join((name or "").split()).strip().lower()


def find_person_by_name(name: str | None):
    normalized_name = normalize_name(name)
    if not normalized_name:
        return None

    for person in employee_directory_collection.find({}, {"_id": 0}):
        if normalize_name(person.get("name")) == normalized_name:
            return person
    return None


def enrich_employee(hubble_data: dict) -> dict:
    manager = find_person_by_name(hubble_data.get("reportsTo"))
    hr = find_person_by_name(hubble_data.get("hrContact"))
    return {
        "id": hubble_data.get("id"),
        "name": hubble_data.get("name"),
        "loginId": hubble_data.get("loginId"),
        "mailId": hubble_data.get("mailId"),
        "reportsTo": hubble_data.get("reportsTo"),
        "reportsToMailId": manager.get("mailId") if manager else None,
        "hrContact": hubble_data.get("hrContact"),
        "hrEmail": hr.get("mailId") if hr else None,
        "department": hubble_data.get("department"),
        "designation": hubble_data.get("designation"),
        "teamName": hubble_data.get("teamName"),
    }


def save_employee(employee: dict) -> dict:
    employees_collection.update_one(
        {"id": employee["id"]}, {"$set": employee}, upsert=True
    )
    return employee
