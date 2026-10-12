# Project_R — Attendance and Mock Hubble Integration

Project_R retains its configurable multi-camera attendance system and adds a local mock Hubble employee API.

## Added Hubble features

- Look up mock employee records by Hubble Login ID from the dashboard.
- Optionally sync Hubble data while creating an attendance employee.
- Persist Hubble ID, login, reporting manager, HR contact, department, designation, and team alongside the SQLite attendance record.
- Enrich manager/HR names with email addresses from the employee directory stored in MongoDB.
- CC the employee's HR email on early-arrival notifications, falling back to the configured email sender if no HR address is available.
- Check Hubble service availability at `GET /hubble/status`.

## Setup

Install the Python dependencies into the existing project environment:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set the email settings and, if needed, `HUBBLE_API_URL`. Copy `.env.hubble.example` to `.env.hubble` and configure MongoDB. Both secret files are ignored by Git; do not commit credentials.

## Run

Start the mock Hubble service in one terminal:

```bash
bash run_hubble.sh
```

Start the attendance API in another terminal:

```bash
bash run_backend.sh
```

Start the dashboard as usual from `hr-dashboard` with `npm run dev`.

## Mock data and MongoDB enrichment

The dashboard's Fetch action reads the local `data/mock_employees.json` through `GET /mock-hubble/employee/{login_id}`; this is mock data, not a live Hubble connector. Supplying a login ID during employee creation calls the Hubble sync endpoint and writes the employee to MongoDB. Reporting-manager and HR email enrichment uses `data/employee_directory.json`, which must first be loaded into MongoDB by calling `POST /api/directory/load`. The MongoDB-backed `GET /api/employees/{login_id}` endpoint only finds employees after they have been synced.

With the Hubble API running and MongoDB configured, initialize the directory once:

```bash
curl -X POST http://127.0.0.1:8001/api/directory/load
```

Useful Hubble endpoints:

- `GET /mock-hubble/employee/{login_id}` — read a local mock record.
- `POST /api/employees/sync/{login_id}` — enrich and save a mock employee to MongoDB.
- `POST /api/directory/load` — load manager/HR email data into MongoDB.
- `GET /api/employees/{login_id}` — retrieve a previously synced employee.

Manual attendance-employee creation remains available when Login ID is omitted. Camera configuration and multi-camera operation remain unchanged.

## Employee shifts, monthly calendar, and Excel reports

- Edit employee shift start/end from the Employees page. The update changes the current employee shift; no effective-date shift history is maintained.
- Open an employee's attendance history to browse a month. Select a day to view its events and mark, update, or remove a leave record. Calendar statuses include full day, partial day, leave, absent, and weekend.
- Download weekly (Monday–Sunday) or monthly XLSX reports for one employee from the employee card, or for all employees from the Attendance page. Workbooks contain attendance summary and event sheets.

The attendance API adds `PUT /employees/{employee_id}/shift`, `GET /employees/{employee_id}/attendance-history?year=YYYY&month=MM`, `POST /employees/{employee_id}/leave`, and matching leave GET/DELETE endpoints. Excel downloads use `GET /reports/employee/{employee_id}?period=weekly|monthly&date=YYYY-MM-DD` or `GET /reports/attendance?period=weekly|monthly&date=YYYY-MM-DD`.
