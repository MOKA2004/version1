Project Overview
Project_R is an end-to-end AI/Computer Vision-powered Smart Attendance & Employee Management System with multi-camera facial recognition, shift & leave tracking, automated email notifications, Excel report generation, and integration with a corporate employee directory service (Hubble).

Project Structure
text
Project_v1/
├── .venv/                         # Python virtual environment
└── Project_R/                     # Main project directory
    ├── app/                       # Mock Hubble & MongoDB enrichment service
    │   ├── database.py            # MongoDB connection
    │   ├── employee_service.py    # Employee directory enrichment & sync logic
    │   ├── main.py                # Hubble service FastAPI entrypoint (Port 8001)
    │   └── mock_hubble.py         # Mock Hubble API endpoints
    │
    ├── data/                      # Dataset & storage directory
    │   ├── embeddings/            # Generated face embedding files (.pkl)
    │   ├── faces_v2/              # Captured/enrolled face images by employee
    │   ├── employee_directory.json# HR & Manager email mapping directory
    │   └── mock_employees.json    # Mock employee corporate records
    │
    ├── hr-dashboard/              # Frontend web application (React + Vite)
    │   ├── src/
    │   │   ├── App.jsx            # Main HR dashboard application component
    │   │   ├── App.css / index.css# Dashboard styling
    │   │   └── main.jsx           # React app mount
    │   ├── package.json           # Node dependencies (React 19, Axios, Vite)
    │   └── vite.config.js         # Vite configuration
    │
    ├── api.py                     # Main attendance REST API (FastAPI, Port 8000)
    ├── attendance.py              # Attendance business logic & SQLite operations
    ├── camera_manager.py          # Multi-camera configuration & RTSP/webcam streaming
    ├── recognize_face_insightface.py # Real-time face detection & recognition pipeline
    ├── capture_faces_insightface.py # Face capture/enrollment CLI tool
    ├── generate_embeddings.py     # Script to extract & compile facial embeddings
    ├── train_model.py             # Model training / embedding generation script
    ├── email_service.py           # Automated email dispatch (early arrivals, HR notifications)
    ├── report_service.py          # Weekly & monthly XLSX Excel report generator
    │
    ├── attendance.db              # SQLite DB: employees, attendance events, leaves
    ├── camera_config.db           # SQLite DB: camera configurations & locations
    ├── requirements.txt           # Python dependencies
    ├── run_backend.sh             # Launch script for the main API (uvicorn api:app)
    ├── run_hubble.sh              # Launch script for Hubble mock API (uvicorn app.main:app)
    │
    ├── test_cameras.py            # Utility to test camera hardware/RTSP feeds
    ├── test_insightface.py        # InsightFace pipeline test
    ├── test_liveness.py           # Face liveness verification test
    ├── test_email.py              # Email dispatch test
    ├── reset_test_attendance.py   # Helper to flush test attendance records
    └── view_attendance.py         # Terminal viewer for attendance records
Key Components & What They Do
1. Computer Vision & Face Recognition Pipeline


recognize_face_insightface.py
: Runs real-time face detection and recognition using InsightFace (ArcFace/ONNX) and OpenCV. Includes liveness checks, threshold tuning, and multi-camera stream processing.


capture_faces_insightface.py
: Captures facial image samples for new employees into 

faces_v2
.


generate_embeddings.py
 & 

train_model.py
: Extracts 512-dimensional facial feature vectors from employee photos and saves them in 

embeddings
 for fast similarity search.
2. Multi-Camera Management


camera_manager.py
: Handles multi-camera setups (webcams, RTSP streams, IP cameras). Allows tagging each camera by purpose (e.g., ENTRY gate or EXIT gate) and saves configs in 

camera_config.db
.
3. Core Attendance Engine & Main Backend


attendance.py
: Manages core database interactions in 

attendance.db
:
Employees: Personal details, shift start/end times, Hubble profile metadata.
Attendance Events: Timestamped IN and OUT events recorded by camera feeds.
Leave Management: Adding, retrieving, and canceling employee leaves.
Monthly Calendar Summaries: Classifying daily status (Full Day, Partial Day, Absent, Leave, Weekend).


api.py
: FastAPI application running on port 8000. Exposes REST endpoints for:
Live camera streaming (MJPEG video streams).
Employee enrollment, face uploads, and deletion.
Attendance records, shift updates, and calendar events.
Downloading Excel reports.
4. Hubble Corporate Directory Service


app/main.py
: A separate microservice running on port 8001 that mimics Hubble (corporate employee directory).


app/employee_service.py
: Enriches employee records with their reporting manager's and HR contact's email addresses using MongoDB and 

data/employee_directory.json
.
5. Reporting & Notifications


report_service.py
: Uses openpyxl to build formatted weekly and monthly Excel (.xlsx) workbooks for single employees or the entire organization.


email_service.py
: Sends automated email alerts (e.g., notifying managers and HR when an employee arrives early or checks in).
6. Frontend HR Dashboard


hr-dashboard/
: Built with React 19 and Vite. Provides a UI for HR administrators to:
Watch live video feeds with bounding-box face recognition overlays.
View real-time punch-in/out logs.
Manage employees, customize shift timings, and sync Hubble profiles.
Browse the monthly attendance calendar and log leaves.
Export weekly and monthly attendance reports.
How to Run the System
Component	Command	Port / Target
Hubble Service	bash Project_R/run_hubble.sh	http://127.0.0.1:8001
Attendance API	bash Project_R/run_backend.sh	http://127.0.0.1:8000
HR Web Dashboard	cd Project_R/hr-dashboard && npm run dev	http://localhost:5173
