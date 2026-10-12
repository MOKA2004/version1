import { useEffect, useMemo, useRef, useState } from "react";

const API_URL = "http://127.0.0.1:8000";
const HUBBLE_API_URL = "http://127.0.0.1:8001";

// ============================================================
// GUIDED FACE ENROLLMENT
// ------------------------------------------------------------
// The employee follows 9 head angles (like the reference sheet).
// A dotted oval is shown on the camera for every angle:
//   grey  = nobody inside the oval yet
//   red   = face is inside the oval but the angle is wrong
//   green = angle is right -> held briefly -> auto capture
//
// Only the face INSIDE the oval counts. Other people in the
// background are ignored, and are masked out of the saved photo.
//
// Angles are in degrees from the USER's point of view:
//   yaw   : negative = turn to your left,  positive = your right
//   pitch : positive = chin up,            negative = chin down
// ============================================================
const POSE_STEPS = [
  { id: "front",      label: "Front",       arrow: "●", yaw: 0,   pitch: 0,   row: 1, col: 1, hint: "Look straight at the camera" },
  { id: "left",       label: "Left 3/4",    arrow: "←", yaw: -30, pitch: 0,   row: 1, col: 0, hint: "Slowly turn your head to your left" },
  { id: "right",      label: "Right 3/4",   arrow: "→", yaw: 30,  pitch: 0,   row: 1, col: 2, hint: "Slowly turn your head to your right" },
  { id: "up",         label: "Up",          arrow: "↑", yaw: 0,   pitch: 15,  row: 0, col: 1, hint: "Lift your chin up a little" },
  { id: "up-left",    label: "Up + Left",   arrow: "↖", yaw: -28, pitch: 14,  row: 0, col: 0, hint: "Chin up and turn to your left" },
  { id: "up-right",   label: "Up + Right",  arrow: "↗", yaw: 28,  pitch: 14,  row: 0, col: 2, hint: "Chin up and turn to your right" },
  { id: "down",       label: "Down",        arrow: "↓", yaw: 0,   pitch: -15, row: 2, col: 1, hint: "Lower your chin a little" },
  { id: "down-left",  label: "Down + Left", arrow: "↙", yaw: -28, pitch: -14, row: 2, col: 0, hint: "Chin down and turn to your left" },
  { id: "down-right", label: "Down + Right",arrow: "↘", yaw: 28,  pitch: -14, row: 2, col: 2, hint: "Chin down and turn to your right" },
];

// ---- Tuning knobs -------------------------------------------
const OVAL_HEIGHT = 0.94;        // oval height / camera height (CSS uses the same 94%)
const OVAL_RATIO = 0.75;         // oval width / height (3:4)
const YAW_TOLERANCE = 12;        // degrees allowed around the target
const PITCH_TOLERANCE = 10;
const FRONT_YAW_LIMIT = 10;      // "front" is judged on raw angles
const FRONT_PITCH_LIMIT = 15;
const FRONT_YAW_CUE_LIMIT = 0.12;
const YAW_CUE_MIN = 0.04;        // min nose shift needed to trust a turn direction
const PITCH_CUE_MIN = 0.02;
const HOLD_FRAMES = 2;           // consecutive good frames before capture
const POLL_MS = 150;             // wait between frame checks
const CAPTURE_COOLDOWN_MS = 500; // pause after a capture
const MIN_FACE_RATIO = 0.40;     // face height / oval height (too far)
const MAX_FACE_RATIO = 1.0;      // (too close)
const MIN_DET_SCORE = 0.6;

const EMPTY_BASELINE = { yaw: 0, pitch: 0, yawCue: 0, pitchCue: 0 };

// Pick the face that is inside the oval (closest to its centre).
// Everybody else is "background" and is ignored.
function pickFaceInOval(faces, aspect) {
  const radiusY = OVAL_HEIGHT / 2;
  const radiusX = (OVAL_HEIGHT * OVAL_RATIO) / aspect / 2;

  let target = null;
  let best = Infinity;

  for (const face of faces) {
    const centerX = (face.bbox.x1 + face.bbox.x2) / 2;
    const centerY = (face.bbox.y1 + face.bbox.y2) / 2;

    const distance =
      ((centerX - 0.5) / radiusX) ** 2 + ((centerY - 0.5) / radiusY) ** 2;

    if (distance <= 1 && distance < best) {
      best = distance;
      target = face;
    }
  }

  return target;
}

// Decide whether the face in the oval is at the requested angle.
// Returns { ok, idle, message, target, yaw, pitch }.
function evaluatePose(step, data, baseline, aspect) {
  const faces = data?.faces || [];

  const target = pickFaceInOval(faces, aspect);

  if (!target) {
    return {
      ok: false,
      idle: true,
      target: null,
      message: "Put your face inside the circle.",
    };
  }

  if (target.det_score !== null && target.det_score < MIN_DET_SCORE) {
    return { ok: false, target, message: "Face is not clear. Improve the lighting." };
  }

  const faceRatio = (target.bbox.y2 - target.bbox.y1) / OVAL_HEIGHT;

  if (faceRatio < MIN_FACE_RATIO) {
    return { ok: false, target, message: "Move a little closer to the camera." };
  }

  if (faceRatio > MAX_FACE_RATIO) {
    return { ok: false, target, message: "Move back a little." };
  }

  if (!target.pose || target.yaw_cue === null || target.pitch_cue === null) {
    return { ok: false, idle: true, target, message: "Could not read head angle." };
  }

  const isFront = step.id === "front";

  // ---------- FRONT ----------
  if (isFront) {
    const yawOk =
      Math.abs(target.pose.yaw) <= FRONT_YAW_LIMIT &&
      Math.abs(target.yaw_cue) <= FRONT_YAW_CUE_LIMIT;

    const pitchOk = Math.abs(target.pose.pitch) <= FRONT_PITCH_LIMIT;

    if (yawOk && pitchOk) {
      return { ok: true, target, yaw: 0, pitch: 0, message: "Perfect. Hold still..." };
    }

    return {
      ok: false,
      target,
      yaw: 0,
      pitch: 0,
      message: !yawOk ? "Face the camera straight." : "Keep your head level.",
    };
  }

  // ---------- OTHER ANGLES ----------
  // Size comes from InsightFace (relative to the front photo).
  // DIRECTION comes from the landmark cues, so left can never be
  // mistaken for right whatever the sign convention is.
  const sizeYaw = Math.abs(target.pose.yaw - baseline.yaw);
  const sizePitch = Math.abs(target.pose.pitch - baseline.pitch);

  const dYawCue = target.yaw_cue - baseline.yawCue;
  const dPitchCue = target.pitch_cue - baseline.pitchCue;

  // nose moved to image-right => user turned to THEIR left (negative)
  let yaw = 0;
  if (Math.abs(dYawCue) >= YAW_CUE_MIN) {
    yaw = (dYawCue > 0 ? -1 : 1) * sizeYaw;
  }

  // nose closer to the eyes => chin up (positive)
  let pitch = 0;
  if (Math.abs(dPitchCue) >= PITCH_CUE_MIN) {
    pitch = (dPitchCue < 0 ? 1 : -1) * sizePitch;
  }

  const yawDiff = step.yaw - yaw;
  const pitchDiff = step.pitch - pitch;

  const yawOk = Math.abs(yawDiff) <= YAW_TOLERANCE;
  const pitchOk = Math.abs(pitchDiff) <= PITCH_TOLERANCE;

  if (yawOk && pitchOk) {
    return { ok: true, target, yaw, pitch, message: "Perfect. Hold still..." };
  }

  let message = step.hint;

  if (!yawOk) {
    const wrongWay = step.yaw !== 0 && yaw * step.yaw < 0 && Math.abs(yaw) > 8;
    const side = step.yaw !== 0 ? (step.yaw < 0 ? "left" : "right") : yawDiff < 0 ? "left" : "right";

    if (wrongWay) {
      message = `Wrong way. Turn to your ${side}.`;
    } else if (step.yaw === 0) {
      message = "Face straight, do not turn sideways.";
    } else if (Math.abs(yaw) > Math.abs(step.yaw)) {
      message = "Turn back a little.";
    } else {
      message = `Turn a bit more to your ${side}.`;
    }
  } else if (!pitchOk) {
    if (step.pitch === 0) {
      message = "Keep your head level.";
    } else if (pitchDiff > 0) {
      message = "Lift your chin a little more.";
    } else {
      message = "Lower your chin a little more.";
    }
  }

  return { ok: false, target, yaw, pitch, message };
}

function App() {
  const [employees, setEmployees] = useState([]);
  const [attendance, setAttendance] = useState([]);
  const [cameraStatus, setCameraStatus] = useState(null);
  const [cameraDraft, setCameraDraft] = useState({
    id: null,
    name: "",
    source_type: "webcam",
    source: "0",
    roles: ["ENTRY"],
    enabled: true,
  });
  const [showCameraEditor, setShowCameraEditor] = useState(false);
  const [cameraBusy, setCameraBusy] = useState(false);
  const [cameraActionError, setCameraActionError] = useState("");
  const [cameraActionMessage, setCameraActionMessage] = useState("");
  const [viewCamera, setViewCamera] = useState(null);

  const [loading, setLoading] = useState(true);
  const [cameraLoading, setCameraLoading] = useState(true);
  const [error, setError] = useState("");
  const [cameraError, setCameraError] = useState("");

  const [activePage, setActivePage] = useState("dashboard");
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("ALL");

  // Date used by the Attendance page.
  // Dashboard continues to use today's live attendance.
  const [selectedDate, setSelectedDate] = useState(
    new Date().toISOString().split("T")[0]
  );
  const [dateAttendanceLoading, setDateAttendanceLoading] = useState(false);
  const [dateAttendanceError, setDateAttendanceError] = useState("");

  const [selectedEmployee, setSelectedEmployee] = useState(null);
  const [attendanceHistory, setAttendanceHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [historyYear, setHistoryYear] = useState(new Date().getFullYear());
  const [historyMonth, setHistoryMonth] = useState(new Date().getMonth() + 1);
  const [selectedHistoryDay, setSelectedHistoryDay] = useState(null);
  const [reportChooser, setReportChooser] = useState(null);
  const [reportDate, setReportDate] = useState(new Date().toISOString().slice(0, 10));
  const [reportDownloading, setReportDownloading] = useState(false);
  const [reportError, setReportError] = useState("");

  const [showAddEmployee, setShowAddEmployee] = useState(false);
  const [employeeForm, setEmployeeForm] = useState({
    employee_id: "",
    employee_name: "",
    shift_start: "14:00",
    shift_end: "22:00",
    email: "",
    login_id: "",
  });

  const [formMessage, setFormMessage] = useState("");
  const [formError, setFormError] = useState("");
  const [deletingEmployeeId, setDeletingEmployeeId] = useState(null);
  const [hubbleLoading, setHubbleLoading] = useState(false);
  const [hubbleError, setHubbleError] = useState("");
  const [hubbleEmployee, setHubbleEmployee] = useState(null);
  const [editingEmployee, setEditingEmployee] = useState(null);
  const [editShiftForm, setEditShiftForm] = useState({ shift_start: "", shift_end: "" });
  const [editShiftLoading, setEditShiftLoading] = useState(false);
  const [editShiftError, setEditShiftError] = useState("");
  const [leaveSaving, setLeaveSaving] = useState(false);

  // Face enrollment / camera capture state.
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [cameraStream, setCameraStream] = useState(null);
  const [enrollmentCameraId, setEnrollmentCameraId] = useState("");
  const cameraImageRef = useRef(null);
  const [captureError, setCaptureError] = useState("");
  const [capturedPhotos, setCapturedPhotos] = useState([]);
  const [enrollmentLoading, setEnrollmentLoading] = useState(false);

  // Guided (auto) capture state.
  const [skippedSteps, setSkippedSteps] = useState([]);
  const [poseState, setPoseState] = useState({
    status: "idle", // idle (grey) | bad (red) | ok (green) | done (green flash)
    message: "Start the camera to begin.",
    debug: null,
  });
  const [showPoseDebug, setShowPoseDebug] = useState(false);
  const poseCanvasRef = useRef(null);
  const holdCountRef = useRef(0);
  const cooldownUntilRef = useRef(0);
  const baselineRef = useRef({ ...EMPTY_BASELINE });
  const [videoAspect, setVideoAspect] = useState(16 / 9);

  // The angle we are asking for right now (first one not captured/skipped).
  const currentStep = useMemo(
    () =>
      POSE_STEPS.find(
        (step) =>
          !capturedPhotos.some((photo) => photo.poseId === step.id) &&
          !skippedSteps.includes(step.id)
      ) || null,
    [capturedPhotos, skippedSteps]
  );

  const loadDashboardData = async () => {
    try {
      setError("");

      const [employeesResponse, attendanceResponse] = await Promise.all([
        fetch(`${API_URL}/employees`),
        fetch(`${API_URL}/attendance/today`),
      ]);

      if (!employeesResponse.ok) {
        throw new Error("Unable to load employees");
      }

      if (!attendanceResponse.ok) {
        throw new Error("Unable to load attendance");
      }

      const employeesData = await employeesResponse.json();
      const attendanceData = await attendanceResponse.json();

      setEmployees(Array.isArray(employeesData) ? employeesData : []);

      const attendanceList = Array.isArray(attendanceData)
        ? attendanceData
        : attendanceData?.attendance || attendanceData?.data || [];

      setAttendance(attendanceList);
    } catch (err) {
      setError(err.message || "Unable to connect to backend");
    } finally {
      setLoading(false);
    }
  };

  const loadAttendanceByDate = async (date) => {
    if (!date) return;

    try {
      setDateAttendanceLoading(true);
      setDateAttendanceError("");

      const response = await fetch(
        `${API_URL}/attendance/date/${date}`
      );

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(
          data.detail || "Unable to load attendance for selected date"
        );
      }

      const data = await response.json();

      const attendanceList = Array.isArray(data)
        ? data
        : data?.attendance || data?.data || [];

      setAttendance(attendanceList);
    } catch (err) {
      setDateAttendanceError(
        err.message || "Unable to load selected date"
      );
      setAttendance([]);
    } finally {
      setDateAttendanceLoading(false);
    }
  };

  const loadCameraStatus = async () => {
    try {
      setCameraError("");

      const response = await fetch(`${API_URL}/cameras/status`);

      if (!response.ok) {
        throw new Error("Unable to load camera status");
      }

      const data = await response.json();
      setCameraStatus(data);
    } catch (err) {
      setCameraError(err.message || "Unable to load camera status");
    } finally {
      setCameraLoading(false);
    }
  };

  const resetCameraDraft = () => {
    setCameraDraft({ id: null, name: "", source_type: "webcam", source: "0", roles: ["ENTRY"], enabled: true });
    setCameraActionError("");
    setCameraActionMessage("");
  };

  const saveCamera = async (event) => {
    event.preventDefault();
    setCameraBusy(true);
    setCameraActionError("");
    setCameraActionMessage("");
    try {
      const response = await fetch(`${API_URL}/cameras${cameraDraft.id ? `/${cameraDraft.id}` : ""}`, {
        method: cameraDraft.id ? "PUT" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: cameraDraft.name,
          source_type: cameraDraft.source_type,
          source: cameraDraft.source,
          roles: cameraDraft.roles,
          enabled: cameraDraft.enabled,
        }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "Unable to save camera");
      setShowCameraEditor(false);
      resetCameraDraft();
      setCameraActionMessage(cameraDraft.id ? "Camera updated." : "Camera added.");
      await loadCameraStatus();
    } catch (err) {
      setCameraActionError(err.message || "Unable to save camera");
    } finally {
      setCameraBusy(false);
    }
  };

  const editCamera = (camera) => {
    setCameraDraft({ ...camera, source: "", roles: camera.roles || [] });
    setCameraActionError("");
    setCameraActionMessage("");
    setShowCameraEditor(true);
  };

  const toggleCameraEnabled = async (camera) => {
    try {
      const response = await fetch(`${API_URL}/cameras/${camera.id}/enabled`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !camera.enabled }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "Unable to update camera");
      await loadCameraStatus();
    } catch (err) {
      setCameraActionError(err.message || "Unable to update camera");
    }
  };

  const deleteCamera = async (camera) => {
    if (!window.confirm(`Delete camera “${camera.name}”?`)) return;
    try {
      if (viewCamera?.id === camera.id) setViewCamera(null);
      const response = await fetch(`${API_URL}/cameras/${camera.id}`, { method: "DELETE" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "Unable to delete camera");
      await loadCameraStatus();
      if (viewCamera?.id === camera.id) setViewCamera(null);
    } catch (err) {
      setCameraActionError(err.message || "Unable to delete camera");
    }
  };

  const testCamera = async (camera) => {
    try {
      setCameraActionError("");
      const response = await fetch(`${API_URL}/cameras/${camera.id}/test`, { method: "POST" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "Camera connection failed");
      setCameraActionMessage(`${camera.name}: connected successfully.`);
    } catch (err) {
      setCameraActionError(err.message || "Camera connection failed");
    }
  };

  const handleCameraDraftChange = (event) => {
    const { name, value, type, checked } = event.target;
    setCameraDraft((previous) => ({ ...previous, [name]: type === "checkbox" ? checked : value }));
  };

  const toggleCameraRole = (role) => {
    setCameraDraft((previous) => ({
      ...previous,
      roles: previous.roles.includes(role)
        ? previous.roles.filter((item) => item !== role)
        : [...previous.roles, role],
    }));
  };

  const openCameraPreview = (camera) => {
    setViewCamera({ ...camera, streamId: globalThis.crypto.randomUUID() });
  };

  const closeCameraPreview = () => setViewCamera(null);

  useEffect(() => {
    if (!viewCamera) return undefined;
    const { id, streamId } = viewCamera;
    return () => {
      fetch(`${API_URL}/cameras/${id}/stream/${streamId}/stop`, {
        method: "POST",
        keepalive: true,
      }).catch(() => {});
    };
  }, [viewCamera]);

  const handleAttendanceDateChange = (event) => {
    const date = event.target.value;

    setSelectedDate(date);

    if (date) {
      loadAttendanceByDate(date);
    }
  };

  useEffect(() => {
    if (activePage === "attendance") {
      loadAttendanceByDate(selectedDate);
    } else {
      loadDashboardData();
    }

    loadCameraStatus();

    const interval = setInterval(() => {
      loadCameraStatus();

      if (activePage === "dashboard") {
        loadDashboardData();
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [activePage]);

  const handleEmployeeFormChange = (event) => {
    const { name, value } = event.target;

    setEmployeeForm((previous) => ({
      ...previous,
      [name]: value,
    }));

    if (name === "login_id") {
      setHubbleError("");
      setHubbleEmployee(null);
    }
  };

  const fetchHubbleEmployee = async (loginIdOverride = null) => {
    const loginId = String(
      loginIdOverride ?? employeeForm.login_id ?? ""
    ).trim();

    if (!loginId) {
      setHubbleError("Please enter a Hubble Login ID.");
      return;
    }

    setHubbleLoading(true);
    setHubbleError("");
    setFormError("");

    try {
      const response = await fetch(
        `${HUBBLE_API_URL}/mock-hubble/employee/${encodeURIComponent(loginId)}`,
        { headers: { Accept: "application/json" } }
      );
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(
          data?.detail || `Hubble employee not found (${response.status})`
        );
      }

      const normalizedEmployee = {
        ...data,
        employee_id: data.id ?? "",
        employee_name: data.name ?? "",
        email: data.mailId ?? "",
        login_id: data.loginId ?? loginId,
        reporting_manager: data.reportsTo ?? "",
        hr_contact: data.hrContact ?? "",
        department: data.department ?? "",
        designation: data.designation ?? "",
        team: data.teamName ?? "",
      };

      setHubbleEmployee(normalizedEmployee);
      setEmployeeForm((previous) => ({
        ...previous,
        login_id: normalizedEmployee.login_id,
        employee_id: normalizedEmployee.employee_id !== ""
          ? String(normalizedEmployee.employee_id)
          : previous.employee_id,
        employee_name: normalizedEmployee.employee_name || previous.employee_name,
        email: normalizedEmployee.email || previous.email,
      }));
      setFormMessage(
        `Mock Hubble employee found: ${normalizedEmployee.employee_name || loginId}`
      );
    } catch (error) {
      setHubbleEmployee(null);
      setHubbleError(error.message || "Unable to fetch employee details from Hubble.");
    } finally {
      setHubbleLoading(false);
    }
  };

  const handleHubbleLoginKeyDown = (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      fetchHubbleEmployee();
    }
  };

  const startEnrollmentCamera = async () => {
    setCaptureError("");

    try {
      if (cameraStream) {
        cameraStream.getTracks?.().forEach((track) => track.stop());
      }

      if (enrollmentCameraId) {
        const response = await fetch(`${API_URL}/cameras/${enrollmentCameraId}/test`, { method: "POST" });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || "Selected camera is not connected");
        setCameraStream({ cameraId: enrollmentCameraId, streamId: globalThis.crypto.randomUUID() });
        return;
      }

      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera access is not supported by this browser.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });

      setCameraStream(stream);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (err) {
      setCaptureError(
        err.message ||
          "Unable to access the camera. Please allow camera permission."
      );
    }
  };

  const stopEnrollmentCamera = () => {
    if (cameraStream?.cameraId && cameraStream?.streamId) {
      fetch(`${API_URL}/cameras/${cameraStream.cameraId}/stream/${cameraStream.streamId}/stop`, {
        method: "POST",
        keepalive: true,
      }).catch(() => {});
    }
    if (cameraStream) {
      cameraStream.getTracks?.().forEach((track) => track.stop());
    }

    setCameraStream(null);

    holdCountRef.current = 0;
    setPoseState({
      status: "idle",
      message: "Start the camera to begin.",
      debug: null,
    });

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (cameraImageRef.current) cameraImageRef.current.src = "";
  };

  useEffect(() => {
    if (cameraStream?.getTracks && videoRef.current) {
      videoRef.current.srcObject = cameraStream;
      videoRef.current.play().catch(() => {});
    }
  }, [cameraStream]);

  useEffect(() => {
    const image = cameraImageRef.current;
    if (cameraStream?.cameraId && image) {
      image.src = `${API_URL}/cameras/${cameraStream.cameraId}/stream.mjpg?stream_id=${encodeURIComponent(cameraStream.streamId)}`;
    }
    return () => {
      if (image) image.src = "";
      if (cameraStream?.cameraId && cameraStream?.streamId) {
        fetch(`${API_URL}/cameras/${cameraStream.cameraId}/stream/${cameraStream.streamId}/stop`, {
          method: "POST",
          keepalive: true,
        }).catch(() => {});
      }
    };
  }, [cameraStream]);


  useEffect(() => {
    return () => {
      if (cameraStream) {
        cameraStream.getTracks?.().forEach((track) => track.stop());
      }
    };
  }, [cameraStream]);

  const resetEnrollmentState = () => {
    stopEnrollmentCamera();

    capturedPhotos.forEach((photo) => {
      if (photo.previewUrl) {
        URL.revokeObjectURL(photo.previewUrl);
      }
    });

    setCapturedPhotos([]);
    setSkippedSteps([]);
    baselineRef.current = { ...EMPTY_BASELINE };
    holdCountRef.current = 0;
    setCaptureError("");
    setEnrollmentLoading(false);
  };

  const closeAddEmployee = () => {
    resetEnrollmentState();
    setShowAddEmployee(false);
    setFormMessage("");
    setFormError("");
  };

  const getEnrollmentImageElement = () =>
    enrollmentCameraId ? cameraImageRef.current : videoRef.current;

  // Small frame (fast to upload) used only to read the head angle.
  const grabPoseFrameBlob = () =>
    new Promise((resolve) => {
      const video = getEnrollmentImageElement();
      const width = video?.videoWidth || video?.naturalWidth || 0;
      const height = video?.videoHeight || video?.naturalHeight || 0;
      const ready = enrollmentCameraId
        ? video?.complete && width > 0
        : video?.readyState >= 2 && width > 0;

      if (!video || !ready) {
        resolve(null);
        return;
      }

      if (!poseCanvasRef.current) {
        poseCanvasRef.current = document.createElement("canvas");
      }

      const canvas = poseCanvasRef.current;
      const scale = 480 / width;

      canvas.width = 480;
      canvas.height = Math.round(height * scale);

      const context = canvas.getContext("2d");

      if (!context) {
        resolve(null);
        return;
      }

      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.7);
    });

  // Full-resolution photo of ONLY the employee:
  //  - every other face in the frame is painted over
  //  - the picture is cropped around the employee's face
  // (the enrollment API rejects photos that contain 2+ faces)
  const grabEmployeeCropBlob = (target, others) =>
    new Promise((resolve) => {
      const video = getEnrollmentImageElement();
      const canvas = canvasRef.current;
      const width = video?.videoWidth || video?.naturalWidth || 0;
      const height = video?.videoHeight || video?.naturalHeight || 0;
      const ready = enrollmentCameraId
        ? video?.complete && width > 0
        : video?.readyState >= 2 && width > 0;

      if (!video || !canvas || !ready) {
        resolve(null);
        return;
      }

      canvas.width = width;
      canvas.height = height;

      const context = canvas.getContext("2d");

      if (!context) {
        resolve(null);
        return;
      }

      context.drawImage(video, 0, 0, width, height);

      const box = target.bbox;
      const faceW = (box.x2 - box.x1) * width;
      const faceH = (box.y2 - box.y1) * height;
      const centerX = ((box.x1 + box.x2) / 2) * width;
      const centerY = ((box.y1 + box.y2) / 2) * height;

      // paint over the other people's faces
      context.fillStyle = "#808080";

      others.forEach((other) => {
        const ob = other.bbox;
        const ocx = ((ob.x1 + ob.x2) / 2) * width;
        const ocy = ((ob.y1 + ob.y2) / 2) * height;

        // never paint over the employee's own face
        const insideTarget =
          ocx > centerX - faceW / 2 &&
          ocx < centerX + faceW / 2 &&
          ocy > centerY - faceH / 2 &&
          ocy < centerY + faceH / 2;

        if (insideTarget) return;

        const ow = (ob.x2 - ob.x1) * width;
        const oh = (ob.y2 - ob.y1) * height;

        context.fillRect(
          ob.x1 * width - ow * 0.15,
          ob.y1 * height - oh * 0.15,
          ow * 1.3,
          oh * 1.3
        );
      });

      // crop around the employee
      const cropW = Math.min(width, faceW * 2.2);
      const cropH = Math.min(height, faceH * 2.0);

      const sx = Math.max(0, Math.min(width - cropW, centerX - cropW / 2));
      const sy = Math.max(0, Math.min(height - cropH, centerY - cropH / 2));

      const crop = document.createElement("canvas");

      crop.width = Math.round(cropW);
      crop.height = Math.round(cropH);

      crop
        .getContext("2d")
        .drawImage(canvas, sx, sy, cropW, cropH, 0, 0, crop.width, crop.height);

      crop.toBlob((blob) => resolve(blob), "image/jpeg", 0.92);
    });

  // Automatic capture loop: checks the angle again and again and
  // captures by itself when the requested angle is held.
  useEffect(() => {
    if (!cameraStream || !currentStep || enrollmentLoading) {
      return undefined;
    }

    let cancelled = false;
    let timer = null;
    const step = currentStep;

    holdCountRef.current = 0;

    setPoseState({
      status: "idle",
      message: step.hint,
      debug: null,
    });

    const tick = async () => {
      if (cancelled) return;

      try {
        if (Date.now() < cooldownUntilRef.current) return;

        const frame = await grabPoseFrameBlob();
        if (!frame || cancelled) return;

        const body = new FormData();
        body.append("file", frame, "frame.jpg");

        const response = await fetch(`${API_URL}/face-pose`, {
          method: "POST",
          body,
        });

        if (!response.ok) throw new Error("Pose check failed");

        const data = await response.json();
        if (cancelled) return;

        const video = getEnrollmentImageElement();
        const aspect =
          video && (video.videoHeight || video.naturalHeight)
            ? (video.videoWidth || video.naturalWidth) / (video.videoHeight || video.naturalHeight)
            : 16 / 9;

        const result = evaluatePose(step, data, baselineRef.current, aspect);
        const target = result.target;

        const debug =
          target && target.pose
            ? {
                yaw: result.yaw ?? 0,
                pitch: result.pitch ?? 0,
                yawCue: target.yaw_cue,
                pitchCue: target.pitch_cue,
                faces: data.faces.length,
              }
            : null;

        if (!result.ok) {
          holdCountRef.current = 0;

          setPoseState({
            status: result.idle ? "idle" : "bad",
            message: result.message,
            debug,
          });

          return;
        }

        holdCountRef.current += 1;

        if (holdCountRef.current < HOLD_FRAMES) {
          setPoseState({ status: "ok", message: result.message, debug });
          return;
        }

        // Angle held long enough -> capture only the employee.
        const others = data.faces.filter((face) => face !== target);
        const blob = await grabEmployeeCropBlob(target, others);

        if (!blob || cancelled) return;

        // The front photo is the reference for every other angle.
        if (step.id === "front") {
          baselineRef.current = {
            yaw: target.pose.yaw,
            pitch: target.pose.pitch,
            yawCue: target.yaw_cue,
            pitchCue: target.pitch_cue,
          };
        }

        const previewUrl = URL.createObjectURL(blob);

        holdCountRef.current = 0;
        cooldownUntilRef.current = Date.now() + CAPTURE_COOLDOWN_MS;

        setCapturedPhotos((previous) =>
          previous.some((photo) => photo.poseId === step.id)
            ? previous
            : [
                ...previous,
                {
                  blob,
                  previewUrl,
                  poseId: step.id,
                  filename: `face_${step.id}.jpg`,
                },
              ]
        );

        setPoseState({
          status: "done",
          message: `${step.label} captured. Next angle...`,
          debug,
        });
      } catch (err) {
        if (!cancelled) {
          setPoseState({
            status: "idle",
            message: "Cannot reach the face service. Is the API running?",
            debug: null,
          });
        }
      }
    };

    // Run one check at a time: the next one is scheduled only after
    // the previous one finished, so slow frames never pile up.
    const loop = async () => {
      await tick();

      if (!cancelled) {
        timer = setTimeout(loop, POLL_MS);
      }
    };

    loop();

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraStream, currentStep?.id, enrollmentLoading]);

  const skipCurrentStep = () => {
    if (!currentStep) return;
    setSkippedSteps((previous) => [...previous, currentStep.id]);
  };

  const removeCapturedPhoto = (index) => {
    setCapturedPhotos((previous) => {
      const photo = previous[index];

      if (photo?.previewUrl) {
        URL.revokeObjectURL(photo.previewUrl);
      }

      return previous.filter((_, photoIndex) => photoIndex !== index);
    });
  };

  const addEmployee = async (event) => {
    event.preventDefault();

    setFormMessage("");
    setFormError("");
    setCaptureError("");

    if (capturedPhotos.length < 5) {
      setCaptureError(
        "Please capture at least 5 angles before adding the employee."
      );
      return;
    }

    if (capturedPhotos.length > 10) {
      setCaptureError("You can capture a maximum of 10 photos.");
      return;
    }

    setEnrollmentLoading(true);

    let employeeCreated = false;

    try {
      // ----------------------------------------------------
      // STEP 1: CREATE EMPLOYEE
      // ----------------------------------------------------

      const employeeResponse = await fetch(`${API_URL}/employees`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(employeeForm),
      });

      const employeeData = await employeeResponse.json();

      if (!employeeResponse.ok) {
        throw new Error(
          employeeData.detail || "Unable to add employee"
        );
      }

      employeeCreated = true;

      // ----------------------------------------------------
      // STEP 2: SEND CAPTURED PHOTOS FOR FACE ENROLLMENT
      // ----------------------------------------------------

      const formData = new FormData();

      capturedPhotos.forEach((photo, index) => {
        formData.append(
          "files",
          photo.blob,
          photo.filename || `face_${index + 1}.jpg`
        );
      });

      const enrollmentResponse = await fetch(
        `${API_URL}/employees/${encodeURIComponent(
          employeeForm.employee_id
        )}/enroll-face`,
        {
          method: "POST",
          body: formData,
        }
      );

      const enrollmentData = await enrollmentResponse.json();

      if (!enrollmentResponse.ok) {
        const detail =
          typeof enrollmentData.detail === "string"
            ? enrollmentData.detail
            : enrollmentData.detail?.message ||
              "Face enrollment failed.";

        throw new Error(detail);
      }

      setFormMessage(
        `Employee added and face enrolled successfully. ${enrollmentData.photos_processed || capturedPhotos.length} photos processed.`
      );

      // ----------------------------------------------------
      // CLEANUP
      // ----------------------------------------------------

      resetEnrollmentState();

      setEmployeeForm({
        employee_id: "",
        employee_name: "",
        shift_start: "14:00",
        shift_end: "22:00",
        email: "",
        login_id: "",
      });

      setHubbleEmployee(null);
      setHubbleError("");

      await loadDashboardData();

      setTimeout(() => {
        setShowAddEmployee(false);
        setFormMessage("");
      }, 1500);
    } catch (err) {
      if (employeeCreated) {
        setFormError(
          `Employee was created, but face enrollment failed: ${
            err.message || "Unknown enrollment error"
          }`
        );
      } else {
        setFormError(err.message || "Unable to add employee");
      }
    } finally {
      setEnrollmentLoading(false);
    }
  };


  const loadEmployeeAttendanceHistory = async (employee, year, month) => {
    setHistoryLoading(true);
    setHistoryError("");
    try {
      const response = await fetch(
        `${API_URL}/employees/${encodeURIComponent(employee.employee_id)}/attendance-history?year=${year}&month=${month}`
      );
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || "Unable to load monthly attendance.");
      }
      const records = Array.isArray(data.attendance) ? data.attendance : [];
      setAttendanceHistory(records);
      setSelectedHistoryDay((previous) =>
        previous ? records.find((day) => day.date === previous.date) || null : null
      );
      return records;
    } catch (err) {
      setAttendanceHistory([]);
      setHistoryError(err.message || "Unable to load monthly attendance.");
      return [];
    } finally {
      setHistoryLoading(false);
    }
  };

  const openEmployeeHistory = async (employee) => {
    const now = new Date();
    const year = now.getFullYear();
    const month = now.getMonth() + 1;
    setSelectedEmployee(employee);
    setHistoryYear(year);
    setHistoryMonth(month);
    setSelectedHistoryDay(null);
    setAttendanceHistory([]);
    await loadEmployeeAttendanceHistory(employee, year, month);
  };

  const changeHistoryMonth = async (offset) => {
    if (!selectedEmployee) return;
    const next = new Date(historyYear, historyMonth - 1 + offset, 1);
    const year = next.getFullYear();
    const month = next.getMonth() + 1;
    setHistoryYear(year);
    setHistoryMonth(month);
    setSelectedHistoryDay(null);
    await loadEmployeeAttendanceHistory(selectedEmployee, year, month);
  };

  const openEditShift = (employee) => {
    setEditingEmployee(employee);
    setEditShiftForm({
      shift_start: employee.shift_start || "09:00",
      shift_end: employee.shift_end || "17:00",
    });
    setEditShiftError("");
  };

  const saveEmployeeShift = async (event) => {
    event.preventDefault();
    if (!editingEmployee) return;
    setEditShiftLoading(true);
    setEditShiftError("");
    try {
      const response = await fetch(
        `${API_URL}/employees/${encodeURIComponent(editingEmployee.employee_id)}/shift`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(editShiftForm),
        }
      );
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "Unable to update shift.");
      setEmployees((previous) => previous.map((employee) =>
        String(employee.employee_id) === String(data.employee.employee_id)
          ? { ...employee, ...data.employee }
          : employee
      ));
      setEditingEmployee(null);
      await loadDashboardData();
    } catch (err) {
      setEditShiftError(err.message || "Unable to update shift.");
    } finally {
      setEditShiftLoading(false);
    }
  };

  const saveSelectedDayLeave = async () => {
    if (!selectedEmployee || !selectedHistoryDay) return;
    if (selectedHistoryDay.status === "WEEKEND") {
      setHistoryError("Leave records cannot be added to weekends.");
      return;
    }
    const leaveType = window.prompt("Leave type", selectedHistoryDay.leave_type || "Leave");
    if (leaveType === null) return;
    const reason = window.prompt("Reason (optional)", selectedHistoryDay.reason || "");
    if (reason === null) return;
    setLeaveSaving(true);
    setHistoryError("");
    try {
      const response = await fetch(
        `${API_URL}/employees/${encodeURIComponent(selectedEmployee.employee_id)}/leave`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            employee_id: String(selectedEmployee.employee_id),
            leave_date: selectedHistoryDay.date,
            leave_type: leaveType.trim() || "Leave",
            reason: reason.trim() || null,
          }),
        }
      );
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "Unable to save leave.");
      const records = await loadEmployeeAttendanceHistory(selectedEmployee, historyYear, historyMonth);
      setSelectedHistoryDay(records.find((day) => day.date === selectedHistoryDay.date) || null);
    } catch (err) {
      setHistoryError(err.message || "Unable to save leave.");
    } finally {
      setLeaveSaving(false);
    }
  };

  const removeSelectedDayLeave = async () => {
    if (!selectedEmployee || !selectedHistoryDay || selectedHistoryDay.status !== "LEAVE") return;
    if (!window.confirm(`Remove leave recorded for ${selectedHistoryDay.date}?`)) return;
    setLeaveSaving(true);
    setHistoryError("");
    try {
      const response = await fetch(
        `${API_URL}/employees/${encodeURIComponent(selectedEmployee.employee_id)}/leave/${selectedHistoryDay.date}`,
        { method: "DELETE" }
      );
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "Unable to remove leave.");
      const records = await loadEmployeeAttendanceHistory(selectedEmployee, historyYear, historyMonth);
      setSelectedHistoryDay(records.find((day) => day.date === selectedHistoryDay.date) || null);
    } catch (err) {
      setHistoryError(err.message || "Unable to remove leave.");
    } finally {
      setLeaveSaving(false);
    }
  };

  const openReportChooser = (employee = null) => {
    setReportChooser(employee ? { employee } : { employee: null });
    setReportDate(selectedDate || new Date().toISOString().slice(0, 10));
    setReportError("");
  };

  const downloadAttendanceReport = async (period) => {
    if (!reportChooser || reportDownloading) return;
    setReportDownloading(true);
    setReportError("");
    try {
      const employee = reportChooser.employee;
      const endpoint = employee
        ? `/reports/employee/${encodeURIComponent(employee.employee_id)}`
        : "/reports/attendance";
      const response = await fetch(
        `${API_URL}${endpoint}?period=${period}&date=${encodeURIComponent(reportDate)}`
      );
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || "Unable to download attendance report.");
      }
      const blob = await response.blob();
      const disposition = response.headers.get("Content-Disposition") || "";
      const filename = disposition.match(/filename="?([^";]+)"?/i)?.[1]
        || `${employee ? employee.employee_id : "all_employees"}_${period}_attendance.xlsx`;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      setReportChooser(null);
    } catch (err) {
      setReportError(err.message || "Unable to download attendance report.");
    } finally {
      setReportDownloading(false);
    }
  };

  const deleteEmployee = async (employee) => {
    const confirmed = window.confirm(
      `Are you sure you want to delete ${employee.employee_name}?\n\n` +
        `This will permanently remove the employee, face embeddings, ` +
        `attendance events, and attendance summary.`
    );

    if (!confirmed) {
      return;
    }

    try {
      setDeletingEmployeeId(String(employee.employee_id));
      setError("");

      const response = await fetch(
        `${API_URL}/employees/${employee.employee_id}`,
        {
          method: "DELETE",
        }
      );

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          data.detail || "Unable to delete employee"
        );
      }

      if (
        selectedEmployee &&
        String(selectedEmployee.employee_id) ===
          String(employee.employee_id)
      ) {
        closeHistory();
      }

      await loadDashboardData();
    } catch (err) {
      setError(
        err.message || "Unable to delete employee"
      );
    } finally {
      setDeletingEmployeeId(null);
    }
  };

  const closeHistory = () => {
    setSelectedEmployee(null);
    setAttendanceHistory([]);
    setSelectedHistoryDay(null);
    setHistoryError("");
  };

  const getAttendance = (employeeId) => {
    return attendance.find(
      (record) => String(record.employee_id) === String(employeeId)
    );
  };

  const getStatus = (record) => {
    if (!record) return "ABSENT";

    // /attendance/today includes a summary row for every employee,
    // including employees who have not checked in today.
    if (!record.first_in && Number(record.event_count || 0) === 0) {
      return "ABSENT";
    }

    const status = String(record.arrival_status || "").toUpperCase();

    if (status === "EARLY") return "EARLY";
    if (status === "LATE") return "LATE";
    if (status === "ON_TIME" || status === "ON TIME") return "ON_TIME";

    return status || (record.first_in ? "PRESENT" : "ABSENT");
  };

  const formatStatus = (status) => {
    if (status === "ON_TIME") return "ON TIME";
    if (status === "ABSENT") return "ABSENT";
    return status;
  };

  const filteredEmployees = useMemo(() => {
    return employees.filter((employee) => {
      const employeeAttendance = getAttendance(employee.employee_id);
      const status = getStatus(employeeAttendance);

      const query = search.toLowerCase().trim();

      const matchesSearch =
        !query ||
        String(employee.employee_id).toLowerCase().includes(query) ||
        String(employee.employee_name || "")
          .toLowerCase()
          .includes(query);

      let matchesFilter = true;

      if (statusFilter === "ALL") {
        matchesFilter = true;
      } else if (statusFilter === "PRESENT") {
        matchesFilter = status !== "ABSENT";
      } else if (statusFilter === "EARLY") {
        matchesFilter = status === "EARLY";
      } else if (statusFilter === "LATE") {
        matchesFilter = status === "LATE";
      } else if (statusFilter === "ON_TIME") {
        matchesFilter = status === "ON_TIME";
      } else if (statusFilter === "INSIDE") {
        matchesFilter =
          String(employeeAttendance?.current_state || "").toUpperCase() ===
          "INSIDE";
      } else if (statusFilter === "OUTSIDE") {
        matchesFilter =
          String(employeeAttendance?.current_state || "").toUpperCase() ===
          "OUTSIDE";
      }

      return matchesSearch && matchesFilter;
    });
  }, [employees, attendance, search, statusFilter]);

  const totalEmployees = employees.length;

  const presentToday = attendance.filter(
    (record) => getStatus(record) !== "ABSENT"
  ).length;

  const lateToday = attendance.filter(
    (record) => getStatus(record) === "LATE"
  ).length;

  const insideNow = attendance.filter(
    (record) =>
      String(record.current_state || "").toUpperCase() === "INSIDE"
  ).length;

  const cameras = cameraStatus?.cameras || [];
  const enrollmentCameras = cameras.filter((camera) => camera.enabled);

  const formatMinutes = (minutes) => {
    const value = Number(minutes || 0);

    const hours = Math.floor(value / 60);
    const mins = value % 60;

    if (hours === 0) return `${mins}m`;

    return `${hours}h ${mins}m`;
  };

  const formatWorkedTime = (record) => {
    if (record?.worked_hours) {
      return record.worked_hours;
    }

    return formatMinutes(record?.worked_minutes);
  };

  const formatTime = (time) => {
    if (!time) return "--";

    const parts = String(time).split(":");

    if (parts.length < 2) return time;

    let hour = Number(parts[0]);
    const minute = parts[1];

    const suffix = hour >= 12 ? "PM" : "AM";

    hour = hour % 12 || 12;

    return `${hour}:${minute} ${suffix}`;
  };

  const todayText = new Date().toLocaleDateString("en-IN", {
    weekday: "long",
    day: "2-digit",
    month: "long",
    year: "numeric",
  });

  const selectedDateText = selectedDate
    ? new Date(`${selectedDate}T00:00:00`).toLocaleDateString("en-IN", {
        weekday: "long",
        day: "2-digit",
        month: "long",
        year: "numeric",
      })
    : "";

  const statusClass = (status) => {
    switch (status) {
      case "EARLY":
        return "status-early";

      case "LATE":
        return "status-late";

      case "ON_TIME":
        return "status-ontime";

      case "INSIDE":
        return "status-inside";

      case "OUTSIDE":
        return "status-outside";

      case "ABSENT":
        return "status-absent";

      default:
        return "status-neutral";
    }
  };

  return (
    <div className="app-shell">
      <style>{`
        * {
          box-sizing: border-box;
        }

        body {
          margin: 0;
          font-family: Inter, ui-sans-serif, system-ui, -apple-system,
            BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #f5f7fb;
          color: #172033;
        }

        button,
        input,
        select {
          font: inherit;
        }

        button {
          cursor: pointer;
        }

        .app-shell {
          min-height: 100vh;
          display: flex;
          background: #f5f7fb;
        }

        .sidebar {
          width: 245px;
          min-height: 100vh;
          background: #101828;
          color: white;
          padding: 24px 16px;
          position: fixed;
          left: 0;
          top: 0;
          bottom: 0;
          z-index: 20;
        }

        .brand {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 0 10px 28px;
          border-bottom: 1px solid rgba(255,255,255,.08);
        }

        .brand-icon {
          width: 42px;
          height: 42px;
          border-radius: 12px;
          background: linear-gradient(135deg, #4f46e5, #7c3aed);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 20px;
          font-weight: 800;
        }

        .brand-title {
          font-size: 18px;
          font-weight: 750;
          letter-spacing: -.3px;
        }

        .brand-subtitle {
          font-size: 11px;
          color: #98a2b3;
          margin-top: 2px;
        }

        .nav-title {
          color: #667085;
          font-size: 10px;
          font-weight: 700;
          letter-spacing: 1px;
          margin: 28px 10px 10px;
          text-transform: uppercase;
        }

        .nav-item {
          width: 100%;
          border: 0;
          background: transparent;
          color: #98a2b3;
          padding: 12px 13px;
          border-radius: 9px;
          display: flex;
          align-items: center;
          gap: 12px;
          text-align: left;
          margin-bottom: 4px;
          transition: .2s ease;
        }

        .nav-item:hover {
          background: rgba(255,255,255,.05);
          color: white;
        }

        .nav-item.active {
          background: #344054;
          color: white;
        }

        .nav-icon {
          width: 20px;
          text-align: center;
          font-size: 15px;
        }

        .sidebar-footer {
          position: absolute;
          left: 16px;
          right: 16px;
          bottom: 20px;
          border-top: 1px solid rgba(255,255,255,.08);
          padding: 18px 10px 0;
          color: #667085;
          font-size: 11px;
          line-height: 1.6;
        }

        .main {
          width: calc(100% - 245px);
          margin-left: 245px;
          min-height: 100vh;
        }

        .topbar {
          height: 72px;
          background: white;
          border-bottom: 1px solid #eaecf0;
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0 32px;
          position: sticky;
          top: 0;
          z-index: 10;
        }

        .breadcrumb {
          color: #667085;
          font-size: 13px;
        }

        .topbar-right {
          display: flex;
          align-items: center;
          gap: 18px;
        }

        .system-status {
          display: flex;
          align-items: center;
          gap: 7px;
          color: #344054;
          font-size: 13px;
          font-weight: 600;
        }

        .online-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #12b76a;
          box-shadow: 0 0 0 3px #dcfae6;
        }

        .refresh-button {
          border: 1px solid #d0d5dd;
          background: white;
          color: #344054;
          padding: 8px 12px;
          border-radius: 8px;
          font-size: 12px;
          font-weight: 600;
        }

        .refresh-button:hover {
          background: #f9fafb;
        }

        .content {
          padding: 32px;
          max-width: 1700px;
          margin: auto;
        }

        .page-heading {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 28px;
        }

        .page-title {
          margin: 0;
          font-size: 27px;
          letter-spacing: -.6px;
          color: #101828;
        }

        .page-description {
          margin: 7px 0 0;
          color: #667085;
          font-size: 13px;
        }

        .primary-button {
          border: 0;
          background: #4f46e5;
          color: white;
          border-radius: 8px;
          padding: 11px 16px;
          font-weight: 650;
          font-size: 13px;
          box-shadow: 0 2px 5px rgba(79,70,229,.2);
        }

        .primary-button:hover {
          background: #4338ca;
        }

        .stats-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 16px;
          margin-bottom: 24px;
        }

        .stat-card {
          background: white;
          border: 1px solid #eaecf0;
          border-radius: 12px;
          padding: 20px;
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
        }

        .stat-label {
          color: #667085;
          font-size: 12px;
          font-weight: 600;
        }

        .stat-value {
          font-size: 27px;
          font-weight: 750;
          color: #101828;
          margin-top: 8px;
        }

        .stat-icon {
          width: 40px;
          height: 40px;
          border-radius: 10px;
          background: #eef2ff;
          color: #4f46e5;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: 800;
        }

        .section {
          margin-bottom: 24px;
        }

        .section-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 13px;
        }

        .section-title {
          font-size: 15px;
          font-weight: 700;
          color: #101828;
        }

        .section-caption {
          color: #667085;
          font-size: 12px;
        }

        .camera-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 16px;
        }

        .camera-card {
          background: white;
          border: 1px solid #eaecf0;
          border-radius: 12px;
          padding: 20px;
        }

        .camera-config-form {
          margin-top: 16px;
          padding: 20px;
          background: white;
          border: 1px solid #eaecf0;
          border-radius: 12px;
        }

        .camera-config-title {
          margin: 0;
          color: #101828;
          font-size: 14px;
          font-weight: 750;
        }

        .camera-config-help {
          margin: 6px 0 16px;
          color: #667085;
          font-size: 11px;
          line-height: 1.6;
        }

        .camera-config-fields {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 14px;
        }

        .camera-config-field label {
          display: block;
          margin-bottom: 6px;
          color: #344054;
          font-size: 11px;
          font-weight: 650;
        }

        .camera-config-field input {
          width: 100%;
          border: 1px solid #d0d5dd;
          border-radius: 8px;
          padding: 10px;
          outline: none;
          font-size: 12px;
        }

        .camera-config-field input:focus {
          border-color: #7f56d9;
          box-shadow: 0 0 0 3px #f4f3ff;
        }

        .camera-config-actions {
          display: flex;
          justify-content: flex-end;
          margin-top: 14px;
        }

        .camera-management-toolbar {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 16px;
          margin: 4px 0 14px;
        }

        .camera-empty {
          padding: 28px;
          border: 1px dashed #d0d5dd;
          border-radius: 12px;
          color: #667085;
          background: #fff;
          text-align: center;
          font-size: 13px;
        }

        .camera-card-disabled { opacity: .72; }

        .camera-card-actions {
          display: flex;
          flex-wrap: wrap;
          gap: 7px;
          margin-top: 16px;
          padding-top: 14px;
          border-top: 1px solid #f2f4f7;
        }

        .camera-action-button {
          border: 1px solid #d0d5dd;
          background: #fff;
          color: #344054;
          border-radius: 7px;
          padding: 7px 10px;
          font-size: 11px;
          font-weight: 650;
          cursor: pointer;
        }

        .camera-action-button:hover { background: #f9fafb; }
        .camera-action-button:disabled { opacity: .5; cursor: not-allowed; }
        .camera-action-button.danger { color: #b42318; border-color: #fecdca; }

        .camera-editor-modal, .camera-view-modal {
          width: min(520px, calc(100vw - 32px));
          max-height: calc(100vh - 40px);
          overflow: auto;
          padding: 24px;
          border-radius: 16px;
          background: #fff;
          box-shadow: 0 24px 60px rgba(16,24,40,.25);
        }

        .camera-view-modal { width: min(920px, calc(100vw - 32px)); }
        .camera-view-modal > img { display: block; width: 100%; max-height: 70vh; object-fit: contain; background: #101828; border-radius: 10px; }

        .camera-editor-modal .modal-header, .camera-view-modal .modal-header {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 20px;
        }

        .camera-editor-modal .modal-header h2, .camera-view-modal .modal-header h2 { margin: 0; color: #101828; font-size: 20px; }
        .camera-editor-modal .modal-header p, .camera-view-modal .modal-header p { margin: 5px 0 0; color: #667085; font-size: 12px; }
        .camera-editor-modal .modal-close, .camera-view-modal .modal-close { border: 0; background: transparent; color: #667085; font-size: 25px; cursor: pointer; }

        .camera-form-label { display: block; margin: 0 0 15px; color: #344054; font-size: 12px; font-weight: 650; }
        .camera-form-label input, .camera-form-label select { display: block; width: 100%; margin-top: 6px; padding: 10px 11px; border: 1px solid #d0d5dd; border-radius: 8px; color: #101828; background: #fff; font: inherit; font-weight: 400; box-sizing: border-box; }
        .camera-form-hint { display: block; margin-top: 5px; color: #667085; font-size: 11px; font-weight: 400; }
        .camera-role-fieldset { display: flex; flex-wrap: wrap; gap: 16px; margin: 6px 0 14px; padding: 12px; border: 1px solid #eaecf0; border-radius: 9px; }
        .camera-role-fieldset legend { padding: 0 5px; color: #344054; font-size: 12px; font-weight: 650; }
        .camera-role-option { display: inline-flex; align-items: center; gap: 7px; color: #475467; font-size: 12px; }
        .camera-role-option input { accent-color: #4f46e5; }
        .camera-enabled-option { margin: 6px 0 14px; }
        .camera-editor-actions { display: flex; justify-content: flex-end; gap: 9px; margin-top: 16px; }

        @media (max-width: 700px) {
          .camera-config-fields {
            grid-template-columns: 1fr;
          }
          .camera-management-toolbar { flex-direction: column; }
        }

        .camera-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
        }

        .camera-name {
          font-size: 14px;
          font-weight: 750;
          color: #101828;
        }

        .camera-responsibility {
          color: #667085;
          font-size: 12px;
          margin-top: 5px;
        }

        .connection-badge {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 5px 9px;
          border-radius: 999px;
          font-size: 10px;
          font-weight: 700;
        }

        .connection-badge.connected {
          color: #067647;
          background: #ecfdf3;
        }

        .connection-badge.configured {
          color: #175cd3;
          background: #eff8ff;
        }

        .connection-badge.disconnected {
          color: #b42318;
          background: #fef3f2;
        }

        .camera-details {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
          margin-top: 20px;
          padding-top: 16px;
          border-top: 1px solid #f2f4f7;
        }

        .detail-label {
          font-size: 10px;
          text-transform: uppercase;
          letter-spacing: .5px;
          color: #98a2b3;
          font-weight: 700;
        }

        .detail-value {
          font-size: 12px;
          color: #344054;
          font-weight: 600;
          margin-top: 4px;
        }

        .table-card {
          background: white;
          border: 1px solid #eaecf0;
          border-radius: 12px;
          overflow: hidden;
        }

        .table-toolbar {
          padding: 16px;
          border-bottom: 1px solid #eaecf0;
          display: flex;
          justify-content: space-between;
          gap: 12px;
          flex-wrap: wrap;
        }

        .search-box {
          min-width: 260px;
          position: relative;
        }

        .search-box input {
          width: 100%;
          border: 1px solid #d0d5dd;
          border-radius: 8px;
          padding: 9px 12px 9px 34px;
          outline: none;
          font-size: 12px;
        }

        .search-box input:focus {
          border-color: #7f56d9;
          box-shadow: 0 0 0 3px #f4f3ff;
        }

        .search-icon {
          position: absolute;
          left: 12px;
          top: 8px;
          color: #98a2b3;
        }

        .filter-group {
          display: flex;
          gap: 5px;
          flex-wrap: wrap;
        }

        .filter-button {
          border: 1px solid #eaecf0;
          background: white;
          color: #667085;
          border-radius: 7px;
          padding: 7px 10px;
          font-size: 11px;
          font-weight: 600;
        }

        .filter-button.active {
          background: #eef2ff;
          color: #4338ca;
          border-color: #c7d2fe;
        }

        .table-wrapper {
          overflow-x: auto;
        }

        table {
          width: 100%;
          border-collapse: collapse;
          min-width: 850px;
        }

        th {
          background: #f9fafb;
          color: #667085;
          font-size: 10px;
          text-transform: uppercase;
          letter-spacing: .5px;
          text-align: left;
          padding: 12px 16px;
          font-weight: 700;
          border-bottom: 1px solid #eaecf0;
        }

        td {
          padding: 14px 16px;
          border-bottom: 1px solid #f2f4f7;
          color: #344054;
          font-size: 12px;
        }

        tr:last-child td {
          border-bottom: 0;
        }

        .employee-cell {
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .avatar {
          width: 34px;
          height: 34px;
          border-radius: 9px;
          background: #eef2ff;
          color: #4338ca;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: 750;
          font-size: 12px;
        }

        .employee-name {
          font-weight: 700;
          color: #101828;
        }

        .employee-id {
          color: #98a2b3;
          font-size: 10px;
          margin-top: 2px;
        }

        .status-badge {
          display: inline-flex;
          align-items: center;
          padding: 4px 8px;
          border-radius: 999px;
          font-size: 9px;
          font-weight: 750;
          letter-spacing: .2px;
        }

        .status-early {
          background: #ecfdf3;
          color: #067647;
        }

        .status-late {
          background: #fef3f2;
          color: #b42318;
        }

        .status-ontime {
          background: #eff8ff;
          color: #175cd3;
        }

        .status-inside {
          background: #f4f3ff;
          color: #6941c6;
        }

        .status-outside {
          background: #f2f4f7;
          color: #475467;
        }

        .status-absent {
          background: #fef3f2;
          color: #b42318;
        }

        .status-neutral {
          background: #f2f4f7;
          color: #475467;
        }

        .action-button {
          border: 0;
          background: transparent;
          color: #4f46e5;
          font-size: 11px;
          font-weight: 700;
        }

        .action-button:hover {
          text-decoration: underline;
        }

        .employee-card-actions {
          margin-top: 15px;
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 10px;
          flex-wrap: wrap;
        }

        .delete-employee-button {
          border: 1px solid #fecdca;
          background: #fff5f4;
          color: #b42318;
          border-radius: 7px;
          padding: 7px 10px;
          font-size: 11px;
          font-weight: 700;
          transition: 0.2s ease;
        }

        .delete-employee-button:hover {
          background: #fee4e2;
          border-color: #fda29b;
        }

        .delete-employee-button:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .history-summary {
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
          margin-bottom: 18px;
        }

        .history-stat {
          flex: 1;
          min-width: 120px;
          background: #f9fafb;
          border: 1px solid #eaecf0;
          border-radius: 9px;
          padding: 10px 12px;
        }

        .history-stat-label {
          font-size: 9px;
          color: #98a2b3;
          text-transform: uppercase;
          letter-spacing: .5px;
          font-weight: 700;
        }

        .history-stat-value {
          margin-top: 4px;
          font-size: 14px;
          color: #101828;
          font-weight: 750;
        }

        .history-date-group {
          margin-bottom: 20px;
        }

        .history-date-heading {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 9px 11px;
          background: #f9fafb;
          border: 1px solid #eaecf0;
          border-radius: 8px;
          margin-bottom: 4px;
        }

        .history-date-title {
          color: #344054;
          font-size: 11px;
          font-weight: 750;
        }

        .history-date-count {
          color: #98a2b3;
          font-size: 10px;
          font-weight: 600;
        }

        .history-event-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 12px 10px;
          border-bottom: 1px solid #f2f4f7;
        }

        .history-event-row:last-child {
          border-bottom: 0;
        }

        .history-event-left {
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .history-event-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }

        .history-event-dot.in {
          background: #12b76a;
        }

        .history-event-dot.out {
          background: #f04438;
        }

        .history-event-time {
          font-size: 13px;
          color: #101828;
          font-weight: 700;
        }

        .history-event-label {
          font-size: 10px;
          color: #98a2b3;
          margin-top: 2px;
        }

        .history-event-badge {
          min-width: 42px;
          text-align: center;
          padding: 5px 8px;
          border-radius: 999px;
          font-size: 9px;
          font-weight: 750;
        }

        .history-event-badge.in {
          color: #067647;
          background: #ecfdf3;
        }

        .history-event-badge.out {
          color: #b42318;
          background: #fef3f2;
        }

        .empty-state {
          text-align: center;
          padding: 50px 20px;
          color: #98a2b3;
        }

        .empty-icon {
          font-size: 30px;
          margin-bottom: 10px;
        }

        .message {
          padding: 12px 14px;
          border-radius: 8px;
          font-size: 12px;
          margin-bottom: 16px;
        }

        .error-message {
          background: #fef3f2;
          color: #b42318;
          border: 1px solid #fecdca;
        }

        .success-message {
          background: #ecfdf3;
          color: #067647;
          border: 1px solid #abefc6;
        }

        .page-panel {
          background: white;
          border: 1px solid #eaecf0;
          border-radius: 12px;
          padding: 22px;
        }

        .employee-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 14px;
        }

        .employee-card {
          border: 1px solid #eaecf0;
          border-radius: 10px;
          padding: 16px;
          background: white;
        }

        .employee-card-top {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .employee-card-name {
          font-size: 14px;
          font-weight: 700;
          color: #101828;
        }

        .employee-card-id {
          font-size: 11px;
          color: #98a2b3;
          margin-top: 3px;
        }

        .employee-info {
          margin-top: 16px;
          display: grid;
          gap: 8px;
        }

        .info-row {
          display: flex;
          justify-content: space-between;
          font-size: 11px;
        }

        .info-row span:first-child {
          color: #98a2b3;
        }

        .info-row span:last-child {
          color: #344054;
          font-weight: 600;
        }

        .modal-backdrop {
          position: fixed;
          inset: 0;
          background: rgba(16,24,40,.55);
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 20px;
          z-index: 100;
        }

        .modal {
          background: white;
          width: min(940px, 100%);
          max-height: 85vh;
          overflow-y: auto;
          border-radius: 14px;
          box-shadow: 0 20px 60px rgba(16,24,40,.2);
        }

        .modal-header {
          padding: 20px;
          border-bottom: 1px solid #eaecf0;
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
        }

        .modal-title {
          margin: 0;
          font-size: 17px;
          color: #101828;
        }

        .modal-subtitle {
          margin-top: 4px;
          color: #667085;
          font-size: 11px;
        }

        .close-button {
          border: 0;
          background: #f2f4f7;
          width: 30px;
          height: 30px;
          border-radius: 7px;
          color: #667085;
        }

        .modal-body {
          padding: 20px;
        }

        .event-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 13px 0;
          border-bottom: 1px solid #f2f4f7;
        }

        .event-row:last-child {
          border-bottom: 0;
        }

        .event-date {
          font-size: 11px;
          color: #667085;
        }

        .event-time {
          font-size: 13px;
          font-weight: 700;
          color: #101828;
          margin-top: 3px;
        }

        .event-in {
          color: #067647;
          background: #ecfdf3;
        }

        .event-out {
          color: #b42318;
          background: #fef3f2;
        }

        .event-type {
          padding: 5px 9px;
          border-radius: 999px;
          font-size: 9px;
          font-weight: 750;
        }

        .form-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 14px;
        }

        .form-group {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }

        .form-group.full {
          grid-column: 1 / -1;
        }

        .form-group label {
          color: #344054;
          font-size: 11px;
          font-weight: 650;
        }

        .form-group input {
          border: 1px solid #d0d5dd;
          border-radius: 8px;
          padding: 10px;
          outline: none;
          font-size: 12px;
        }

        .form-group input:focus {
          border-color: #7f56d9;
          box-shadow: 0 0 0 3px #f4f3ff;
        }

        .modal-footer {
          display: flex;
          justify-content: flex-end;
          gap: 8px;
          padding: 16px 20px;
          border-top: 1px solid #eaecf0;
        }

        .secondary-button {
          border: 1px solid #d0d5dd;
          background: white;
          color: #344054;
          border-radius: 8px;
          padding: 10px 14px;
          font-size: 12px;
          font-weight: 650;
        }

        .enrollment-section {
          margin-top: 20px;
          padding-top: 20px;
          border-top: 1px solid #eaecf0;
        }

        .enrollment-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 12px;
          margin-bottom: 12px;
        }

        .enrollment-title {
          font-size: 13px;
          font-weight: 750;
          color: #101828;
        }

        .enrollment-caption {
          margin-top: 4px;
          color: #667085;
          font-size: 11px;
          line-height: 1.5;
        }

        .capture-layout {
          display: grid;
          grid-template-columns: 148px 1fr 172px;
          gap: 16px;
          align-items: start;
        }

        .camera-preview {
          width: 100%;
          background: #101828;
          border-radius: 10px;
          overflow: hidden;
          position: relative;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .camera-preview video,
        .camera-preview img {
          width: 100%;
          height: 100%;
          object-fit: cover;
          display: block;
          transform: scaleX(-1);
        }

        .enrollment-camera-select {
          display: flex;
          align-items: center;
          justify-content: flex-end;
          gap: 10px;
          margin: 0 0 12px;
          color: #475467;
          font-size: 11px;
          font-weight: 650;
        }

        .enrollment-camera-select select {
          max-width: 260px;
          padding: 8px 10px;
          border: 1px solid #d0d5dd;
          border-radius: 8px;
          background: white;
          color: #344054;
          font-size: 11px;
        }

        .camera-placeholder {
          color: #98a2b3;
          font-size: 12px;
          text-align: center;
          padding: 20px;
        }

        .capture-controls {
          display: flex;
          gap: 8px;
          margin-top: 10px;
          flex-wrap: wrap;
        }

        .capture-button {
          border: 0;
          background: #4f46e5;
          color: white;
          border-radius: 8px;
          padding: 9px 13px;
          font-size: 11px;
          font-weight: 700;
        }

        .capture-button:hover {
          background: #4338ca;
        }

        .capture-button:disabled {
          opacity: .5;
          cursor: not-allowed;
        }

        .camera-start-button {
          border: 1px solid #d0d5dd;
          background: white;
          color: #344054;
          border-radius: 8px;
          padding: 9px 13px;
          font-size: 11px;
          font-weight: 700;
        }


        /* ---------- Guided face enrollment ---------- */
        .pose-guide {
          position: absolute;
          left: 50%;
          top: 50%;
          height: 94%;
          aspect-ratio: 3 / 4;
          transform: translate(-50%, -50%);
          border: 3px dashed #d0d5dd;
          border-radius: 50%;
          pointer-events: none;
          transition: border-color .15s ease, box-shadow .15s ease;
        }

        .pose-guide.pose-idle {
          border-color: rgba(208, 213, 221, .9);
        }

        .pose-guide.pose-bad {
          border-color: #f04438;
          box-shadow: 0 0 0 2px rgba(240, 68, 56, .25);
        }

        .pose-guide.pose-ok {
          border-color: #12b76a;
          box-shadow: 0 0 0 2px rgba(18, 183, 106, .3);
        }

        .pose-guide.pose-done {
          border-style: solid;
          border-color: #12b76a;
          box-shadow: 0 0 0 4px rgba(18, 183, 106, .4);
        }



        .pose-info {
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
          gap: 6px;
          padding: 12px 10px;
          border: 1.5px solid #e4e7ec;
          border-radius: 10px;
          background: #f9fafb;
          align-self: stretch;
          transition: border-color .15s ease, background .15s ease;
        }

        .pose-info-bad { border-color: #f04438; background: #fef3f2; }
        .pose-info-ok,
        .pose-info-done { border-color: #12b76a; background: #ecfdf3; }

        .pose-info-label {
          font-size: 10px;
          font-weight: 800;
          letter-spacing: .06em;
          text-transform: uppercase;
          color: #667085;
        }

        .pose-info-arrow {
          font-size: 40px;
          line-height: 1;
          font-weight: 700;
          color: #4f46e5;
        }

        .pose-info-name {
          font-size: 14px;
          font-weight: 800;
          color: #101828;
        }

        .pose-info-message {
          margin-top: auto;
          width: 100%;
          font-size: 11px;
          font-weight: 650;
          line-height: 1.35;
          padding: 8px;
          border-radius: 8px;
          color: #344054;
          background: #eaecf0;
        }

        .pose-info-message.pose-message-bad {
          color: white;
          background: #d92d20;
        }

        .pose-info-message.pose-message-ok,
        .pose-info-message.pose-message-done {
          color: white;
          background: #039855;
        }

        .pose-debug {
          margin-top: 6px;
          font-size: 10px;
          color: #667085;
          font-family: monospace;
        }

        .pose-debug-toggle {
          display: inline-flex;
          align-items: center;
          gap: 5px;
          font-size: 11px;
          color: #667085;
        }

        .pose-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 6px;
        }

        .pose-cell {
          position: relative;
          aspect-ratio: 1 / 1;
          border: 1.5px dashed #d0d5dd;
          border-radius: 8px;
          background: #f9fafb;
          padding: 0;
          overflow: hidden;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: default;
        }

        .pose-cell.current {
          border-color: #4f46e5;
          background: #eef2ff;
        }

        .pose-cell.done {
          border: 1.5px solid #12b76a;
          cursor: pointer;
        }

        .pose-cell.skipped {
          cursor: pointer;
          opacity: .6;
        }

        .pose-cell img {
          width: 100%;
          height: 100%;
          object-fit: cover;
          transform: scaleX(-1);
        }

        .pose-cell-arrow {
          font-size: 22px;
          color: #98a2b3;
        }

        .pose-cell.current .pose-cell-arrow {
          color: #4f46e5;
        }

        .pose-cell-label {
          position: absolute;
          left: 0;
          right: 0;
          bottom: 0;
          font-size: 8px;
          font-weight: 700;
          text-align: center;
          padding: 2px 0;
          background: rgba(255, 255, 255, .85);
          color: #344054;
        }

        .capture-count {
          margin-top: 9px;
          font-size: 11px;
          color: #667085;
          font-weight: 650;
        }

        .photo-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 8px;
        }

        .photo-card {
          position: relative;
          border: 1px solid #eaecf0;
          border-radius: 8px;
          overflow: hidden;
          background: #f9fafb;
        }

        .photo-card img {
          width: 100%;
          aspect-ratio: 1 / 1;
          object-fit: cover;
          display: block;
        }

        .remove-photo {
          position: absolute;
          right: 5px;
          top: 5px;
          width: 24px;
          height: 24px;
          border: 0;
          border-radius: 50%;
          background: rgba(16, 24, 40, .78);
          color: white;
          font-size: 14px;
          line-height: 24px;
          padding: 0;
        }

        .photo-number {
          position: absolute;
          left: 5px;
          bottom: 5px;
          background: rgba(16, 24, 40, .75);
          color: white;
          border-radius: 5px;
          padding: 3px 6px;
          font-size: 9px;
          font-weight: 700;
        }

        .capture-help {
          border: 1px solid #eaecf0;
          background: #f9fafb;
          border-radius: 9px;
          padding: 12px;
          color: #667085;
          font-size: 10px;
          line-height: 1.6;
        }

        .capture-help strong {
          color: #344054;
        }

        @media (max-width: 900px) {
          .capture-layout {
            grid-template-columns: 1fr;
          }

          .pose-info {
            flex-direction: row;
            flex-wrap: wrap;
            justify-content: center;
            align-items: center;
          }

          .pose-info-arrow { font-size: 26px; }
          .pose-info-message { margin-top: 0; flex: 1 1 100%; }
        }

        @media (max-width: 700px) {
          .capture-layout {
            grid-template-columns: 1fr;
          }
        }

        @media (max-width: 1100px) {
          .stats-grid {
            grid-template-columns: repeat(2, 1fr);
          }

          .employee-grid {
            grid-template-columns: repeat(2, 1fr);
          }
        }

        @media (max-width: 800px) {
          .sidebar {
            width: 70px;
            padding: 20px 10px;
          }

          .brand {
            justify-content: center;
            padding-left: 0;
            padding-right: 0;
          }

          .brand-text,
          .nav-title,
          .nav-label,
          .sidebar-footer {
            display: none;
          }

          .nav-item {
            justify-content: center;
          }

          .main {
            width: calc(100% - 70px);
            margin-left: 70px;
          }

          .content {
            padding: 20px;
          }

          .topbar {
            padding: 0 20px;
          }

          .camera-grid {
            grid-template-columns: 1fr;
          }
        }

        @media (max-width: 600px) {
          .stats-grid,
          .employee-grid {
            grid-template-columns: 1fr;
          }

          .page-heading {
            flex-direction: column;
            gap: 15px;
          }

          .form-grid {
            grid-template-columns: 1fr;
          }

          .form-group.full {
            grid-column: auto;
          }

          .system-status {
            display: none;
          }
        }
      `}</style>

      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">F</div>

          <div className="brand-text">
            <div className="brand-title">FaceAttend</div>
            <div className="brand-subtitle">Attendance System</div>
          </div>
        </div>

        <div className="nav-title">Workspace</div>

        <button
          className={`nav-item ${
            activePage === "dashboard" ? "active" : ""
          }`}
          onClick={() => setActivePage("dashboard")}
        >
          <span className="nav-icon">⌂</span>
          <span className="nav-label">Dashboard</span>
        </button>

        <button
          className={`nav-item ${
            activePage === "employees" ? "active" : ""
          }`}
          onClick={() => setActivePage("employees")}
        >
          <span className="nav-icon">◉</span>
          <span className="nav-label">Employees</span>
        </button>

        <button
          className={`nav-item ${
            activePage === "attendance" ? "active" : ""
          }`}
          onClick={() => setActivePage("attendance")}
        >
          <span className="nav-icon">▣</span>
          <span className="nav-label">Attendance</span>
        </button>

        <div className="sidebar-footer">
          Face recognition attendance
          <br />
          Multi-camera architecture
          <br />
          <span>v1.0</span>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace /{" "}
            <strong style={{ color: "#344054" }}>
              {activePage === "dashboard"
                ? "Dashboard"
                : activePage === "employees"
                ? "Employees"
                : "Attendance"}
            </strong>
          </div>

          <div className="topbar-right">
            <div className="system-status">
              <span className="online-dot"></span>
              System Online
            </div>

            <button
              className="refresh-button"
              onClick={() => {
                if (activePage === "attendance") {
                  loadAttendanceByDate(selectedDate);
                  loadCameraStatus();
                } else {
                  loadDashboardData();
                  loadCameraStatus();
                }
              }}
            >
              ↻ Refresh
            </button>
          </div>
        </header>

        <div className="content">
          {activePage === "dashboard" && (
            <>
              <div className="page-heading">
                <div>
                  <h1 className="page-title">Attendance Dashboard</h1>
                  <p className="page-description">{todayText}</p>
                </div>

                <button
                  className="primary-button"
                  onClick={() => setShowAddEmployee(true)}
                >
                  + Add Employee
                </button>
              </div>

              {error && (
                <div className="message error-message">
                  {error}
                </div>
              )}

              <div className="stats-grid">
                <div className="stat-card">
                  <div>
                    <div className="stat-label">Total Employees</div>
                    <div className="stat-value">{totalEmployees}</div>
                  </div>
                  <div className="stat-icon">◉</div>
                </div>

                <div className="stat-card">
                  <div>
                    <div className="stat-label">Present Today</div>
                    <div className="stat-value">{presentToday}</div>
                  </div>
                  <div className="stat-icon">✓</div>
                </div>

                <div className="stat-card">
                  <div>
                    <div className="stat-label">Late Today</div>
                    <div className="stat-value">{lateToday}</div>
                  </div>
                  <div className="stat-icon">!</div>
                </div>

                <div className="stat-card">
                  <div>
                    <div className="stat-label">Currently Inside</div>
                    <div className="stat-value">{insideNow}</div>
                  </div>
                  <div className="stat-icon">↗</div>
                </div>
              </div>

              <section className="section">
                <div className="section-header">
                  <div>
                    <div className="section-title">Camera Monitoring</div>
                    <div className="section-caption">
                      Configure webcams and IP cameras for entry, exit, and enrollment
                    </div>
                  </div>
                </div>

                {cameraError && (
                  <div className="message error-message">
                    {cameraError}
                  </div>
                )}

                <div className="camera-management-toolbar">
                  <div>
                    <div className="camera-config-title">Configured cameras</div>
                    <p className="camera-config-help">
                      Add local webcams or IP/CC cameras. Assign one camera to multiple roles, and assign any number of cameras to Entry or Exit.
                    </p>
                  </div>
                  <button className="primary-button" type="button" onClick={() => { resetCameraDraft(); setShowCameraEditor(true); }}>
                    + Add Camera
                  </button>
                </div>

                {cameraActionError && <div className="message error-message" role="alert">{cameraActionError}</div>}
                {cameraActionMessage && <div className="message success-message" role="status">{cameraActionMessage}</div>}
                {cameraLoading ? (
                  <div className="camera-empty">Loading cameras…</div>
                ) : cameras.length ? (
                  <div className="camera-grid">
                    {cameras.map((camera) => (
                      <CameraCard
                        key={camera.id}
                        camera={camera}
                        onView={() => openCameraPreview(camera)}
                        onEdit={() => editCamera(camera)}
                        onDelete={() => deleteCamera(camera)}
                        onToggle={() => toggleCameraEnabled(camera)}
                        onTest={() => testCamera(camera)}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="camera-empty">No cameras configured yet. Add a webcam or IP camera to get started.</div>
                )}
              </section>

              <AttendanceTable
                employees={filteredEmployees}
                getAttendance={getAttendance}
                getStatus={getStatus}
                formatStatus={formatStatus}
                formatTime={formatTime}
                formatWorkedTime={formatWorkedTime}
                statusClass={statusClass}
                search={search}
                setSearch={setSearch}
                statusFilter={statusFilter}
                setStatusFilter={setStatusFilter}
                openEmployeeHistory={openEmployeeHistory}
                loading={loading}
              />
            </>
          )}

          {activePage === "employees" && (
            <>
              <div className="page-heading">
                <div>
                  <h1 className="page-title">Employees</h1>
                  <p className="page-description">
                    Manage employees registered in the attendance system.
                  </p>
                </div>

                <button
                  className="primary-button"
                  onClick={() => setShowAddEmployee(true)}
                >
                  + Add Employee
                </button>
              </div>

              <div className="page-panel">
                <div className="employee-grid">
                  {employees.map((employee) => {
                    const record = getAttendance(employee.employee_id);
                    const status = getStatus(record);

                    return (
                      <div
                        className="employee-card"
                        key={employee.employee_id}
                      >
                        <div className="employee-card-top">
                          <div className="employee-cell">
                            <div className="avatar">
                              {String(employee.employee_name || "E")
                                .charAt(0)
                                .toUpperCase()}
                            </div>

                            <div>
                              <div className="employee-card-name">
                                {employee.employee_name}
                              </div>

                              <div className="employee-card-id">
                                ID: {employee.employee_id}
                              </div>
                            </div>
                          </div>

                          <span
                            className={`status-badge ${statusClass(status)}`}
                          >
                            {formatStatus(status)}
                          </span>
                        </div>

                        <div className="employee-info">
                          <div className="info-row">
                            <span>Shift</span>
                            <span>
                              {employee.shift_start || "--"} -{" "}
                              {employee.shift_end || "--"}
                            </span>
                          </div>

                          <div className="info-row">
                            <span>Email</span>
                            <span>{employee.email || "--"}</span>
                          </div>

                          <div className="info-row">
                            <span>First In</span>
                            <span>
                              {formatTime(record?.first_in)}
                            </span>
                          </div>

                          <div className="info-row">
                            <span>Worked</span>
                            <span>{formatWorkedTime(record)}</span>
                          </div>
                        </div>

                        <div className="employee-card-actions">
                          <button
                            className="action-button"
                            onClick={() => openEditShift(employee)}
                          >
                            Edit Shift
                          </button>

                          <button
                            className="action-button"
                            onClick={() =>
                              openEmployeeHistory(employee)
                            }
                          >
                            View attendance history →
                          </button>

                          <button
                            className="action-button"
                            onClick={() => openReportChooser(employee)}
                          >
                            Export report
                          </button>

                          <button
                            className="delete-employee-button"
                            onClick={() =>
                              deleteEmployee(employee)
                            }
                            disabled={
                              deletingEmployeeId ===
                              String(employee.employee_id)
                            }
                          >
                            {deletingEmployeeId ===
                            String(employee.employee_id)
                              ? "Deleting..."
                              : "🗑 Delete"}
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {!employees.length && (
                  <div className="empty-state">
                    <div className="empty-icon">◉</div>
                    No employees registered.
                  </div>
                )}
              </div>
            </>
          )}

          {activePage === "attendance" && (
            <>
              <div className="page-heading">
                <div>
                  <h1 className="page-title">Attendance</h1>
                  <p className="page-description">
                    View employee attendance and working-time summary by date.
                  </p>
                </div>

                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                  }}
                >
                  <label
                    style={{
                      fontSize: 12,
                      fontWeight: 650,
                      color: "#344054",
                    }}
                  >
                    Select Date
                  </label>

                  <input
                    type="date"
                    value={selectedDate}
                    onChange={handleAttendanceDateChange}
                    style={{
                      border: "1px solid #d0d5dd",
                      borderRadius: 8,
                      padding: "9px 11px",
                      color: "#344054",
                      outline: "none",
                      background: "white",
                      fontSize: 12,
                    }}
                  />

                  <button
                    type="button"
                    className="primary-button"
                    onClick={() => openReportChooser()}
                  >
                    Export attendance
                  </button>
                </div>
              </div>

              {dateAttendanceError && (
                <div className="message error-message">
                  {dateAttendanceError}
                </div>
              )}

              <div
                style={{
                  marginBottom: 16,
                  padding: "12px 14px",
                  background: "#f9fafb",
                  border: "1px solid #eaecf0",
                  borderRadius: 8,
                  color: "#667085",
                  fontSize: 12,
                }}
              >
                Showing attendance for{" "}
                <strong style={{ color: "#344054" }}>
                  {selectedDateText}
                </strong>
              </div>

              <AttendanceTable
                employees={filteredEmployees}
                getAttendance={getAttendance}
                getStatus={getStatus}
                formatStatus={formatStatus}
                formatTime={formatTime}
                formatWorkedTime={formatWorkedTime}
                statusClass={statusClass}
                search={search}
                setSearch={setSearch}
                statusFilter={statusFilter}
                setStatusFilter={setStatusFilter}
                openEmployeeHistory={openEmployeeHistory}
                loading={dateAttendanceLoading}
                tableTitle={`Attendance — ${selectedDate}`}
                tableCaption="Historical event-based attendance and working-time summary"
              />
            </>
          )}
        </div>
      </main>

      {showCameraEditor && (
        <div className="modal-backdrop" onClick={(event) => event.target === event.currentTarget && setShowCameraEditor(false)}>
          <form className="camera-editor-modal" onSubmit={saveCamera}>
            <div className="modal-header">
              <div>
                <h2>{cameraDraft.id ? "Edit camera" : "Add camera"}</h2>
                <p>Configure a local webcam or network camera.</p>
              </div>
              <button type="button" className="modal-close" onClick={() => setShowCameraEditor(false)}>×</button>
            </div>

            <label className="camera-form-label">Camera name
              <input name="name" value={cameraDraft.name} onChange={handleCameraDraftChange} placeholder="Front entrance" required />
            </label>

            <label className="camera-form-label">Camera type
              <select name="source_type" value={cameraDraft.source_type} onChange={(event) => setCameraDraft((previous) => ({ ...previous, source_type: event.target.value, source: event.target.value === "webcam" ? "0" : "" }))}>
                <option value="webcam">Webcam / local device</option>
                <option value="ip">IP / CC camera</option>
              </select>
            </label>

            <label className="camera-form-label">{cameraDraft.source_type === "webcam" ? "Device index or device path" : "Stream URL"}
              <input
                name="source"
                value={cameraDraft.source}
                onChange={handleCameraDraftChange}
                placeholder={cameraDraft.source_type === "webcam" ? "0 or /dev/video0" : "rtsp://camera-ip:554/stream"}
                required={!cameraDraft.id}
                autoComplete="off"
                spellCheck="false"
              />
              {cameraDraft.id && <span className="camera-form-hint">Leave blank to keep the currently saved source.</span>}
            </label>

            <fieldset className="camera-role-fieldset">
              <legend>Assign camera to</legend>
              {["ENTRY", "EXIT", "ENROLLMENT"].map((role) => (
                <label className="camera-role-option" key={role}>
                  <input type="checkbox" checked={cameraDraft.roles.includes(role)} onChange={() => toggleCameraRole(role)} />
                  {role === "ENTRY" ? "Entry" : role === "EXIT" ? "Exit" : "Enrollment"}
                </label>
              ))}
            </fieldset>

            <label className="camera-role-option camera-enabled-option">
              <input type="checkbox" name="enabled" checked={cameraDraft.enabled} onChange={handleCameraDraftChange} />
              Camera enabled
            </label>

            {cameraActionError && <div className="message error-message" role="alert">{cameraActionError}</div>}
            <div className="camera-editor-actions">
              <button type="button" className="camera-action-button" onClick={() => setShowCameraEditor(false)}>Cancel</button>
              <button type="submit" className="primary-button" disabled={cameraBusy}>{cameraBusy ? "Saving…" : cameraDraft.id ? "Save changes" : "Add camera"}</button>
            </div>
          </form>
        </div>
      )}

      {viewCamera && (
        <div className="modal-backdrop" onClick={(event) => event.target === event.currentTarget && closeCameraPreview()}>
          <div className="camera-view-modal">
            <div className="modal-header">
              <div>
                <h2>{viewCamera.name}</h2>
                <p>{viewCamera.source_type === "webcam" ? "Webcam" : "IP / CC camera"} · Live preview</p>
              </div>
              <button type="button" className="modal-close" onClick={closeCameraPreview}>×</button>
            </div>
            <img src={`${API_URL}/cameras/${viewCamera.id}/stream.mjpg?stream_id=${encodeURIComponent(viewCamera.streamId)}`} alt={`Live preview of ${viewCamera.name}`} />
          </div>
        </div>
      )}

      {showAddEmployee && (
        <div
          className="modal-backdrop"
          onClick={(event) => {
            if (
              event.target === event.currentTarget &&
              !enrollmentLoading
            ) {
              closeAddEmployee();
            }
          }}
        >
          <div className="modal">
            <div className="modal-header">
              <div>
                <h2 className="modal-title">Add Employee</h2>
                <div className="modal-subtitle">
                  Add employee details and capture face photos for recognition.
                </div>
              </div>

              <button
                className="close-button"
                onClick={closeAddEmployee}
                disabled={enrollmentLoading}
              >
                ×
              </button>
            </div>

            <form onSubmit={addEmployee}>
              <div className="modal-body">
                {formError && (
                  <div className="message error-message">
                    {formError}
                  </div>
                )}

                {formMessage && (
                  <div className="message success-message">
                    {formMessage}
                  </div>
                )}

                <div className="form-grid">
                  <div className="form-group">
                    <label>Employee ID</label>
                    <input
                      name="employee_id"
                      value={employeeForm.employee_id}
                      onChange={handleEmployeeFormChange}
                      placeholder="e.g. 6495"
                      required
                      disabled={enrollmentLoading}
                    />
                  </div>

                  <div className="form-group full">
                    <label>
                      Hubble Login ID <span style={{ fontWeight: 400, color: "#667085" }}>(optional)</span>
                    </label>
                    <div style={{ display: "flex", gap: "8px" }}>
                      <input
                        name="login_id"
                        value={employeeForm.login_id}
                        onChange={handleEmployeeFormChange}
                        onKeyDown={handleHubbleLoginKeyDown}
                        placeholder="e.g. mbhasyam"
                        disabled={enrollmentLoading || hubbleLoading}
                        style={{ flex: 1 }}
                      />
                      <button
                        type="button"
                        onClick={() => fetchHubbleEmployee()}
                        disabled={enrollmentLoading || hubbleLoading || !employeeForm.login_id.trim()}
                        style={{
                          border: "1px solid #c7d2fe",
                          background: "#eef2ff",
                          color: "#4338ca",
                          borderRadius: "8px",
                          padding: "0 12px",
                          fontSize: "11px",
                          fontWeight: 700,
                          whiteSpace: "nowrap",
                          cursor: enrollmentLoading || hubbleLoading || !employeeForm.login_id.trim() ? "not-allowed" : "pointer",
                          opacity: enrollmentLoading || hubbleLoading || !employeeForm.login_id.trim() ? 0.6 : 1,
                        }}
                      >
                        {hubbleLoading ? "Fetching…" : "Fetch"}
                      </button>
                    </div>
                    {hubbleError && (
                      <div style={{ marginTop: "7px", color: "#b42318", fontSize: "11px", lineHeight: 1.4 }} role="alert">
                        {hubbleError}
                      </div>
                    )}
                    {hubbleEmployee && !hubbleError && (
                      <div style={{ marginTop: "7px", color: "#067647", fontSize: "11px" }}>
                        ✓ Mock Hubble details loaded successfully.
                      </div>
                    )}
                  </div>

                  <div className="form-group">
                    <label>Employee Name</label>
                    <input
                      name="employee_name"
                      value={employeeForm.employee_name}
                      onChange={handleEmployeeFormChange}
                      placeholder="Employee name"
                      required={!employeeForm.login_id.trim()}
                      disabled={enrollmentLoading}
                    />
                  </div>

                  <div className="form-group">
                    <label>Shift Start</label>
                    <input
                      type="time"
                      name="shift_start"
                      value={employeeForm.shift_start}
                      onChange={handleEmployeeFormChange}
                      required
                      disabled={enrollmentLoading}
                    />
                  </div>

                  <div className="form-group">
                    <label>Shift End</label>
                    <input
                      type="time"
                      name="shift_end"
                      value={employeeForm.shift_end}
                      onChange={handleEmployeeFormChange}
                      required
                      disabled={enrollmentLoading}
                    />
                  </div>

                  <div className="form-group full">
                    <label>Email Address</label>
                    <input
                      type="email"
                      name="email"
                      value={employeeForm.email}
                      onChange={handleEmployeeFormChange}
                      placeholder="employee@company.com"
                      required={!employeeForm.login_id.trim()}
                      disabled={enrollmentLoading}
                    />
                  </div>
                </div>

                {employeeForm.login_id && (
                  <div
                    style={{
                      marginTop: "12px",
                      padding: "10px 12px",
                      borderRadius: "8px",
                      background: hubbleEmployee ? "#ecfdf3" : "#f4f3ff",
                      border: hubbleEmployee ? "1px solid #abefc6" : "1px solid #e9d7fe",
                      color: hubbleEmployee ? "#067647" : "#5925dc",
                      fontSize: "11px",
                      lineHeight: 1.5,
                    }}
                  >
                    {hubbleEmployee ? (
                      <>
                        <div style={{ fontWeight: 700, marginBottom: "6px" }}>Mock Hubble Employee Details</div>
                        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "5px 14px", color: "#344054" }}>
                          <div><strong>Employee ID:</strong> {hubbleEmployee.employee_id || "--"}</div>
                          <div><strong>Login ID:</strong> {hubbleEmployee.login_id || "--"}</div>
                          <div><strong>Reporting Manager:</strong> {hubbleEmployee.reporting_manager || "--"}</div>
                          <div><strong>HR Contact:</strong> {hubbleEmployee.hr_contact || "--"}</div>
                          <div><strong>Department:</strong> {hubbleEmployee.department || "--"}</div>
                          <div><strong>Designation:</strong> {hubbleEmployee.designation || "--"}</div>
                          <div><strong>Team:</strong> {hubbleEmployee.team || "--"}</div>
                        </div>
                      </>
                    ) : (
                      <>Enter a Login ID and press <strong>Enter</strong> or click <strong>Fetch</strong> to load mock employee details. Sync to MongoDB occurs when you save this employee.</>
                    )}
                  </div>
                )}

                <div className="enrollment-section">
                  <div className="enrollment-header">
                    <div>
                      <div className="enrollment-title">
                        Face Enrollment
                      </div>
                      <div className="enrollment-caption">
                        Follow each angle on the guide. The photo is taken
                        automatically when the guide turns green. Keep only
                        the employee in the frame.
                      </div>
                    </div>

                    <div className="capture-count">
                      {capturedPhotos.length} / {POSE_STEPS.length} angles
                    </div>
                  </div>

                  <label className="enrollment-camera-select">
                    Enrollment camera
                    <select
                      value={enrollmentCameraId}
                      onChange={(event) => {
                        stopEnrollmentCamera();
                        setEnrollmentCameraId(event.target.value);
                      }}
                      disabled={Boolean(cameraStream) || enrollmentLoading}
                    >
                      <option value="">Use a browser webcam (default)</option>
                      {enrollmentCameras.map((camera) => (
                        <option key={camera.id} value={camera.id}>
                          {camera.name} · {camera.source_type === "webcam" ? "Webcam" : "IP / CC"}
                        </option>
                      ))}
                    </select>
                  </label>

                  {captureError && (
                    <div className="message error-message">
                      {captureError}
                    </div>
                  )}

                  <div className="capture-layout">
                    <div className={`pose-info pose-info-${cameraStream && currentStep ? poseState.status : "idle"}`}>
                      {!cameraStream ? (
                        <>
                          <div className="pose-info-label">Ready</div>
                          <div className="pose-info-arrow">●</div>
                          <div className="pose-info-name">9 angles</div>
                          <div className="pose-info-message">
                            Click “Start Camera”. Then follow each angle.
                            The photo is taken automatically.
                          </div>
                        </>
                      ) : currentStep ? (
                        <>
                          <div className="pose-info-label">
                            Step{" "}
                            {POSE_STEPS.findIndex(
                              (step) => step.id === currentStep.id
                            ) + 1}{" "}
                            / {POSE_STEPS.length}
                          </div>

                          <div className="pose-info-arrow">
                            {currentStep.arrow}
                          </div>

                          <div className="pose-info-name">
                            {currentStep.label}
                          </div>

                          <div
                            className={`pose-info-message pose-message-${poseState.status}`}
                          >
                            {poseState.message}
                          </div>
                        </>
                      ) : (
                        <>
                          <div className="pose-info-label">Done</div>
                          <div className="pose-info-arrow">✓</div>
                          <div className="pose-info-name">All angles</div>
                          <div className="pose-info-message pose-message-done">
                            Click “Save Employee &amp; Enroll Face”.
                          </div>
                        </>
                      )}
                    </div>

                    <div>
                      <div
                        className="camera-preview"
                        style={{ aspectRatio: videoAspect }}
                      >
                        {cameraStream ? (
                          <>
                            {cameraStream.cameraId ? (
                              <img
                                ref={cameraImageRef}
                                crossOrigin="anonymous"
                                alt="Selected enrollment camera preview"
                                onLoad={(event) => {
                                  const { naturalWidth, naturalHeight } = event.currentTarget;
                                  if (naturalWidth && naturalHeight) setVideoAspect(naturalWidth / naturalHeight);
                                }}
                                onError={() => setCaptureError("Unable to display this camera stream. Check its connection and try again.")}
                              />
                            ) : (
                              <video
                                ref={videoRef}
                                autoPlay
                                muted
                                playsInline
                                onLoadedMetadata={(event) => {
                                  const { videoWidth, videoHeight } = event.currentTarget;
                                  if (videoWidth && videoHeight) setVideoAspect(videoWidth / videoHeight);
                                }}
                              />
                            )}

                            {currentStep && (
                              <div
                                className={`pose-guide pose-${poseState.status}`}
                              />
                            )}
                          </>
                        ) : (
                          <div className="camera-placeholder">
                            Camera is not started.
                            <br />
                            Click “Start Camera” to begin.
                          </div>
                        )}
                      </div>

                      <canvas
                        ref={canvasRef}
                        style={{ display: "none" }}
                      />

                      {cameraStream && showPoseDebug && poseState.debug && (
                        <div className="pose-debug">
                          yaw {poseState.debug.yaw.toFixed(1)}° · pitch{" "}
                          {poseState.debug.pitch.toFixed(1)}° · cues{" "}
                          {poseState.debug.yawCue.toFixed(2)} /{" "}
                          {poseState.debug.pitchCue.toFixed(2)} · faces{" "}
                          {poseState.debug.faces}
                        </div>
                      )}

                      <div className="capture-controls">
                        {!cameraStream ? (
                          <button
                            type="button"
                            className="camera-start-button"
                            onClick={startEnrollmentCamera}
                            disabled={enrollmentLoading}
                          >
                            Start Camera
                          </button>
                        ) : (
                          <>
                            <button
                              type="button"
                              className="camera-start-button"
                              onClick={skipCurrentStep}
                              disabled={
                                enrollmentLoading ||
                                !currentStep ||
                                currentStep.id === "front"
                              }
                            >
                              Skip this angle
                            </button>

                            <button
                              type="button"
                              className="camera-start-button"
                              onClick={stopEnrollmentCamera}
                              disabled={enrollmentLoading}
                            >
                              Stop Camera
                            </button>

                            <label className="pose-debug-toggle">
                              <input
                                type="checkbox"
                                checked={showPoseDebug}
                                onChange={(event) =>
                                  setShowPoseDebug(event.target.checked)
                                }
                              />
                              Show angles
                            </label>
                          </>
                        )}
                      </div>
                    </div>

                    <div>
                      <div className="pose-grid">
                        {[0, 1, 2].flatMap((row) =>
                          [0, 1, 2].map((col) => {
                            const step = POSE_STEPS.find(
                              (item) => item.row === row && item.col === col
                            );

                            const photoIndex = capturedPhotos.findIndex(
                              (photo) => photo.poseId === step.id
                            );

                            const photo =
                              photoIndex >= 0 ? capturedPhotos[photoIndex] : null;

                            const skipped = skippedSteps.includes(step.id);
                            const isCurrent = currentStep?.id === step.id;

                            return (
                              <button
                                type="button"
                                key={step.id}
                                className={`pose-cell${isCurrent ? " current" : ""}${photo ? " done" : ""}${skipped ? " skipped" : ""}`}
                                disabled={enrollmentLoading || (!photo && !skipped)}
                                title={
                                  photo || skipped
                                    ? `Retake: ${step.label}`
                                    : step.label
                                }
                                onClick={() => {
                                  setSkippedSteps((previous) =>
                                    previous.filter((id) => id !== step.id)
                                  );

                                  if (photoIndex >= 0) {
                                    removeCapturedPhoto(photoIndex);
                                  }
                                }}
                              >
                                {photo ? (
                                  <img
                                    src={photo.previewUrl}
                                    alt={`${step.label} captured`}
                                  />
                                ) : (
                                  <span className="pose-cell-arrow">
                                    {step.arrow}
                                  </span>
                                )}

                                <span className="pose-cell-label">
                                  {photo ? "✓ " : skipped ? "skipped · " : ""}
                                  {step.label}
                                </span>
                              </button>
                            );
                          })
                        )}
                      </div>

                      <div className="capture-help">
                        Tap a captured angle to retake it. At least 5 angles
                        are required.
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="modal-footer">
                <button
                  type="button"
                  className="secondary-button"
                  onClick={closeAddEmployee}
                  disabled={enrollmentLoading}
                >
                  Cancel
                </button>

                <button
                  type="submit"
                  className="primary-button"
                  disabled={
                    enrollmentLoading ||
                    capturedPhotos.length < 5
                  }
                >
                  {enrollmentLoading
                    ? "Enrolling Face..."
                    : "Save Employee & Enroll Face"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {selectedEmployee && (
        <div
          className="modal-backdrop"
          onClick={(event) => {
            if (event.target === event.currentTarget) {
              closeHistory();
            }
          }}
        >
          <div className="modal">
            <div className="modal-header">
              <div>
                <h2 className="modal-title">
                  {selectedEmployee.employee_name}
                </h2>

                <div className="modal-subtitle">
                  Employee ID: {selectedEmployee.employee_id}
                </div>
              </div>

              <button className="close-button" onClick={closeHistory}>
                ×
              </button>
            </div>

            <div className="modal-body">
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, marginBottom: 16 }}>
                <button type="button" className="secondary-button" onClick={() => changeHistoryMonth(-1)} disabled={historyLoading}>← Previous</button>
                <strong style={{ color: "#344054", fontSize: 16 }}>
                  {new Date(historyYear, historyMonth - 1, 1).toLocaleDateString("en", { month: "long", year: "numeric" })}
                </strong>
                <button type="button" className="secondary-button" onClick={() => changeHistoryMonth(1)} disabled={historyLoading}>Next →</button>
              </div>

              {historyError && <div className="message error-message" role="alert">{historyError}</div>}
              {historyLoading ? (
                <div className="empty-state">Loading monthly attendance…</div>
              ) : (
                <>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(7, minmax(0, 1fr))", gap: 6, marginBottom: 6, textAlign: "center", color: "#667085", fontSize: 11, fontWeight: 700 }}>
                    {["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map((weekday) => <div key={weekday}>{weekday}</div>)}
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(7, minmax(0, 1fr))", gap: 6 }}>
                    {Array.from({ length: (new Date(historyYear, historyMonth - 1, 1).getDay() + 6) % 7 }, (_, index) => <div key={`blank-${index}`} />)}
                    {attendanceHistory.map((day) => {
                      const colors = {
                        FULL_DAY: ["#ecfdf3", "#067647", "#abefc6"],
                        HALF_DAY: ["#fffaeb", "#b54708", "#fedf89"],
                        LEAVE: ["#f4f3ff", "#5925dc", "#d9d6fe"],
                        ABSENT: ["#fef3f2", "#b42318", "#fecdca"],
                        WEEKEND: ["#f9fafb", "#667085", "#eaecf0"],
                      }[day.status] || ["#f9fafb", "#344054", "#eaecf0"];
                      const selected = selectedHistoryDay?.date === day.date;
                      return (
                        <button
                          type="button"
                          key={day.date}
                          onClick={() => setSelectedHistoryDay(day)}
                          aria-pressed={selected}
                          style={{ minHeight: 66, padding: "7px 4px", borderRadius: 8, border: `1px solid ${selected ? "#344054" : colors[2]}`, background: colors[0], color: colors[1], cursor: "pointer", outline: selected ? "2px solid #344054" : "none" }}
                        >
                          <div style={{ fontWeight: 700 }}>{day.day}</div>
                          <div style={{ fontSize: 10, marginTop: 4, lineHeight: 1.25 }}>{day.status.replaceAll("_", " ")}</div>
                          {day.worked_hours && day.worked_minutes > 0 && <div style={{ fontSize: 10, marginTop: 2 }}>{day.worked_hours}</div>}
                        </button>
                      );
                    })}
                  </div>

                  <div style={{ display: "flex", flexWrap: "wrap", gap: 12, margin: "14px 0", fontSize: 11, color: "#475467" }}>
                    {[["FULL_DAY", "Full day"], ["HALF_DAY", "Partial day"], ["LEAVE", "Leave"], ["ABSENT", "Absent"], ["WEEKEND", "Weekend"]].map(([status, label]) => (
                      <span key={status}><span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 3, marginRight: 5, background: { FULL_DAY: "#12b76a", HALF_DAY: "#f79009", LEAVE: "#7a5af8", ABSENT: "#f04438", WEEKEND: "#98a2b3" }[status] }} />{label}</span>
                    ))}
                  </div>

                  {selectedHistoryDay && (
                    <section style={{ border: "1px solid #eaecf0", borderRadius: 10, padding: 14, background: "#fcfcfd" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, flexWrap: "wrap" }}>
                        <div>
                          <strong>{new Date(`${selectedHistoryDay.date}T00:00:00`).toLocaleDateString("en", { weekday: "long", day: "numeric", month: "long", year: "numeric" })}</strong>
                          <div style={{ marginTop: 5, color: "#667085", fontSize: 12 }}>
                            {selectedHistoryDay.status.replaceAll("_", " ")} · Worked {selectedHistoryDay.worked_hours || "0h 0m"}
                            {selectedHistoryDay.first_in ? ` · First in ${formatTime(selectedHistoryDay.first_in)}` : ""}
                            {selectedHistoryDay.last_out ? ` · Last out ${formatTime(selectedHistoryDay.last_out)}` : ""}
                          </div>
                          {selectedHistoryDay.status === "LEAVE" && <div style={{ marginTop: 7, color: "#5925dc", fontSize: 12 }}>{selectedHistoryDay.leave_type}{selectedHistoryDay.reason ? ` — ${selectedHistoryDay.reason}` : ""}</div>}
                        </div>
                        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                          {selectedHistoryDay.status === "LEAVE" ? (
                            <button type="button" className="secondary-button" onClick={removeSelectedDayLeave} disabled={leaveSaving}>{leaveSaving ? "Saving…" : "Remove leave"}</button>
                          ) : (
                            <button type="button" className="secondary-button" onClick={saveSelectedDayLeave} disabled={leaveSaving || selectedHistoryDay.status === "WEEKEND"}>{leaveSaving ? "Saving…" : "Mark leave / edit"}</button>
                          )}
                        </div>
                      </div>
                      {selectedHistoryDay.events?.length > 0 && (
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 12 }}>
                          {selectedHistoryDay.events.map((event, index) => (
                            <span key={`${event.id || index}-${event.event_time}`} style={{ borderRadius: 6, padding: "5px 8px", background: event.event_type === "IN" ? "#ecfdf3" : "#eff8ff", color: event.event_type === "IN" ? "#067647" : "#175cd3", fontSize: 11, fontWeight: 650 }}>
                              {event.event_type} · {formatTime(event.event_time)}
                            </span>
                          ))}
                        </div>
                      )}
                    </section>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {editingEmployee && (
        <div className="modal-backdrop" onClick={(event) => event.target === event.currentTarget && !editShiftLoading && setEditingEmployee(null)}>
          <form className="modal" onSubmit={saveEmployeeShift}>
            <div className="modal-header">
              <div>
                <h2 className="modal-title">Edit Shift</h2>
                <div className="modal-subtitle">{editingEmployee.employee_name} · ID: {editingEmployee.employee_id}</div>
              </div>
              <button type="button" className="close-button" onClick={() => setEditingEmployee(null)} disabled={editShiftLoading}>×</button>
            </div>
            <div className="modal-body">
              {editShiftError && <div className="message error-message" role="alert">{editShiftError}</div>}
              <div className="form-grid">
                <div className="form-group">
                  <label>Shift Start</label>
                  <input type="time" required value={editShiftForm.shift_start} onChange={(event) => setEditShiftForm((previous) => ({ ...previous, shift_start: event.target.value }))} disabled={editShiftLoading} />
                </div>
                <div className="form-group">
                  <label>Shift End</label>
                  <input type="time" required value={editShiftForm.shift_end} onChange={(event) => setEditShiftForm((previous) => ({ ...previous, shift_end: event.target.value }))} disabled={editShiftLoading} />
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button type="button" className="secondary-button" onClick={() => setEditingEmployee(null)} disabled={editShiftLoading}>Cancel</button>
              <button type="submit" className="primary-button" disabled={editShiftLoading}>{editShiftLoading ? "Saving…" : "Save Shift"}</button>
            </div>
          </form>
        </div>
      )}

      {reportChooser && (
        <div className="modal-backdrop" onClick={(event) => event.target === event.currentTarget && !reportDownloading && setReportChooser(null)}>
          <div className="modal">
            <div className="modal-header">
              <div>
                <h2 className="modal-title">Export Attendance</h2>
                <div className="modal-subtitle">{reportChooser.employee ? `${reportChooser.employee.employee_name} · ID: ${reportChooser.employee.employee_id}` : "All employees"}</div>
              </div>
              <button type="button" className="close-button" onClick={() => setReportChooser(null)} disabled={reportDownloading}>×</button>
            </div>
            <div className="modal-body">
              {reportError && <div className="message error-message" role="alert">{reportError}</div>}
              <div className="form-group">
                <label>Choose a date in the report period</label>
                <input type="date" value={reportDate} onChange={(event) => setReportDate(event.target.value)} required disabled={reportDownloading} />
              </div>
              <p style={{ color: "#667085", fontSize: 12, lineHeight: 1.5 }}>Weekly reports cover Monday–Sunday. Monthly reports cover the selected calendar month. The XLSX includes daily attendance and event sheets.</p>
            </div>
            <div className="modal-footer" style={{ justifyContent: "space-between" }}>
              <button type="button" className="secondary-button" onClick={() => setReportChooser(null)} disabled={reportDownloading}>Cancel</button>
              <div style={{ display: "flex", gap: 8 }}>
                <button type="button" className="secondary-button" onClick={() => downloadAttendanceReport("weekly")} disabled={reportDownloading || !reportDate}>{reportDownloading ? "Preparing…" : "Download Weekly"}</button>
                <button type="button" className="primary-button" onClick={() => downloadAttendanceReport("monthly")} disabled={reportDownloading || !reportDate}>{reportDownloading ? "Preparing…" : "Download Monthly"}</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function CameraCard({ camera, onView, onEdit, onDelete, onToggle, onTest }) {
  return (
    <article className={`camera-card ${camera.enabled ? "" : "camera-card-disabled"}`}>
      <div className="camera-header">
        <div>
          <div className="camera-name">{camera.name}</div>
          <div className="camera-responsibility">{camera.source_type === "webcam" ? "Webcam" : "IP / CC camera"}</div>
        </div>

        <div className={`connection-badge ${camera.enabled ? "configured" : "disconnected"}`}>
          <span>●</span>{camera.enabled ? "ENABLED" : "DISABLED"}
        </div>
      </div>

      <div className="camera-details">
        <div>
          <div className="detail-label">Source</div>
          <div className="detail-value">{camera.device || "--"}</div>
        </div>
        <div>
          <div className="detail-label">Assigned to</div>
          <div className="detail-value">{camera.roles?.length ? camera.roles.join(" · ") : "Not assigned"}</div>
        </div>
      </div>

      <div className="camera-card-actions">
        <button type="button" className="camera-action-button" onClick={onView} disabled={!camera.enabled}>View</button>
        <button type="button" className="camera-action-button" onClick={onTest} disabled={!camera.enabled}>Test</button>
        <button type="button" className="camera-action-button" onClick={onEdit}>Edit</button>
        <button type="button" className="camera-action-button" onClick={onToggle}>{camera.enabled ? "Disable" : "Enable"}</button>
        <button type="button" className="camera-action-button danger" onClick={onDelete}>Delete</button>
      </div>
    </article>
  );
}

function AttendanceTable({
  employees,
  getAttendance,
  getStatus,
  formatStatus,
  formatTime,
  formatWorkedTime,
  statusClass,
  search,
  setSearch,
  statusFilter,
  setStatusFilter,
  openEmployeeHistory,
  loading,
  tableTitle = "Today's Attendance",
  tableCaption = "Event-based attendance and working-time summary",
}) {
  const filters = [
    ["ALL", "All"],
    ["PRESENT", "Present"],
    ["EARLY", "Early"],
    ["ON_TIME", "On Time"],
    ["LATE", "Late"],
    ["INSIDE", "Inside"],
    ["OUTSIDE", "Outside"],
  ];

  return (
    <section className="section">
      <div className="section-header">
        <div>
          <div className="section-title">{tableTitle}</div>
          <div className="section-caption">
            {tableCaption}
          </div>
        </div>
      </div>

      <div className="table-card">
        <div className="table-toolbar">
          <div className="search-box">
            <span className="search-icon">⌕</span>

            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search employee..."
            />
          </div>

          <div className="filter-group">
            {filters.map(([value, label]) => (
              <button
                key={value}
                className={`filter-button ${
                  statusFilter === value ? "active" : ""
                }`}
                onClick={() => setStatusFilter(value)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Employee</th>
                <th>First In</th>
                <th>Arrival</th>
                <th>Worked</th>
                <th>Sessions</th>
                <th>Last Out</th>
                <th>State</th>
                <th>Action</th>
              </tr>
            </thead>

            <tbody>
              {loading ? (
                <tr>
                  <td colSpan="8">
                    <div className="empty-state">
                      Loading attendance...
                    </div>
                  </td>
                </tr>
              ) : employees.length === 0 ? (
                <tr>
                  <td colSpan="8">
                    <div className="empty-state">
                      <div className="empty-icon">◷</div>
                      No attendance records found.
                    </div>
                  </td>
                </tr>
              ) : (
                employees.map((employee) => {
                  const record = getAttendance(employee.employee_id);
                  const status = getStatus(record);

                  return (
                    <tr key={employee.employee_id}>
                      <td>
                        <div className="employee-cell">
                          <div className="avatar">
                            {String(employee.employee_name || "E")
                              .charAt(0)
                              .toUpperCase()}
                          </div>

                          <div>
                            <div className="employee-name">
                              {employee.employee_name}
                            </div>

                            <div className="employee-id">
                              ID: {employee.employee_id}
                            </div>
                          </div>
                        </div>
                      </td>

                      <td>{formatTime(record?.first_in)}</td>

                      <td>
                        <span
                          className={`status-badge ${statusClass(status)}`}
                        >
                          {formatStatus(status)}
                        </span>
                      </td>

                      <td>
                        <strong>
                          {formatWorkedTime(record)}
                        </strong>
                      </td>

                      <td>
                        {record?.session_count ??
                          record?.sessions?.length ??
                          0}
                      </td>

                      <td>{formatTime(record?.last_out)}</td>

                      <td>
                        <span
                          className={`status-badge ${statusClass(
                            String(
                              record?.current_state || "OUTSIDE"
                            ).toUpperCase()
                          )}`}
                        >
                          {String(
                            record?.current_state || "OUTSIDE"
                          ).toUpperCase()}
                        </span>
                      </td>

                      <td>
                        <button
                          className="action-button"
                          onClick={() =>
                            openEmployeeHistory(employee)
                          }
                        >
                          View
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

export default App;