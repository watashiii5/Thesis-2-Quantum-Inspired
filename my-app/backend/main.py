"""
FastAPI Backend for College Room Allocation System

This API provides endpoints for:
- Generating class schedules with room allocation
- Managing rooms, courses, sections, and teachers
- Viewing and analyzing scheduling results
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union, Deque
from datetime import datetime
from collections import deque
import asyncio
import time
import uuid
import uvicorn
import os
from dotenv import load_dotenv

from models import (
    Room, Course, Section, Teacher, TimeSlot, 
    ScheduleEntry, GenerateScheduleRequest, ScheduleResult,
    RoomType, DayOfWeek
)
from database import (
    get_supabase_client, get_all_rooms, get_sections_for_scheduling, get_all_sections,
    get_all_teachers, get_time_slots, save_schedule_entries, get_available_rooms,
    check_room_conflicts, check_teacher_conflicts, get_room_utilization,
    get_schedule_by_id, delete_schedule, get_all_schedules,
    create_generated_schedule, update_generated_schedule, save_room_allocations,
    get_generated_schedules, get_generated_schedule_by_id, delete_generated_schedule,
    get_room_allocations_by_schedule
)
from scheduler import run_scheduler
# Import enhanced v2 scheduler with 30-min slots and validation
from scheduler_v2 import run_enhanced_scheduler, generate_30min_slots, generate_time_slots, validate_scheduling_data

# Load environment variables
load_dotenv()

# Import RL API
from rl_api import rl_router, initialize_persistence

# Create FastAPI app
app = FastAPI(
    title="College Room Allocation API",
    description="Quantum-Inspired Optimization for Class Room Scheduling",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include RL Router
app.include_router(rl_router)


# ========================
# Schedule Queue (FIFO)
# ========================

class ScheduleQueueTimeoutError(Exception):
    """Raised when a request waits too long in the schedule generation queue."""


schedule_generation_queue: Deque[str] = deque()
schedule_generation_active_job: Optional[str] = None
schedule_generation_condition = asyncio.Condition()
SCHEDULE_QUEUE_MAX_WAIT_SECONDS = max(30, int(os.getenv("SCHEDULE_QUEUE_MAX_WAIT_SECONDS", "600")))
SCHEDULE_QUEUE_MAX_LENGTH = max(0, int(os.getenv("SCHEDULE_QUEUE_MAX_LENGTH", "0") or 0))


def _is_free_tier_resource_profile_enabled() -> bool:
    profile = (os.getenv("SCHEDULER_RESOURCE_PROFILE") or "").strip().lower()
    if profile:
        return profile in {"render-free", "free", "low"}
    # Render sets service env vars; use as a safe heuristic when profile isn't explicitly set.
    return bool(os.getenv("RENDER") or os.getenv("RENDER_SERVICE_ID"))


async def _acquire_schedule_generation_slot(job_id: str) -> Dict[str, Any]:
    """Queue schedule generation requests and grant one active slot at a time."""
    global schedule_generation_active_job

    queued_at = time.monotonic()
    async with schedule_generation_condition:
        if SCHEDULE_QUEUE_MAX_LENGTH > 0:
            total_in_system = len(schedule_generation_queue) + (1 if schedule_generation_active_job else 0)
            if total_in_system >= SCHEDULE_QUEUE_MAX_LENGTH:
                raise ScheduleQueueTimeoutError(
                    f"Queue is full (max_length={SCHEDULE_QUEUE_MAX_LENGTH}). Try again shortly."
                )

        initial_position = len(schedule_generation_queue) + (1 if schedule_generation_active_job else 0) + 1
        schedule_generation_queue.append(job_id)
        deadline = queued_at + SCHEDULE_QUEUE_MAX_WAIT_SECONDS

        try:
            while schedule_generation_queue[0] != job_id or schedule_generation_active_job is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    try:
                        schedule_generation_queue.remove(job_id)
                    except ValueError:
                        pass
                    schedule_generation_condition.notify_all()
                    raise ScheduleQueueTimeoutError(
                        f"Queue wait exceeded {SCHEDULE_QUEUE_MAX_WAIT_SECONDS} seconds"
                    )
                await asyncio.wait_for(schedule_generation_condition.wait(), timeout=remaining)
        except asyncio.CancelledError:
            # If client disconnects while waiting, remove stale queue entry.
            try:
                schedule_generation_queue.remove(job_id)
            except ValueError:
                pass
            schedule_generation_condition.notify_all()
            raise

        schedule_generation_queue.popleft()
        schedule_generation_active_job = job_id

        return {
            "job_id": job_id,
            "initial_position": initial_position,
            "wait_seconds": round(time.monotonic() - queued_at, 3),
            "pending_after_start": len(schedule_generation_queue),
            "max_wait_seconds": SCHEDULE_QUEUE_MAX_WAIT_SECONDS,
        }


async def _release_schedule_generation_slot(job_id: str):
    """Release the active slot and wake the next queued request."""
    global schedule_generation_active_job

    async with schedule_generation_condition:
        if schedule_generation_active_job == job_id:
            schedule_generation_active_job = None
        else:
            # Safety fallback in case the job failed before becoming active.
            try:
                schedule_generation_queue.remove(job_id)
            except ValueError:
                pass
        schedule_generation_condition.notify_all()

# Build allowed origins list for CORS
def get_allowed_origins():
    """Get list of allowed origins including Vercel preview URLs"""
    origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    
    # Add production frontend URL from environment
    frontend_url = os.getenv("FRONTEND_URL")
    if frontend_url:
        origins.append(frontend_url)
    
    # Add additional origins from comma-separated list (for multiple Vercel previews)
    additional_origins = os.getenv("ADDITIONAL_ORIGINS", "")
    if additional_origins:
        origins.extend([o.strip() for o in additional_origins.split(",") if o.strip()])
    
    return origins

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_origin_regex=r"https://.*\.vercel\.app",  # Allow all Vercel preview URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================
# Health Check
# ========================

@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "status": "healthy",
        "service": "College Room Allocation API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        client = get_supabase_client()
        # Simple query to verify DB connection (sync client, no await needed)
        await asyncio.to_thread(client.table("rooms").select("id").limit(1).execute)
        db_status = "connected"
    except Exception as e:
        db_status = "error"
    
    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "database": db_status,
        "configured": bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY")),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/schedules/generate/queue-status")
async def get_schedule_generation_queue_status():
    """Get current FIFO queue status for schedule generation requests."""
    async with schedule_generation_condition:
        return {
            "active": schedule_generation_active_job is not None,
            "waiting_count": len(schedule_generation_queue),
            "active_job_id": schedule_generation_active_job,
            "queued_job_ids": list(schedule_generation_queue),
            "max_wait_seconds": SCHEDULE_QUEUE_MAX_WAIT_SECONDS,
            "timestamp": datetime.utcnow().isoformat(),
        }


# ========================
# Room Management
# ========================

class RoomResponse(BaseModel):
    id: int
    room_code: str
    room_name: str
    building: str
    campus: str
    capacity: int
    room_type: str
    floor: Optional[int] = 1
    is_accessible: Optional[bool] = False


@app.get("/api/rooms", response_model=List[RoomResponse])
async def list_rooms(
    campus: Optional[str] = None,
    building: Optional[str] = None,
    room_type: Optional[str] = None,
    min_capacity: Optional[int] = None
):
    """Get all rooms with optional filtering"""
    try:
        # Let database handle filtering for efficiency
        rooms = await get_available_rooms(
            room_type=room_type,
            min_capacity=min_capacity or 0,
            campus=campus
        )
        
        # Client-side filtering only if needed
        if building:
            rooms = [r for r in rooms if r.get("building") == building]
        
        return rooms
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve rooms")


@app.get("/api/rooms/{room_id}")
async def get_room(room_id: int):
    """Get a specific room by ID"""
    try:
        # Fetch all rooms and find by ID (efficient since rooms list is typically small)
        rooms = await get_all_rooms()
        room = next((r for r in rooms if r.get("id") == room_id), None)
        if not room:
            raise HTTPException(status_code=404, detail="Room not found")
        return room
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve room")


@app.get("/api/rooms/{room_id}/utilization")
async def get_room_utilization_stats(room_id: int, schedule_id: Optional[int] = None):
    """Get room utilization statistics"""
    try:
        resolved_schedule_id = schedule_id
        if resolved_schedule_id is None:
            schedules = await get_all_schedules()
            if schedules:
                resolved_schedule_id = schedules[0].get("id")

        if resolved_schedule_id is None:
            raise HTTPException(status_code=404, detail="No schedules found")

        utilization = await get_room_utilization(int(resolved_schedule_id))
        room_stats = next((row for row in utilization if row.get("room_id") == room_id), None)
        if room_stats is None:
            raise HTTPException(status_code=404, detail="Room not found")
        return {"schedule_id": int(resolved_schedule_id), **room_stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve room utilization")


# ========================
# Section Management
# ========================

@app.get("/api/sections")
async def list_sections(
    department: Optional[str] = None,
    course_code: Optional[str] = None
):
    """Get all sections with optional filtering"""
    try:
        # Database-level filtering for department
        sections = await get_all_sections(department)
        
        # Client-side filtering only for course_code (less likely to be indexed)
        if course_code:
            sections = [s for s in sections if s.get("course_code") == course_code]
        
        return sections
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve sections")


# ========================
# Teacher Management
# ========================

@app.get("/api/teachers")
async def list_teachers(department: Optional[str] = None):
    """Get all teachers with optional filtering"""
    try:
        teachers = await get_all_teachers(department)
        
        return teachers
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve teachers")


# ========================
# Time Slots
# ========================

@app.get("/api/time-slots")
async def list_time_slots():
    """Get all available time slots"""
    try:
        slots = await get_time_slots()
        return slots
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve time slots")


# ========================
# Schedule Generation
# ========================

class TimeSlotModel(BaseModel):
    """Time slot model"""
    id: int
    slot_name: str
    start_time: str
    end_time: str
    duration_minutes: int

class SectionDataModel(BaseModel):
    """Section data from frontend - Enhanced v2 with subject support"""
    id: int
    section_code: str
    course_code: str
    course_name: str
    subject_code: Optional[str] = None  # New: Subject code (e.g., IT-311)
    subject_name: Optional[str] = None  # New: Subject name
    teacher_id: Union[int, str]  # Support UUIDs from faculty_profiles
    teacher_name: str
    year_level: int = 1
    student_count: int
    required_room_type: str
    weekly_hours: int
    lec_hours: int = 0  # New: Lecture hours
    lab_hours: int = 0  # New: Lab hours
    requires_lab: bool
    department: str
    college: Optional[str] = None  # New: College name
    semester: Optional[str] = "1st Semester"
    required_features: Optional[List[str]] = None  # NEW: Required equipment tags
    lec_required_features: Optional[List[str]] = None  # NEW: Lecture-specific equipment
    lab_required_features: Optional[List[str]] = None  # NEW: Lab-specific equipment

class RoomDataModel(BaseModel):
    """Room data from frontend - Enhanced with equipment and college assignment"""
    id: int
    room_code: str
    room_name: str = ""  # Default empty string if not provided
    building: str
    campus: str
    capacity: int
    room_type: str
    floor: int = 1  # Default to first floor
    is_accessible: bool = False  # Default to False for PWD accessibility
    has_projector: bool = False
    has_ac: bool = False
    has_computers: int = 0
    has_lab_equipment: bool = False
    feature_tags: Optional[List[str]] = None  # NEW: Equipment tags like "Desktop_PC", "DC_Power_Supply"
    college: Optional[str] = None  # NEW: College assignment (e.g., "CS", "CAFA", "Shared")

class FacultyTypeModel(BaseModel):
    """Rules for different faculty employment types"""
    max_hours_per_week: int
    max_hours_per_day: int
    max_sections_total: int
    max_sections_per_course: int
    required_office_hours: int = 0
    research_required: bool = False

class ScheduleGenerationRequest(BaseModel):
    """Request model for schedule generation - Enhanced v2 with 30-min slots and BulSU QSA"""
    schedule_name: str
    semester: str
    academic_year: str
    campus_group_id: Optional[int] = None  # Campus group ID for database storage
    class_group_id: Optional[int] = None   # Class group ID for database storage  
    section_ids: Optional[List[int]] = None
    room_ids: Optional[List[int]] = None
    time_slots: Optional[List[TimeSlotModel]] = None
    active_days: Optional[List[str]] = None
    sections_data: Optional[List[SectionDataModel]] = None
    rooms_data: Optional[List[RoomDataModel]] = None
    
    # BulSU QSA: Online Day Support
    online_days: Optional[List[str]] = None  # Days designated for online classes (e.g., ['saturday'])
    
    # Time configuration - USE FRONTEND'S SLOT DURATION
    start_time: str = "07:00"
    end_time: str = "20:00"  # Default 8PM closing
    slot_duration: int = 90  # Default to 90 minutes (1.5 hours) - standard academic period
    
    # Enhanced optimization parameters
    max_iterations: int = 5000  # Increased default
    initial_temperature: float = 150.0
    cooling_rate: float = 0.997
    quantum_tunneling_prob: float = 0.12
    max_teacher_hours_per_day: int = 8
    max_consecutive_hours: int = 4
    prioritize_accessibility: bool = True
    avoid_lunch_conflicts: bool = True
    lunch_start: str = "13:00"  # 1:00 PM - UPDATED DEFAULT
    lunch_end: str = "14:00"    # 2:00 PM - UPDATED DEFAULT
    
    # NEW: Constraint settings for BulSU rules
    lunch_mode: str = "strict"  # 'strict', 'flexible', or 'none' - STRICT BY DEFAULT
    lunch_start_hour: int = 13  # 1:00 PM - UPDATED DEFAULT
    lunch_end_hour: int = 14    # 2:00 PM - UPDATED DEFAULT
    strict_lab_room_matching: bool = True  # Lab classes MUST be in lab rooms
    strict_lecture_room_matching: bool = True  # Lectures should NOT be in lab rooms
    
    # Split session settings - allow classes to be split into multiple sessions
    allow_split_sessions: bool = True
    combine_split_lectures: bool = True
    allow_g1_g2_split_sessions: bool = True
    enforce_g1_g2_equal_hours: bool = True # NEW: Ensure split groups have balanced hours
    
    # Faculty Type overrides
    faculty_types: Optional[Dict[str, FacultyTypeModel]] = None
    
    # Use enhanced scheduler
    use_enhanced_scheduler: bool = True

    # Resource tuning (optional). Keep quality-first defaults.
    low_resource_mode: bool = False
    auto_low_resource_mode: bool = False
    cpu_yield_every_iterations: int = 0
    cpu_yield_ms: float = 0.0
    
    # NEW: Fixed/Manual allocations to prioritize
    fixed_allocations: Optional[List[Dict[str, Any]]] = None
    
    # NEW: Soft Constraint Penalties (Fine-tuning)
    SOFT_ROOM_TYPE_MISMATCH: Optional[int] = 50
    SOFT_ROOM_TYPE_MAJOR_MISMATCH: Optional[int] = 500
    SOFT_CAPACITY_WASTE: Optional[int] = 15
    SOFT_LUNCH_OVERLAP: Optional[int] = 500
    SOFT_TEACHER_OVERLOAD: Optional[int] = 80
    SOFT_ACCESSIBILITY_BONUS: Optional[int] = -10
    SOFT_MORNING_PREFERENCE: Optional[int] = 40
    SOFT_DAY_DISTRIBUTION: Optional[int] = 20
    SOFT_SIBLING_DIFFERENT_DAY: Optional[int] = 100
    SOFT_OVERLOADED_TEACHER: Optional[int] = 200
    SOFT_TEACHER_NO_BREAK: Optional[int] = 1000
    SOFT_CONSECUTIVE_HOURS_EXCEEDED: Optional[int] = 500
    SOFT_FACULTY_IDLE_TIME: Optional[int] = 600
    SOFT_FACULTY_LONG_GAP_2H: Optional[int] = 400
    SOFT_FACULTY_LONG_GAP_3H: Optional[int] = 1200
    SOFT_FACULTY_LONG_GAP_4H: Optional[int] = 3000
    SOFT_LOAD_IMBALANCE: Optional[int] = 300
    SOFT_LATE_CLASS: Optional[int] = 250
    SOFT_ROOM_IDLE_GAP: Optional[int] = 200
    SOFT_ROOM_PROFILE_EXTRA_ROOMS: Optional[int] = 900
    SOFT_ROOM_PROFILE_SPREAD: Optional[int] = 450
    SOFT_UNEVEN_SECTION_DIST: Optional[int] = 250
    SOFT_CONSECUTIVE_DAY_PENALTY: Optional[int] = 0
    SOFT_SECTION_GAP: Optional[int] = 120
    SOFT_EVENING_CLASS: Optional[int] = 350
    SOFT_NIGHT_CLASS_EXTRA: Optional[int] = 600
    SOFT_COHORT_FRAGMENTATION: Optional[int] = 250
    SOFT_COHORT_DAILY_SPAN: Optional[int] = 180
    SOFT_COHORT_EXTRA_DAYS: Optional[int] = 250
    SOFT_COHORT_LIGHT_DAY: Optional[int] = 100
    SOFT_FACULTY_EVENING_CLASS: Optional[int] = 250
    SOFT_FACULTY_EXTRA_DAYS: Optional[int] = 200
    SOFT_FACULTY_FRAGMENTATION: Optional[int] = 300
    SOFT_FACULTY_NIGHT_CLASS: Optional[int] = 200
    SOFT_FACULTY_DAILY_SPAN: Optional[int] = 500
    SOFT_VSL_SHIFT_MISMATCH: Optional[int] = 500
    SOFT_PART_TIME_SATURDAY: Optional[int] = 2000


class ScheduleGenerationResponse(BaseModel):
    """Response model for schedule generation with BulSU QSA stats"""
    success: bool
    schedule_id: int
    message: str
    total_classes: int
    scheduled_classes: int
    unscheduled_classes: int
    unscheduled_list: List[Dict[str, Any]] = [] # Detailed list of failures
    optimization_stats: Dict[str, Any]
    conflicts: List[Dict[str, Any]]
    schedule_entries: Optional[List[Dict[str, Any]]] = None  # Include entries for frontend
    online_days: Optional[List[str]] = None  # BulSU QSA: Online days used
    online_class_count: Optional[int] = 0  # BulSU QSA: Count of online classes
    physical_class_count: Optional[int] = 0  # BulSU QSA: Count of physical classes
    # Split session stats
    split_session_stats: Optional[Dict[str, Any]] = None  # Info about split sessions
    # Queue metadata
    queue_job_id: Optional[str] = None
    queue_initial_position: Optional[int] = None
    queue_wait_seconds: Optional[float] = 0
    queue_pending_after_start: Optional[int] = 0
    queue_max_wait_seconds: Optional[int] = 0


@app.post("/api/schedules/generate", response_model=ScheduleGenerationResponse)
async def generate_schedule(request: ScheduleGenerationRequest):
    """
    Generate a new class schedule using quantum-inspired annealing.
    
    This endpoint accepts data directly from frontend or fetches from database.
    Supports 30-minute time slot intervals for flexible scheduling.
    """
    try:
        queue_job_id: Optional[str] = None
        queue_info: Dict[str, Any] = {
            "job_id": None,
            "initial_position": None,
            "wait_seconds": 0,
            "pending_after_start": 0,
            "max_wait_seconds": SCHEDULE_QUEUE_MAX_WAIT_SECONDS,
        }
        queue_slot_acquired = False

        # Input validation
        if not request.schedule_name or not request.schedule_name.strip():
            raise HTTPException(status_code=400, detail="schedule_name is required")
        if not request.semester or not request.semester.strip():
            raise HTTPException(status_code=400, detail="semester is required")
        if not request.academic_year or not request.academic_year.strip():
            raise HTTPException(status_code=400, detail="academic_year is required")
        if request.max_iterations < 100:
            raise HTTPException(status_code=400, detail="max_iterations must be at least 100")
        if request.cooling_rate <= 0 or request.cooling_rate >= 1:
            raise HTTPException(status_code=400, detail="cooling_rate must be between 0 and 1")
        if request.initial_temperature <= 0:
            raise HTTPException(status_code=400, detail="initial_temperature must be positive")

        # Queue all schedule generation calls (user/admin) in FIFO order.
        queue_job_id = uuid.uuid4().hex
        queue_info = await _acquire_schedule_generation_slot(queue_job_id)
        queue_slot_acquired = True
        print(
            f"🧾 Queue slot acquired | job={queue_info['job_id']} | "
            f"initial_position={queue_info['initial_position']} | "
            f"wait={queue_info['wait_seconds']}s | "
            f"pending_after_start={queue_info['pending_after_start']}"
        )
        
        print("=" * 60)
        print("🚀 SCHEDULE GENERATION STARTED (Enhanced v2)")
        print("=" * 60)
        print(f"📋 Schedule Name: {request.schedule_name}")
        print(f"📅 Semester: {request.semester} | Year: {request.academic_year}")
        print(f"⏰ Slot Duration: {request.slot_duration} minutes")
        print(f"🏫 Campus Hours: {request.start_time} - {request.end_time}")
        
        # Use direct data from frontend if provided, otherwise fetch from database
        if request.sections_data and request.rooms_data:
            print("📦 Using data provided directly from frontend")
            sections = [s.model_dump() for s in request.sections_data]
            rooms = [r.model_dump() for r in request.rooms_data]
            
            # Generate time slots using frontend's slot duration (skip lunch gap)
            lunch_start_str = request.lunch_start if request.lunch_mode != 'none' else None
            lunch_end_str = request.lunch_end if request.lunch_mode != 'none' else None
            if request.use_enhanced_scheduler:
                # Use frontend's slot_duration (e.g., 90 minutes)
                time_slots = [
                    {
                        'id': s.id,
                        'slot_name': s.slot_name,
                        'start_time': s.start_time,
                        'end_time': s.end_time,
                        'duration_minutes': s.duration_minutes
                    }
                    for s in generate_time_slots(request.start_time, request.end_time, request.slot_duration, lunch_start=lunch_start_str, lunch_end=lunch_end_str)
                ]
                print(f"⏰ Generated {len(time_slots)} time slots of {request.slot_duration} minutes ({request.start_time} - {request.end_time}, lunch gap: {lunch_start_str}-{lunch_end_str})")
            elif request.time_slots:
                time_slots = [t.model_dump() for t in request.time_slots]
                print(f"⏰ Using {len(time_slots)} custom time slots from frontend")
            else:
                time_slots = await get_time_slots()
                print(f"⏰ Using {len(time_slots)} time slots from database")
        else:
            print("🔍 Fetching data from database")
            all_sections = await get_sections_for_scheduling(request.semester, request.academic_year)
            all_rooms = await get_all_rooms()
            
            lunch_start_str2 = request.lunch_start if request.lunch_mode != 'none' else None
            lunch_end_str2 = request.lunch_end if request.lunch_mode != 'none' else None
            if request.use_enhanced_scheduler:
                time_slots = [
                    {
                        'id': s.id,
                        'slot_name': s.slot_name,
                        'start_time': s.start_time,
                        'end_time': s.end_time,
                        'duration_minutes': s.duration_minutes
                    }
                    for s in generate_time_slots(request.start_time, request.end_time, request.slot_duration, lunch_start=lunch_start_str2, lunch_end=lunch_end_str2)
                ]
            else:
                time_slots = await get_time_slots()
            
            # Filter sections if specific IDs provided
            if request.section_ids:
                sections = [s for s in all_sections if s.get("id") in request.section_ids]
            else:
                sections = all_sections
            
            # Filter rooms if specific IDs provided
            if request.room_ids:
                rooms = [r for r in all_rooms if r.get("id") in request.room_ids]
            else:
                rooms = all_rooms
        
        print(f"📚 Sections to schedule: {len(sections)}")
        print(f"🏢 Available rooms: {len(rooms)}")
        print(f"⏰ Time slots: {len(time_slots)}")
        
        if not sections:
            raise HTTPException(status_code=400, detail="No sections to schedule")
        if not rooms:
            raise HTTPException(status_code=400, detail="No rooms available")
        if not time_slots:
            raise HTTPException(status_code=400, detail="No time slots defined")
        
        # Active days - use from request or default to Mon-Sat
        active_days = request.active_days if request.active_days else ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        print(f"📅 Active days: {', '.join(active_days)}")
        
        # Configuration for enhanced scheduler
        config = {
            "max_iterations": request.max_iterations,
            "initial_temperature": request.initial_temperature,
            "cooling_rate": request.cooling_rate,
            "max_teacher_hours_per_day": request.max_teacher_hours_per_day,
            "max_consecutive_hours": request.max_consecutive_hours,
            "prioritize_accessibility": request.prioritize_accessibility,
            "avoid_lunch_conflicts": request.avoid_lunch_conflicts,
            "active_days": active_days,
            "start_time": request.start_time,
            "end_time": request.end_time,
            "slot_duration": request.slot_duration,  # Add slot duration to config
            # NEW: Constraint settings for BulSU rules
            "lunch_mode": request.lunch_mode,
            "lunch_start_hour": request.lunch_start_hour,
            "lunch_end_hour": request.lunch_end_hour,
            "strict_lab_room_matching": request.strict_lab_room_matching,
            "strict_lecture_room_matching": request.strict_lecture_room_matching,
            # Split session settings
            "allow_split_sessions": request.allow_split_sessions,
            "combine_split_lectures": request.combine_split_lectures,
            "allow_g1_g2_split_sessions": request.allow_g1_g2_split_sessions,
            "enforce_g1_g2_equal_hours": request.enforce_g1_g2_equal_hours,
            "faculty_types": request.faculty_types,
            # Runtime resource tuning
            "low_resource_mode": request.low_resource_mode,
            "auto_low_resource_mode": request.auto_low_resource_mode,
            "cpu_yield_every_iterations": request.cpu_yield_every_iterations,
            "cpu_yield_ms": request.cpu_yield_ms,
            # Soft Penalties
            "SOFT_ROOM_TYPE_MISMATCH": request.SOFT_ROOM_TYPE_MISMATCH,
            "SOFT_ROOM_TYPE_MAJOR_MISMATCH": request.SOFT_ROOM_TYPE_MAJOR_MISMATCH,
            "SOFT_CAPACITY_WASTE": request.SOFT_CAPACITY_WASTE,
            "SOFT_LUNCH_OVERLAP": request.SOFT_LUNCH_OVERLAP,
            "SOFT_TEACHER_OVERLOAD": request.SOFT_TEACHER_OVERLOAD,
            "SOFT_ACCESSIBILITY_BONUS": request.SOFT_ACCESSIBILITY_BONUS,
            "SOFT_MORNING_PREFERENCE": request.SOFT_MORNING_PREFERENCE,
            "SOFT_DAY_DISTRIBUTION": request.SOFT_DAY_DISTRIBUTION,
            "SOFT_SIBLING_DIFFERENT_DAY": request.SOFT_SIBLING_DIFFERENT_DAY,
            "SOFT_OVERLOADED_TEACHER": request.SOFT_OVERLOADED_TEACHER,
            "SOFT_TEACHER_NO_BREAK": request.SOFT_TEACHER_NO_BREAK,
            "SOFT_CONSECUTIVE_HOURS_EXCEEDED": request.SOFT_CONSECUTIVE_HOURS_EXCEEDED,
            "SOFT_FACULTY_IDLE_TIME": request.SOFT_FACULTY_IDLE_TIME,
            "SOFT_FACULTY_LONG_GAP_2H": request.SOFT_FACULTY_LONG_GAP_2H,
            "SOFT_FACULTY_LONG_GAP_3H": request.SOFT_FACULTY_LONG_GAP_3H,
            "SOFT_FACULTY_LONG_GAP_4H": request.SOFT_FACULTY_LONG_GAP_4H,
            "SOFT_LOAD_IMBALANCE": request.SOFT_LOAD_IMBALANCE,
            "SOFT_LATE_CLASS": request.SOFT_LATE_CLASS,
            "SOFT_ROOM_IDLE_GAP": request.SOFT_ROOM_IDLE_GAP,
            "SOFT_ROOM_PROFILE_EXTRA_ROOMS": request.SOFT_ROOM_PROFILE_EXTRA_ROOMS,
            "SOFT_ROOM_PROFILE_SPREAD": request.SOFT_ROOM_PROFILE_SPREAD,
            "SOFT_UNEVEN_SECTION_DIST": request.SOFT_UNEVEN_SECTION_DIST,
            "SOFT_CONSECUTIVE_DAY_PENALTY": request.SOFT_CONSECUTIVE_DAY_PENALTY,
            "SOFT_SECTION_GAP": request.SOFT_SECTION_GAP,
            "SOFT_EVENING_CLASS": request.SOFT_EVENING_CLASS,
            "SOFT_NIGHT_CLASS_EXTRA": request.SOFT_NIGHT_CLASS_EXTRA,
            "SOFT_COHORT_FRAGMENTATION": request.SOFT_COHORT_FRAGMENTATION,
            "SOFT_COHORT_DAILY_SPAN": request.SOFT_COHORT_DAILY_SPAN,
            "SOFT_COHORT_EXTRA_DAYS": request.SOFT_COHORT_EXTRA_DAYS,
            "SOFT_COHORT_LIGHT_DAY": request.SOFT_COHORT_LIGHT_DAY,
            "SOFT_FACULTY_EVENING_CLASS": request.SOFT_FACULTY_EVENING_CLASS,
            "SOFT_FACULTY_EXTRA_DAYS": request.SOFT_FACULTY_EXTRA_DAYS,
            "SOFT_FACULTY_FRAGMENTATION": request.SOFT_FACULTY_FRAGMENTATION,
            "SOFT_FACULTY_NIGHT_CLASS": request.SOFT_FACULTY_NIGHT_CLASS,
            "SOFT_FACULTY_DAILY_SPAN": request.SOFT_FACULTY_DAILY_SPAN,
            "SOFT_VSL_SHIFT_MISMATCH": request.SOFT_VSL_SHIFT_MISMATCH,
            "SOFT_PART_TIME_SATURDAY": request.SOFT_PART_TIME_SATURDAY
        }

        # Render/free-tier guardrails: keep the instance responsive and avoid CPU/memory runaway.
        # This does NOT change results correctness; it only limits how deep optimization goes.
        if _is_free_tier_resource_profile_enabled():
            problem_scale = max(1, len(sections)) * max(1, len(rooms)) * max(1, len(time_slots))

            # Enable yielding (stability) even for smaller inputs.
            config["low_resource_mode"] = True
            config["auto_low_resource_mode"] = True

            # Cap iterations; large problems get a lower cap.
            base_cap = int(os.getenv("SCHEDULER_MAX_ITERATIONS_CAP", "2500"))
            large_problem_scale = int(os.getenv("SCHEDULER_LARGE_PROBLEM_SCALE", "200000"))
            large_cap = int(os.getenv("SCHEDULER_MAX_ITERATIONS_CAP_LARGE", "1600"))
            cap = large_cap if problem_scale >= large_problem_scale else base_cap
            config["max_iterations"] = max(500, min(int(config.get("max_iterations") or 0), cap))

            # Reduce expensive full re-optimizations on free tier.
            adaptive_passes = int(os.getenv("SCHEDULER_ADAPTIVE_RETRY_PASSES", "1"))
            config["adaptive_retry_passes"] = max(0, adaptive_passes)
        
        print("🎯 Running Enhanced Quantum-Inspired Annealing Algorithm...")
        print(f"   Max Iterations: {config['max_iterations']}")
        print(f"   Initial Temperature: {config['initial_temperature']}")
        print(f"   Cooling Rate: {config['cooling_rate']}")
        print(f"   ⏰ Slot Duration: {config['slot_duration']} minutes")
        print(f"   🍽️ Lunch Mode: {config['lunch_mode']} ({config['lunch_start_hour']}:00-{config['lunch_end_hour']}:00)")
        print(f"   🔬 Strict Lab Matching: {config['strict_lab_room_matching']}")
        print(f"   ✂️ Allow Split Sessions: {config['allow_split_sessions']}")
        if request.online_days:
            print(f"   🌐 Online Days: {', '.join(request.online_days)}")
        
        # Fetch teacher profiles for constraints (VSL, shifts, etc.)
        print("👤 Fetching teacher profiles for constraints...")
        all_teachers = await get_all_teachers()
        
        # Run the enhanced scheduler with 30-minute slots and BulSU QSA
        if request.use_enhanced_scheduler:
            result = await asyncio.to_thread(
                run_enhanced_scheduler,
                sections_data=sections,
                rooms_data=rooms,
                time_slots_data=time_slots,
                config=config,
                online_days=request.online_days,  # BulSU QSA: Pass online days
                faculty_profiles_data=all_teachers,  # Pass teacher data for constraints
                fixed_allocations=request.fixed_allocations # NEW: Manual edits to prioritize
            )
            # Map result to expected format
            result["schedule_entries"] = result.get("allocations", [])
            online_count = result.get("online_class_count", 0)
            physical_count = result.get("physical_class_count", 0)
            result["message"] = (
                f"Enhanced scheduler completed. Scheduled {result['scheduled_sections']}/{result['total_sections']} "
                f"sections with {request.slot_duration}-minute time slots. ({online_count} online, {physical_count} physical)"
            )
            result["conflicts"] = []  # Enhanced scheduler handles conflicts internally
        else:
            # Fallback to original scheduler
            result = await asyncio.to_thread(
                run_scheduler,
                sections_data=sections,
                rooms_data=rooms,
                time_slots_data=time_slots,
                config=config
            )
        
        print(f"✅ Scheduling complete!")
        print(f"   Success: {result['success']}")
        print(f"   Scheduled: {result['scheduled_sections']}/{result['total_sections']}")
        print(f"   Unscheduled: {result['unscheduled_sections']}")
        
        # Save to generated_schedules table
        generated_schedule_data = {
            "schedule_name": request.schedule_name,
            "semester": request.semester,
            "academic_year": request.academic_year,
            "campus_group_id": request.campus_group_id or 1,  # Use request value or default to 1
            "class_group_id": request.class_group_id or 1,    # Use request value or default to 1
            "total_classes": result["total_sections"],
            "scheduled_classes": result["scheduled_sections"],
            "unscheduled_classes": result["unscheduled_sections"],
            "optimization_stats": result["optimization_stats"],
            "status": "completed" if result["unscheduled_sections"] == 0 else "partial"
        }
        
        # Actually save the schedule metadata to get an ID
        generated_schedule_obj = await create_generated_schedule(generated_schedule_data)
        generated_schedule_id = generated_schedule_obj.get("id") if isinstance(generated_schedule_obj, dict) else generated_schedule_obj
        print(f"📊 DEBUG: Created generated_schedule with ID: {generated_schedule_id}")
        
        if result["schedule_entries"] and generated_schedule_id:
            # DEBUG: Log the number of schedule entries received
            print(f"📊 DEBUG: Received {len(result['schedule_entries'])} schedule entries from scheduler")
            
            # STEP 1: Build allocations from schedule entries
            all_allocations = []
            for entry in result["schedule_entries"]:
                allocation = {
                    "schedule_id": generated_schedule_id,
                    "class_id": entry.get("section_id"),
                    "section_id": entry.get("section_id"),  # NEW: section_id for analytics
                    "room_id": entry.get("room_id"),
                    "course_code": entry.get("course_code", ""),
                    "course_name": entry.get("course_name", ""),
                    "section": entry.get("section_code", ""),
                    "section_code": entry.get("section_code", ""),
                    "year_level": entry.get("year_level", 1),
                    "schedule_day": entry.get("day_of_week", ""),
                    "day_of_week": entry.get("day_of_week", ""),  # NEW: day_of_week for analytics
                    "schedule_time": f"{entry.get('start_time', '')} - {entry.get('end_time', '')}",
                    "campus": entry.get("campus", ""),
                    "building": entry.get("building", ""),
                    "room": entry.get("room_code", entry.get("room_name", "")),
                    "room_code": entry.get("room_code", ""),
                    "capacity": entry.get("room_capacity", 0),
                    "teacher_id": entry.get("teacher_id"),  # NEW: teacher_id for analytics
                    "teacher_name": entry.get("teacher_name", ""),
                    "department": entry.get("department", ""),
                    "lec_hours": entry.get("lec_hours", 0),
                    "lab_hours": entry.get("lab_hours", 0),
                    "component": entry.get("component") or entry.get("section_type", "lecture"),
                    "status": "scheduled"
                }
                all_allocations.append(allocation)
            
            # DEBUG: Log first and last allocation
            print(f"📊 DEBUG: Built {len(all_allocations)} allocation entries to save")
            if all_allocations:
                print(f"   First: {all_allocations[0].get('course_code')} - {all_allocations[0].get('schedule_day')} {all_allocations[0].get('schedule_time')}")
                if len(all_allocations) > 1:
                    print(f"   Last: {all_allocations[-1].get('course_code')} - {all_allocations[-1].get('schedule_day')} {all_allocations[-1].get('schedule_time')}")
            
            # Save all allocations without merging LAB/LEC
            # The frontend will handle combining them for display
            print(f"✅ Prepared {len(all_allocations)} room allocation entries")
            saved_allocations = await save_room_allocations(all_allocations)
            print(f"✅ Saved {len(saved_allocations)} room allocations to database")
        
        print("=" * 60)
        print("🎉 SCHEDULE GENERATION COMPLETED")
        print("=" * 60)
        
        return ScheduleGenerationResponse(
            success=result["success"],
            schedule_id=int(generated_schedule_id) if generated_schedule_id else 0,
            message=result["message"],
            total_classes=result["total_sections"],
            scheduled_classes=result["scheduled_sections"],
            unscheduled_classes=result["unscheduled_sections"],
            unscheduled_list=result.get("unscheduled_list", []),
            optimization_stats=result["optimization_stats"],
            conflicts=result["conflicts"],
            schedule_entries=result["schedule_entries"],  # Include for frontend
            online_days=result.get("online_days", []),  # BulSU QSA
            online_class_count=result.get("online_class_count", 0),  # BulSU QSA
            physical_class_count=result.get("physical_class_count", 0),  # BulSU QSA
            split_session_stats=result.get("split_session_stats"),  # Split session info
            queue_job_id=queue_info.get("job_id"),
            queue_initial_position=queue_info.get("initial_position"),
            queue_wait_seconds=queue_info.get("wait_seconds", 0),
            queue_pending_after_start=queue_info.get("pending_after_start", 0),
            queue_max_wait_seconds=queue_info.get("max_wait_seconds", SCHEDULE_QUEUE_MAX_WAIT_SECONDS),
        )

    except ScheduleQueueTimeoutError as e:
        raise HTTPException(
            status_code=429,
            detail={
                "message": "Schedule generation queue is busy. Please retry shortly.",
                "reason": str(e),
                "max_wait_seconds": SCHEDULE_QUEUE_MAX_WAIT_SECONDS,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = str(e) if str(e) else repr(e)
        error_type = type(e).__name__
        print("=" * 60)
        print(f"❌ ERROR: {error_msg}")
        print(f"❌ ERROR TYPE: {error_type}")
        print("❌ TRACEBACK:")
        traceback.print_exc()
        print("=" * 60)
        raise HTTPException(status_code=500, detail=f"Schedule generation failed: {error_type}: {error_msg}")
    finally:
        if queue_slot_acquired and queue_job_id:
            await _release_schedule_generation_slot(queue_job_id)


@app.get("/api/schedules")
async def list_schedules():
    """Get all schedules"""
    try:
        schedules = await get_all_schedules()
        return schedules
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve schedules")


@app.get("/api/schedules/{schedule_id}")
async def get_schedule(schedule_id: int):
    """Get a specific schedule with all entries"""
    try:
        schedule = await get_schedule_by_id(schedule_id)
        if not schedule:
            raise HTTPException(status_code=404, detail="Schedule not found")
        return schedule
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve schedule")


@app.delete("/api/schedules/{schedule_id}")
async def remove_schedule(schedule_id: int):
    """Delete a schedule and all its entries"""
    try:
        await delete_schedule(schedule_id)
        return {"message": "Schedule deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete schedule")


# ========================
# Generated Schedules (QIA Results with Room Allocations)
# ========================

@app.get("/api/generated-schedules")
async def list_generated_schedules():
    """Get all generated schedules with QIA results"""
    try:
        schedules = await get_generated_schedules()
        return schedules
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve generated schedules")


@app.get("/api/generated-schedules/{schedule_id}")
async def get_generated_schedule(schedule_id: int):
    """Get a specific generated schedule with all room allocations"""
    try:
        schedule = await get_generated_schedule_by_id(schedule_id)
        if not schedule:
            raise HTTPException(status_code=404, detail="Generated schedule not found")
        return schedule
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve generated schedule")


@app.get("/api/generated-schedules/{schedule_id}/allocations")
async def get_schedule_allocations(schedule_id: int):
    """Get all room allocations for a specific generated schedule"""
    try:
        allocations = await get_room_allocations_by_schedule(schedule_id)
        return {
            "schedule_id": schedule_id,
            "allocations": allocations,
            "total": len(allocations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve allocations")


@app.delete("/api/generated-schedules/{schedule_id}")
async def remove_generated_schedule(schedule_id: int):
    """Delete a generated schedule and all its room allocations"""
    try:
        await delete_generated_schedule(schedule_id)
        return {"message": "Generated schedule deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete generated schedule")


# ========================
# Schedule Queries
# ========================

@app.get("/api/schedules/{schedule_id}/by-room/{room_id}")
async def get_room_schedule(schedule_id: int, room_id: int):
    """Get schedule entries for a specific room"""
    try:
        schedule = await get_schedule_by_id(schedule_id)
        if not schedule:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        entries = schedule.get("entries", [])
        room_entries = [e for e in entries if e.get("room_id") == room_id]
        
        return {
            "room_id": room_id,
            "schedule_id": schedule_id,
            "entries": room_entries
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve room schedule")


@app.get("/api/schedules/{schedule_id}/by-teacher/{teacher_id}")
async def get_teacher_schedule(schedule_id: int, teacher_id: int):
    """Get schedule entries for a specific teacher"""
    try:
        schedule = await get_schedule_by_id(schedule_id)
        if not schedule:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        entries = schedule.get("entries", [])
        teacher_entries = [e for e in entries if e.get("teacher_id") == teacher_id]
        
        return {
            "teacher_id": teacher_id,
            "schedule_id": schedule_id,
            "entries": teacher_entries
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve teacher schedule")


@app.get("/api/schedules/{schedule_id}/by-section/{section_id}")
async def get_section_schedule(schedule_id: int, section_id: int):
    """Get schedule entries for a specific section"""
    try:
        schedule = await get_schedule_by_id(schedule_id)
        if not schedule:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        entries = schedule.get("entries", [])
        section_entries = [e for e in entries if e.get("section_id") == section_id]
        
        return {
            "section_id": section_id,
            "schedule_id": schedule_id,
            "entries": section_entries
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve section schedule")


@app.get("/api/schedules/{schedule_id}/by-day/{day}")
async def get_day_schedule(schedule_id: int, day: str):
    """Get schedule entries for a specific day"""
    try:
        schedule = await get_schedule_by_id(schedule_id)
        if not schedule:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        entries = schedule.get("entries", [])
        day_entries = [e for e in entries if e.get("day_of_week", "").lower() == day.lower()]
        
        return {
            "day": day,
            "schedule_id": schedule_id,
            "entries": day_entries
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve day schedule")


# ========================
# Conflict Detection
# ========================

@app.get("/api/schedules/{schedule_id}/conflicts")
async def check_conflicts(schedule_id: int):
    """Check for conflicts in a schedule"""
    try:
        schedule = await get_schedule_by_id(schedule_id)
        if not schedule:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        # Get allocations for this schedule
        allocations = await get_room_allocations_by_schedule(schedule_id)
        
        # Detect room-time conflicts: same room booked twice on same day/time
        room_conflicts = []
        room_day_time_map = {}
        for alloc in allocations:
            key = (alloc.get("room_id"), alloc.get("schedule_day"), alloc.get("schedule_time"))
            if key in room_day_time_map:
                room_conflicts.append({
                    "type": "room_double_booking",
                    "room_id": key[0],
                    "day": key[1],
                    "time": key[2],
                    "allocations": [room_day_time_map[key], alloc]
                })
            else:
                room_day_time_map[key] = alloc
        
        # Detect teacher-time conflicts: same teacher booked twice on same day/time
        teacher_conflicts = []
        teacher_day_time_map = {}
        for alloc in allocations:
            teacher = alloc.get("teacher_name") or ""
            if not teacher:
                continue
            key = (teacher, alloc.get("schedule_day"), alloc.get("schedule_time"))
            if key in teacher_day_time_map:
                teacher_conflicts.append({
                    "type": "teacher_double_booking",
                    "teacher": key[0],
                    "day": key[1],
                    "time": key[2],
                    "allocations": [teacher_day_time_map[key], alloc]
                })
            else:
                teacher_day_time_map[key] = alloc
        
        return {
            "schedule_id": schedule_id,
            "has_conflicts": len(room_conflicts) > 0 or len(teacher_conflicts) > 0,
            "room_conflicts": room_conflicts,
            "teacher_conflicts": teacher_conflicts,
            "total_conflicts": len(room_conflicts) + len(teacher_conflicts),
            "total_allocations": len(allocations)
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Conflict check failed")


# ========================
# Analytics
# ========================

@app.get("/api/analytics/room-utilization")
async def get_analytics_room_utilization(schedule_id: Optional[int] = None):
    """Get room utilization analytics"""
    try:
        if not schedule_id:
            raise HTTPException(status_code=400, detail="schedule_id is required")
        
        utilization_data = await get_room_utilization(schedule_id)
        
        # Calculate averages
        total_utilization = sum(r.get("utilization_percentage", 0) for r in utilization_data)
        avg_utilization = total_utilization / len(utilization_data) if utilization_data else 0
        
        return {
            "schedule_id": schedule_id,
            "average_utilization": avg_utilization,
            "total_rooms": len(utilization_data),
            "rooms": utilization_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics")


@app.get("/api/analytics/summary")
async def get_analytics_summary(schedule_id: Optional[int] = None):
    """Get overall scheduling analytics summary"""
    try:
        rooms = await get_all_rooms()
        sections = await get_all_sections()
        teachers = await get_all_teachers()
        
        summary = {
            "total_rooms": len(rooms),
            "total_sections": len(sections),
            "total_teachers": len(teachers),
            "rooms_by_type": {},
            "rooms_by_building": {},
            "capacity_distribution": {
                "small": 0,  # < 30
                "medium": 0,  # 30-60
                "large": 0,  # 60-100
                "extra_large": 0  # > 100
            }
        }
        
        for room in rooms:
            # Count by type
            room_type = room.get("room_type", "unknown")
            summary["rooms_by_type"][room_type] = summary["rooms_by_type"].get(room_type, 0) + 1
            
            # Count by building
            building = room.get("building", "unknown")
            summary["rooms_by_building"][building] = summary["rooms_by_building"].get(building, 0) + 1
            
            # Capacity distribution
            capacity = room.get("capacity", 0)
            if capacity < 30:
                summary["capacity_distribution"]["small"] += 1
            elif capacity < 60:
                summary["capacity_distribution"]["medium"] += 1
            elif capacity < 100:
                summary["capacity_distribution"]["large"] += 1
            else:
                summary["capacity_distribution"]["extra_large"] += 1
        
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========================
# Run Server
# ========================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
