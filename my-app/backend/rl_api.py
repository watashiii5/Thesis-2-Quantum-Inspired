"""
FastAPI endpoints for RL Scheduling Engine
Integrate with existing scheduler to enable learning from custom rules.
"""

from fastapi import APIRouter, HTTPException
import asyncio
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from rl_scheduler import (
    RLSchedulingEngine, CustomRule,
    create_no_friday_afternoon_rule,
    create_teacher_lunch_break_rule,
    create_room_clustering_rule
)
from rl_persistence import RLStatePersistence


# ==================== Pydantic Models ====================

class CustomRuleRequest(BaseModel):
    """Request model for adding a custom rule."""
    rule_id: str
    name: str
    description: str
    penalty: float
    rule_type: Optional[str] = None  # 'no_friday_afternoon', 'teacher_lunch', 'room_clustering', etc.


class TrainRLRequest(BaseModel):
    """Request to train RL agent on custom rules."""
    classes: List[Dict[str, Any]]
    rooms: List[Dict[str, Any]]
    time_slots: List[str]
    iterations: int = 10
    persist: bool = True


class GetLearnedScheduleRequest(BaseModel):
    """Request to get schedule optimized with learned policy."""
    classes: List[Dict[str, Any]]
    rooms: List[Dict[str, Any]]
    time_slots: List[str]


class RLStatsResponse(BaseModel):
    """RL agent statistics."""
    episodes_trained: int
    avg_reward: float
    best_reward: Optional[float] = None
    worst_reward: Optional[float] = None
    agent_stats: Dict[str, Any]


class TrainResponseModel(BaseModel):
    """Response from training endpoint."""
    episode: int
    avg_reward: float
    agent_stats: Dict[str, Any]


# ==================== Router Setup ====================

rl_router = APIRouter(prefix="/api/rl", tags=["RL Scheduling"])

# Global RL Engine (in production, use database or cache)
rl_engine = RLSchedulingEngine()
rl_persistence = RLStatePersistence(supabase_client=None)  # Set by initialize_persistence()

# Register built-in rules
rl_engine.add_custom_rule(create_no_friday_afternoon_rule())
rl_engine.add_custom_rule(create_teacher_lunch_break_rule())
rl_engine.add_custom_rule(create_room_clustering_rule())


# ==================== QIA (scheduler_v2) Outcome Tracking ====================

_qia_best_cost: Optional[float] = None
_qia_best_result: Optional[Dict[str, Any]] = None
_qia_run_count: int = 0


class QiaSampleRunRequest(BaseModel):
    """Run a small sample QIA optimization and return outcome metrics."""
    max_iterations: int = 600
    initial_temperature: float = 1500.0
    cooling_rate: float = 0.95
    slot_duration: int = 90
    start_time: str = "07:00"
    end_time: str = "20:00"
    lunch_mode: str = "auto"  # auto/strict/flexible/none
    low_resource_mode: bool = True


@rl_router.post("/qia/run-sample", summary="Run QIA Sample Optimization")
async def run_qia_sample(request: QiaSampleRunRequest):
    """Run scheduler_v2 on a small built-in dataset and track best (lowest) final_cost."""
    global _qia_best_cost, _qia_best_result, _qia_run_count

    try:
        from scheduler_v2 import run_enhanced_scheduler
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import scheduler_v2: {e}")

    sections_data: List[Dict[str, Any]] = [
        {
            "id": 1,
            "section_code": "BSCS-1A",
            "course_code": "BSCS",
            "course_name": "BSCS",
            "subject_code": "CS101",
            "subject_name": "Intro to CS",
            "teacher_id": 11,
            "teacher_name": "Teacher A",
            "year_level": 1,
            "student_count": 30,
            "lec_hours": 3,
            "lab_hours": 0,
        },
        {
            "id": 2,
            "section_code": "BSCS-1B",
            "course_code": "BSCS",
            "course_name": "BSCS",
            "subject_code": "CS102",
            "subject_name": "Data Structures",
            "teacher_id": 12,
            "teacher_name": "Teacher B",
            "year_level": 1,
            "student_count": 25,
            "lec_hours": 3,
            "lab_hours": 0,
        },
        {
            "id": 3,
            "section_code": "BSCS-1A",
            "course_code": "BSCS",
            "course_name": "BSCS",
            "subject_code": "CS101L",
            "subject_name": "Intro to CS Lab",
            "teacher_id": 11,
            "teacher_name": "Teacher A",
            "year_level": 1,
            "student_count": 30,
            "lec_hours": 0,
            "lab_hours": 2,
        },
    ]

    rooms_data: List[Dict[str, Any]] = [
        {
            "id": 101,
            "room_code": "R101",
            "room_name": "Lecture Room 101",
            "building": "Main",
            "campus": "Main Campus",
            "capacity": 50,
            "room_type": "lecture",
        },
        {
            "id": 102,
            "room_code": "R102",
            "room_name": "Lecture Room 102",
            "building": "Main",
            "campus": "Main Campus",
            "capacity": 40,
            "room_type": "lecture",
        },
        {
            "id": 201,
            "room_code": "LAB201",
            "room_name": "Computer Lab 201",
            "building": "Main",
            "campus": "Main Campus",
            "capacity": 30,
            "room_type": "computer lab",
        },
    ]

    config: Dict[str, Any] = {
        "max_iterations": max(100, int(request.max_iterations)),
        "initial_temperature": float(request.initial_temperature),
        "cooling_rate": float(request.cooling_rate),
        "slot_duration": int(request.slot_duration),
        "start_time": request.start_time,
        "end_time": request.end_time,
        "lunch_mode": request.lunch_mode,
        "low_resource_mode": bool(request.low_resource_mode),
        "auto_low_resource_mode": bool(request.low_resource_mode),
    }

    result = await asyncio.to_thread(
        run_enhanced_scheduler,
        sections_data=sections_data,
        rooms_data=rooms_data,
        time_slots_data=None,
        config=config,
        online_days=[],
        faculty_profiles_data=None,
        fixed_allocations=None,
    )

    _qia_run_count += 1

    final_cost = None
    try:
        final_cost = float((result.get("optimization_stats") or {}).get("final_cost"))
    except Exception:
        final_cost = None

    if result.get("success") and final_cost is not None:
        if _qia_best_cost is None or final_cost < _qia_best_cost:
            _qia_best_cost = final_cost
            _qia_best_result = result

    return {
        "run_count": _qia_run_count,
        "final_cost": final_cost,
        "result": result,
    }


@rl_router.get("/qia/best", summary="Get Best QIA Outcome")
async def get_best_qia_result():
    """Return best (lowest) QIA final_cost seen during /qia/run-sample calls."""
    return {
        "run_count": _qia_run_count,
        "best_cost": _qia_best_cost,
        "best_result": _qia_best_result,
    }


@rl_router.post("/qia/reset", summary="Reset QIA Outcome Tracker")
async def reset_qia_tracker():
    """Reset in-memory QIA run counters and best result tracking."""
    global _qia_best_cost, _qia_best_result, _qia_run_count
    _qia_best_cost = None
    _qia_best_result = None
    _qia_run_count = 0
    return {"status": "success"}


def initialize_persistence(supabase_client):
    """Initialize persistence with Supabase client."""
    global rl_persistence
    rl_persistence.supabase = supabase_client



# ==================== Endpoints ====================

@rl_router.post("/rules/add", summary="Add Custom Rule")
async def add_custom_rule(request: CustomRuleRequest):
    """
    Add a custom scheduling rule for the RL agent to learn.

    **Example:**
    ```json
    {
        "rule_id": "no_7am",
        "name": "No 7 AM Classes",
        "description": "Avoid scheduling classes before 8 AM",
        "penalty": 250
    }
    ```
    """
    try:
        rule = CustomRule(
            rule_id=request.rule_id,
            name=request.name,
            description=request.description,
            penalty=request.penalty
        )
        rl_engine.add_custom_rule(rule)
        return {
            "status": "success",
            "message": f"Rule '{request.name}' added successfully",
            "rule_id": request.rule_id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@rl_router.post("/train", summary="Train RL Agent", response_model=TrainResponseModel)
async def train_rl_agent(request: TrainRLRequest):
    """
    Train the RL agent on provided classes and rules.

    - **classes**: List of class objects with id, name, capacity_needed
    - **rooms**: List of room objects with id, capacity, type
    - **time_slots**: List of available time slots (format: "monday_08:00")
    - **iterations**: Number of training iterations (default: 10)

    Returns training metrics including average reward and agent statistics.

    **Example:**
    ```json
    {
        "classes": [{"id": "cs101", "name": "Intro to CS", "capacity_needed": 30}],
        "rooms": [{"id": "r101", "capacity": 50, "type": "lecture"}],
        "time_slots": ["monday_08:00", "monday_09:00"],
        "iterations": 10
    }
    ```
    """
    try:
        result = rl_engine.train_episode(
            classes=request.classes,
            rooms=request.rooms,
            time_slots=request.time_slots,
            iterations=request.iterations
        )

        # Save agent state to Supabase after training (optional; can be noisy in tight loops)
        if request.persist:
            await rl_persistence.save_agent_state(rl_engine)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@rl_router.get("/best", summary="Get Best Observed Result")
async def get_best_result():
    """Get the best (lowest-cost) schedule observed during training."""
    try:
        return rl_engine.get_best_result()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _parse_qia_rules_from_scheduler_v2_doc() -> Dict[str, Any]:
    """Extract a human-readable constraint list from scheduler_v2's cost-function docstring."""
    try:
        from scheduler_v2 import EnhancedQuantumScheduler, HARD_CONSTRAINT_PENALTY
    except Exception:
        return {
            "hard_penalty": None,
            "hard_constraints": [],
            "soft_constraints": [],
            "source": "scheduler_v2 import failed"
        }

    doc = (EnhancedQuantumScheduler._calculate_cost.__doc__ or "").splitlines()
    hard: List[str] = []
    soft: List[str] = []
    mode: Optional[str] = None

    for raw in doc:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("HARD CONSTRAINTS"):
            mode = "hard"
            continue
        if line.startswith("SOFT CONSTRAINTS"):
            mode = "soft"
            continue

        if mode == "hard":
            if line[0].isdigit() and "." in line:
                # e.g. "1. The Ghost Room: ..."
                item = line.split(".", 1)[1].strip()
                hard.append(item)
        elif mode == "soft":
            if line.startswith("-"):
                soft.append(line.lstrip("-").strip())

    return {
        "hard_penalty": int(HARD_CONSTRAINT_PENALTY),
        "hard_constraints": hard,
        "soft_constraints": soft,
        "source": "scheduler_v2.EnhancedQuantumScheduler._calculate_cost.__doc__"
    }


@rl_router.get("/qia-rules", summary="List QIA (Scheduler v2) Constraints")
async def list_qia_rules():
    """Return the constraints implemented in the QIA cost function (scheduler_v2)."""
    try:
        return _parse_qia_rules_from_scheduler_v2_doc()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@rl_router.post("/persist", summary="Persist RL Agent State")
async def persist_agent_state():
    """Persist current RL agent state to Supabase without training."""
    try:
        ok = await rl_persistence.save_agent_state(rl_engine)
        return {"status": "success" if ok else "skipped", "persisted": ok}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@rl_router.post("/schedule/learned", summary="Get RL-Optimized Schedule")
async def get_learned_schedule(request: GetLearnedScheduleRequest):
    """
    Get a schedule optimized using the RL agent's learned policy.

    After training the agent on custom rules, use this endpoint to generate
    schedules that respect learned preferences.

    Returns a mapping of class_id -> (room_id, time_slot)
    """
    try:
        schedule = rl_engine.get_learned_schedule(
            classes=request.classes,
            rooms=request.rooms,
            time_slots=request.time_slots
        )
        return {
            "status": "success",
            "schedule": schedule,
            "timestamp": None  # Will be filled with datetime
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@rl_router.get("/stats", summary="Get Training Statistics", response_model=RLStatsResponse)
async def get_rl_stats():
    """Get overall RL agent training statistics and learning progress."""
    try:
        stats = rl_engine.get_training_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@rl_router.get("/rules", summary="List Registered Rules")
async def list_rules():
    """Get list of all registered custom rules."""
    try:
        rules = []
        for rule in rl_engine.environment.custom_rules:
            rules.append({
                "rule_id": rule.rule_id,
                "name": rule.name,
                "description": rule.description,
                "penalty": rule.penalty
            })
        return {
            "count": len(rules),
            "rules": rules
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@rl_router.get("/training-history", summary="Get Training History")
async def get_training_history(limit: int = 20):
    """Get last N training episodes history."""
    try:
        history = rl_engine.training_history[-limit:]
        return {
            "episodes": len(history),
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@rl_router.post("/reset", summary="Reset RL Agent")
async def reset_agent():
    """Reset the RL agent (clears learned Q-values and history)."""
    try:
        global rl_engine
        rl_engine = RLSchedulingEngine()
        # Re-add built-in rules
        rl_engine.add_custom_rule(create_no_friday_afternoon_rule())
        rl_engine.add_custom_rule(create_teacher_lunch_break_rule())
        rl_engine.add_custom_rule(create_room_clustering_rule())
        # Also delete from database
        await rl_persistence.reset_agent_state()
        return {"status": "success", "message": "RL agent reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Integration with Existing Scheduler ====================

def integrate_rl_with_qia(qia_scheduler_fn):
    """
    Connect RL engine to your existing QIA scheduler.
    Pass your run_enhanced_scheduler function.
    """
    rl_engine.qia_scheduler = qia_scheduler_fn
    return rl_engine
