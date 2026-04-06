# server.py
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import deque
from contextlib import asynccontextmanager
import uuid
import time
import asyncio
from typing import Dict, Optional
import os 

# ========== Configuration ==========
TASK_TIMEOUT = 60*5      # seconds minutes without worker ping before task is re-queued  (5mins)
CLIENT_TIMEOUT = 25    # seconds without client polling before task is abandoned  (25s)
CLEANUP_INTERVAL = 10  # seconds between timeout checks (10s)

STORAGE_DIR = "audio_storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

# ========== Data Structures ==========
task_queue = deque()                     # each task: {id, data, client_id, last_seen, ...}
active_tasks: Dict[str, dict] = {}       # task_id -> task details
results: Dict[str, dict] = {}            # task_id -> result

# ========== Lifespan for background cleanup ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: start background task
    cleanup_task = asyncio.create_task(background_cleanup())
    yield
    # Shutdown: cancel background task
    cleanup_task.cancel()
    await cleanup_task

app = FastAPI(
    title="TTS Task Queue Server",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],            # Allows specific list of origins
    allow_credentials=True,          
    allow_methods=["*"],              
    allow_headers=["*"],       
)


# ========== Pydantic Models ==========
class TaskSubmitRequest(BaseModel):
    text: str
    voice_ref: str
    speed: Optional[float] = 0.9
    client_id: str

class TaskSubmitResponse(BaseModel):
    task_id: str
    status: str

class ClaimResponse(BaseModel):
    has_work: bool
    task: Optional[dict] = None

class WorkCompleteRequest(BaseModel):
    task_id: str
    worker_id: str
    audio_url: str

class WorkPingRequest(BaseModel):
    task_id: str
    worker_id: str

class ResultResponse(BaseModel):
    status: str  # "pending", "processing", "completed", "not_found"
    audio_url: Optional[str] = None

# ========== Helper Functions ==========
def update_task_last_seen(task_id: str, client_id: str) -> bool:
    """Update last_seen for a task in queue or active. Returns True if found."""
    # Check in queue
    for task in task_queue:
        if task["id"] == task_id and task["client_id"] == client_id:
            task["last_seen"] = time.time()
            return True
    # Check in active tasks
    if task_id in active_tasks and active_tasks[task_id]["client_id"] == client_id:
        active_tasks[task_id]["last_seen"] = time.time()
        return True
    return False

# ========== Endpoints ==========
@app.post("/submit", response_model=TaskSubmitResponse)
async def submit_task(request: TaskSubmitRequest):
    """Client submits a new task."""
    task_id = str(uuid.uuid4())
    now = time.time()
    task = {
        "id": task_id,
        "text": request.text,
        "voice_ref": request.voice_ref,
        "speed": request.speed,
        "client_id": request.client_id,
        "submitted_at": now,
        "last_seen": now
    }
    task_queue.append(task)
    return TaskSubmitResponse(task_id=task_id, status="queued")

@app.get("/claim", response_model=ClaimResponse)
async def claim_task(worker_id: str):
    """Worker requests next available task."""
    # Run a quick cleanup to avoid assigning stale tasks
    cleanup_task_timeouts()
    cleanup_client_timeouts()
    
    if not task_queue:
        return ClaimResponse(has_work=False, task=None)
    
    task = task_queue.popleft()
    now = time.time()
    
    # Move to active_tasks
    active_tasks[task["id"]] = {
        "id": task["id"],
        "text": task["text"],
        "voice_ref": task["voice_ref"],
        "speed": task["speed"],
        "client_id": task["client_id"],
        "worker_id": worker_id,
        "claimed_at": now,
        "last_ping": now,
        "last_seen": task["last_seen"]
    }
    
    return ClaimResponse(has_work=True, task={
        "id": task["id"],
        "text": task["text"],
        "speed": task["speed"],
        "voice_ref": task["voice_ref"]
    })

@app.post("/work/ping")
async def work_ping(request: WorkPingRequest):
    """Worker updates heartbeat for a task still in progress."""
    if request.task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found or already completed")
    
    if active_tasks[request.task_id]["worker_id"] != request.worker_id:
        raise HTTPException(status_code=403, detail="Worker ID mismatch")
    
    active_tasks[request.task_id]["last_ping"] = time.time()
    return {"status": "alive"}


@app.post("/work/complete")
async def work_complete(request: WorkCompleteRequest):
    """Worker submits completed task result."""
    if request.task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found or already completed")
    
    task = active_tasks.pop(request.task_id)
    
    results[request.task_id] = {
        "audio_url": request.audio_url,
        "client_id": task["client_id"],
        "completed_at": time.time()
    }
    
    return {"status": "success", "message": "Result stored"}

@app.post("/work/upload/{task_id}")
async def upload_audio(task_id: str, worker_id: str, file: UploadFile = File(...)):
    """Worker uploads generated audio file."""
    # Verify task exists in active_tasks
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not active")
    
    if active_tasks[task_id]["worker_id"] != worker_id:
        raise HTTPException(status_code=403, detail="Worker ID mismatch")
    
    # Save file
    file_path = os.path.join(STORAGE_DIR, f"{task_id}.wav")
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Return URL that client will use
    audio_url = f"/audio/{task_id}.wav"
    return {"audio_url": audio_url}

@app.get("/audio/{task_id}.wav")
async def download_audio(task_id: str):
    """Client downloads the generated audio."""
    file_path = os.path.join(STORAGE_DIR, f"{task_id}.wav")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(file_path, media_type="audio/wav", filename=f"{task_id}.wav")


@app.get("/result/{task_id}", response_model=ResultResponse)
async def get_result(task_id: str, client_id: str):
    """Client polls for task result and updates heartbeat."""
    # Update last_seen for this task (client is alive)
    update_task_last_seen(task_id, client_id)
    
    # Check if completed
    if task_id in results:
        return ResultResponse(
            status="completed",
            audio_url=results[task_id]["audio_url"]
        )
    
    # Check if in active tasks
    if task_id in active_tasks:
        return ResultResponse(status="processing", audio_url=None)
    
    # Check if still in queue
    for task in task_queue:
        if task["id"] == task_id:
            return ResultResponse(status="pending", audio_url=None)
    
    # Not found anywhere
    return ResultResponse(status="not_found", audio_url=None)

@app.get("/stats")
async def get_stats():
    return {
        "queue_size": len(task_queue),
        "active_tasks": len(active_tasks),
        "completed_tasks": len(results),
    }

# ========== Timeout Cleanup Functions ==========
def cleanup_task_timeouts():
    """Reassign tasks that have been active too long (worker died)."""
    now = time.time()
    timed_out = []
    
    for task_id, task in active_tasks.items():
        if now - task["last_ping"] > TASK_TIMEOUT:
            timed_out.append(task)
    
    for task in timed_out:
        # Remove from active_tasks
        del active_tasks[task["id"]]
        # Re-add to front of queue with original data and preserve last_seen
        task_queue.appendleft({
            "id": task["id"],
            "text": task["text"],
            "speed": task["speed"],
            "voice_ref": task["voice_ref"],
            "client_id": task["client_id"],
            "last_seen": task["last_seen"],
            "submitted_at": task.get("claimed_at", time.time())
        })
        print(f"Task {task['id']} timed out (worker {task['worker_id']}), re-queued")

def cleanup_client_timeouts():
    """Remove tasks whose client hasn't polled recently."""
    now = time.time()
    
    # Clean tasks in queue: collect first, then remove
    to_remove = [task for task in task_queue if now - task["last_seen"] > CLIENT_TIMEOUT]
    for task in to_remove:
        task_queue.remove(task)
        print(f"Removed task {task['id']} from queue (client {task['client_id']} inactive)")
    
    # Clean active tasks (assigned but client abandoned)
    to_abort = [task_id for task_id, task in active_tasks.items() 
                if now - task["last_seen"] > CLIENT_TIMEOUT]
    for task_id in to_abort:
        del active_tasks[task_id]
        print(f"Aborted active task {task_id} (client abandoned)")

async def background_cleanup():
    """Background loop that runs cleanup periodically."""
    try:
        while True:
            await asyncio.sleep(CLEANUP_INTERVAL)
            cleanup_task_timeouts()
            cleanup_client_timeouts()
    except asyncio.CancelledError:
        # Task was cancelled during shutdown – exit gracefully
        print("Background cleanup task cancelled, shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)