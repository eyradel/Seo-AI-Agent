from typing import List, Optional, Dict, Any
from enum import Enum
from uuid import uuid4
import os
import threading

from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, Header
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
import httpx
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# We reuse the analysis engine from the Streamlit app
# Note: importing app.py will import Streamlit; this is acceptable since it's in requirements
from seo_agent import SEOAgent


class AnalysisDepth(str, Enum):
    Basic = "Basic"
    Standard = "Standard"
    Deep = "Deep"


class Issue(BaseModel):
    type: str
    severity: str
    message: str


class PageResult(BaseModel):
    url: str
    title: str
    title_length: int
    description: str
    description_length: int
    text_sample: str
    word_count: int
    h1_tags: List[str]
    h2_tags: List[str]
    h3_tags: List[str]
    h1_count: int
    h2_count: int
    h3_count: int
    img_count: int
    img_missing_alt: int
    has_schema: bool
    has_viewport: bool
    load_time: float
    page_size: float
    score: int
    issues: List[Issue]
    ai_analysis: Optional[Dict[str, Any]] = None
    keyword_presence: Optional[Dict[str, Any]] = None


class Insights(BaseModel):
    overall_score: float
    grade: str
    avg_word_count: float
    avg_load_time: float
    top_issues: List[List[Any]] = Field(description="List of [issue_type, count]")
    keyword_presence: Optional[Dict[str, Any]] = None


class RecommendationItem(BaseModel):
    type: str
    message: str
    details: str
    affected_pages: Optional[List[str]] = None
    keyword: Optional[str] = None
    competitors: Optional[List[str]] = None


class Recommendations(BaseModel):
    critical: List[RecommendationItem] = []
    important: List[RecommendationItem] = []
    opportunity: List[RecommendationItem] = []


class ActionPlanItem(BaseModel):
    priority: str
    task: str
    details: str
    type: str


class AnalysisResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    url: Optional[str] = None
    pages: Optional[List[PageResult]] = None
    insights: Optional[Insights] = None
    recommendations: Optional[Recommendations] = None


class SummaryResponse(BaseModel):
    job_id: str
    summary: str


class ActionPlanResponse(BaseModel):
    job_id: str
    action_plan: List[ActionPlanItem]


class AnalyzeRequest(BaseModel):
    url: HttpUrl
    keywords: Optional[str] = Field(default="")
    max_pages: int = Field(default=5, ge=1, le=50)
    analysis_depth: AnalysisDepth = Field(default=AnalysisDepth.Standard)


class HealthResponse(BaseModel):
    status: str
    openai: bool
    google: bool


app = FastAPI(title="AI SEO Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
AUTH_BASE_URL = os.environ.get(
    "AUTH_BASE_URL",
    "https://services-backend-635062712814.europe-west1.run.app",
)

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


class AuthedUser(BaseModel):
    id: Optional[int] = None
    username: Optional[str] = None
    email: Optional[str] = None


_http_bearer = HTTPBearer(auto_error=False)


def _validate_bearer_token(token: str) -> AuthedUser:
    verify_url = f"{AUTH_BASE_URL}/auth/verify-token"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(verify_url, headers={"Authorization": f"Bearer {token}"})
            if resp.status_code != 200:
                raise HTTPException(status_code=401, detail="Invalid token", headers={"WWW-Authenticate": "Bearer"})
            # Optionally fetch user info
            me_url = f"{AUTH_BASE_URL}/auth/users/me"
            me_resp = client.get(me_url, headers={"Authorization": f"Bearer {token}"})
            if me_resp.status_code == 200:
                data = me_resp.json() or {}
                return AuthedUser(id=data.get("id"), username=data.get("username"), email=data.get("email"))
            return AuthedUser()
    except httpx.HTTPError:
        raise HTTPException(status_code=503, detail="Auth service unavailable")


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_http_bearer)) -> AuthedUser:
    if not credentials or credentials.scheme.lower() != "bearer" or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    return _validate_bearer_token(credentials.credentials)


def _run_analysis(job_id: str, req: AnalyzeRequest):
    agent = SEOAgent(openai_key=OPENAI_API_KEY, google_key=GOOGLE_API_KEY, gemini_key="")
    try:
        pages = agent.analyze_website(
            url=str(req.url),
            keywords=req.keywords or "",
            max_pages=req.max_pages,
            analysis_depth=req.analysis_depth.value,
        )
        # Ensure site-wide insights and recommendations are present
        insights = agent.insights
        recommendations = agent.recommendations
        with _jobs_lock:
            _jobs[job_id]["status"] = "complete"
            _jobs[job_id]["agent"] = agent
            _jobs[job_id]["results"] = {
                "url": str(req.url),
                "pages": pages,
                "insights": insights,
                "recommendations": recommendations,
            }
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)

@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": "seo-api"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        openai=bool(OPENAI_API_KEY),
        google=bool(GOOGLE_API_KEY),
    )


@app.get("/me", response_model=AuthedUser)
def me(user: AuthedUser = Depends(get_current_user)) -> AuthedUser:
    return user


@app.post("/analyze", response_model=AnalysisResponse)
def start_analysis(
    req: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    user: AuthedUser = Depends(get_current_user),
) -> AnalysisResponse:
    job_id = uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "owner": user.id or user.username or user.email}
    background_tasks.add_task(_run_analysis, job_id, req)
    return AnalysisResponse(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str, user: AuthedUser = Depends(get_current_user)) -> JobStatusResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    owner = job.get("owner")
    requester = user.id or user.username or user.email
    if owner and requester and owner != requester:
        raise HTTPException(status_code=403, detail="Forbidden")
    status = job.get("status", "unknown")
    if status != "complete":
        return JobStatusResponse(job_id=job_id, status=status)
    results = job.get("results", {})
    pages = [PageResult(**p) for p in results.get("pages", [])]
    insights = Insights(**results.get("insights", {})) if results.get("insights") else None
    recs = results.get("recommendations", {}) or {}
    recommendations = Recommendations(**recs)
    return JobStatusResponse(
        job_id=job_id,
        status=status,
        url=results.get("url"),
        pages=pages,
        insights=insights,
        recommendations=recommendations,
    )


@app.get("/summary/{job_id}", response_model=SummaryResponse)
def get_summary(job_id: str, user: AuthedUser = Depends(get_current_user)) -> SummaryResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    owner = job.get("owner")
    requester = user.id or user.username or user.email
    if owner and requester and owner != requester:
        raise HTTPException(status_code=403, detail="Forbidden")
    if job.get("status") != "complete":
        raise HTTPException(status_code=400, detail="Job is not complete")
    agent: SEOAgent = job.get("agent")
    summary = agent.generate_seo_summary()
    return SummaryResponse(job_id=job_id, summary=summary)


@app.get("/action-plan/{job_id}", response_model=ActionPlanResponse)
def get_action_plan(job_id: str, user: AuthedUser = Depends(get_current_user)) -> ActionPlanResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    owner = job.get("owner")
    requester = user.id or user.username or user.email
    if owner and requester and owner != requester:
        raise HTTPException(status_code=403, detail="Forbidden")
    if job.get("status") != "complete":
        raise HTTPException(status_code=400, detail="Job is not complete")
    agent: SEOAgent = job.get("agent")
    plan = agent.get_action_plan()
    items = [ActionPlanItem(**i) for i in plan]
    return ActionPlanResponse(job_id=job_id, action_plan=items)


# Uvicorn entrypoint
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
