import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from app.config import Settings, get_settings
from app.db import Database
from app.embedder import make_embedder
from app.index import Index, IndexLoadError
from app.llm import LLMError, OpenAICompatibleLLM
from app.metrics import Metrics
from app.logger import request_id_var, setup_logging
from app.rag import RagService
from app.retriever import HybridRetriever, RetrievalParams
from app.schemas import AskRequest, AskResponse, FeedbackRequest, Source
from app.security import RateLimiter, make_auth_dependency

log = logging.getLogger(__name__)

REQUEST_ID_OK = re.compile(r"^[\w\-]{1,64}$")
STATIC_DIR = Path(__file__).parent / "static"


def build_rag(s: Settings) -> RagService:
    """Load the embedding model and the index from disk and connect the LLM."""
    embedder = make_embedder(s)
    index = Index.load(s.index_dir, embedder.name, s.bm25_k1, s.bm25_b)
    log.info("index ready: %d chunks from %d books", len(index), len(index.meta.get("books", [])))
    retriever = HybridRetriever(index, embedder, RetrievalParams.from_settings(s))
    return RagService(retriever, OpenAICompatibleLLM(s), s.max_context_chars, s.min_dense_score)


def create_app(settings: Settings | None = None, rag: RagService | None = None, db: Database | None = None) -> FastAPI:
    """`rag` and `db` can be passed in (tests do this) instead of being loaded from disk."""
    s = settings or get_settings()
    setup_logging(s.log_level)
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rag = rag
        app.state.startup_error = None
        if rag is None:
            try:
                app.state.rag = build_rag(s)
            except IndexLoadError as exc:
                # keep running so /health works, but /ready and /v1/ask report the problem
                log.error("%s", exc)
                app.state.startup_error = str(exc)
        yield

    app = FastAPI(
        title="Namazi Hospital Medical RAG",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None if s.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if s.is_production else "/openapi.json",
    )
    app.state.rag = rag
    app.state.startup_error = None
    database = db or Database(s.db_path, s.log_questions)
    guard = make_auth_dependency(s.key_list, RateLimiter(s.rate_limit_per_minute))

    if s.origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=s.origin_list,
            allow_methods=["GET", "POST"],
            allow_headers=["x-api-key", "content-type"],
        )

    if not s.key_list:
        log.warning("API_KEYS is empty: authentication is OFF (ok for development only)")

    # ---- middleware and error handling -------------------------------------

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if REQUEST_ID_OK.match(incoming) else uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        request.state.started = started  # /v1/ask uses it, so waiting for a free worker counts too
        try:
            response = await call_next(request)
        finally:
            ms = int((time.perf_counter() - started) * 1000)
            log.info("%s %s took %dms", request.method, request.url.path, ms)
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError):
        log.error("LLM failure: %s", exc)
        metrics.add_error()
        return JSONResponse(status_code=503, content={"detail": "The language model is unavailable, try again shortly"})

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        log.exception("unhandled error")  # details go to the log, never to the client
        metrics.add_error()
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # ---- routes --------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def home():
        """The simple question page for the staff."""
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health():
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/metrics")
    def show_metrics():
        """Numbers for Prometheus and Grafana."""
        return PlainTextResponse(metrics.text())

    @app.get("/ready")
    def ready(request: Request):
        """Readiness: the index is loaded and questions can be answered."""
        rag_service = request.app.state.rag
        if rag_service is None:
            raise HTTPException(503, request.app.state.startup_error or "Index is not loaded")
        return {"status": "ready", "chunks": len(rag_service.retriever.index)}

    @app.post("/v1/ask", response_model=AskResponse, dependencies=[Depends(guard)])
    def ask(body: AskRequest, request: Request):
        rag_service = request.app.state.rag
        if rag_service is None:
            raise HTTPException(503, "The knowledge base is not loaded yet")

        started = time.perf_counter()
        result = rag_service.ask(body.question, body.top_k)
        latency_ms = int((time.perf_counter() - started) * 1000)
        waited = time.perf_counter() - request.state.started
        metrics.add_question(waited, result.answered, result.prompt_tokens, result.completion_tokens)

        sources = [
            Source(
                id=h.chunk.id,
                book=h.chunk.book,
                page=h.chunk.page,
                score=round(h.score, 5),
                snippet=h.chunk.text[:300],
            )
            for h in result.sources
        ]
        conversation_id = uuid.uuid4().hex
        try:
            database.save_conversation(
                conversation_id, body.question, result.answer, result.answered,
                [x.model_dump() for x in sources], latency_ms,
                result.prompt_tokens, result.completion_tokens,
            )
        except Exception:
            # losing a log row is better than failing a doctor's question
            log.exception("could not save conversation")

        log.info("conversation=%s answered=%s sources=%d latency=%dms",
                 conversation_id, result.answered, len(sources), latency_ms)
        return AskResponse(
            conversation_id=conversation_id,
            answer=result.answer,
            answered=result.answered,
            sources=sources,
            latency_ms=latency_ms,
        )

    @app.post("/v1/feedback", dependencies=[Depends(guard)])
    def feedback(body: FeedbackRequest):
        if not database.save_feedback(body.conversation_id, body.feedback, body.comment):
            raise HTTPException(404, "Unknown conversation_id")
        return {"status": "saved"}

    return app


def app_factory() -> FastAPI:
    """Entry point for gunicorn / uvicorn: `app.main:app_factory`"""
    return create_app()
