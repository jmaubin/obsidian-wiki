"""HTTP + MCP front end for a vault, so remote agents can use it as memory.

Single tenant: one process, one vault, one API key. The container does no LLM
work — search, packing and frontmatter parsing all delegate to the existing
`graphrag` / `context_pack` modules, and the caller's agent does the thinking.

Needs the optional extra: ``pip install 'obsidian-wiki[server]'``.
Run it with ``python -m obsidian_wiki.server``.
"""

from __future__ import annotations

import difflib
import functools
import hmac
import ipaddress
import json
import os
import re
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from obsidian_wiki.cache import _iter_entries
from obsidian_wiki.context_pack import ContextError, build_context_pack
from obsidian_wiki.graphrag import query as graph_query
from obsidian_wiki.staging import StagingError, list_staged, resolve_in_vault
from obsidian_wiki.lint import lint_vault
from obsidian_wiki.sync import _git
from obsidian_wiki.vault import iter_md

VAULT = Path(os.environ.get("OBSIDIAN_VAULT_PATH", "/vault")).expanduser()
API_KEY = os.environ.get("WIKI_API_KEY", "")
ANONYMOUS = os.environ.get("WIKI_ALLOW_ANONYMOUS") == "1"

if not API_KEY and not ANONYMOUS:
    raise RuntimeError(
        "refusing to start without WIKI_API_KEY. "
        "Set it, or set WIKI_ALLOW_ANONYMOUS=1 for local development."
    )


# --- vault operations -------------------------------------------------------
# Every route and every MCP tool goes through these four functions, so the
# path check below is the single trust boundary for the whole service.

def _resolve(rel: str) -> Path:
    """Resolve a caller-supplied path inside the vault, or refuse."""
    try:
        return resolve_in_vault(VAULT, rel)
    except StagingError as exc:
        raise HTTPException(400, str(exc)) from exc


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "untitled"


def _category_slug(category: str) -> str:
    """Slugify a category, preserving a reserved leading underscore."""
    slug = _slug(category)
    return f"_{slug}" if category.lower().startswith("_") else slug


def search(q: str, limit: int = 8) -> dict[str, Any]:
    return graph_query(VAULT, q, top_n=limit)


def read_page(path: str) -> dict[str, Any]:
    target = _resolve(path)
    if not target.is_file():
        raise HTTPException(404, f"no such page: {path}")
    return {"path": path, "markdown": target.read_text(encoding="utf-8")}


def _folded(key: str, value: str) -> str:
    """Emit a free-text scalar as a YAML folded block (`key: >-`).

    A bare scalar containing ": ", "#", or a quote breaks YAML parsing, and
    Obsidian then reports "Invalid properties". A folded block needs no
    escaping, so it stays readable on disk for any value — including
    non-ASCII — and the vault's own frontmatter readers already understand it.
    """
    return f"{key}: >-\n  " + " ".join(str(value).split())


_MANAGED_KEYS = ("title", "category", "tags", "sources", "summary", "created", "updated")
_FRONTMATTER_KEY_RE = re.compile(r"^([A-Za-z_][\w-]*):")


def _frontmatter_chunks(text: str) -> dict[str, str]:
    """Top-level frontmatter keys -> their raw lines, multi-line values included.

    Raw text rather than parsed values, so a field this writer does not manage
    (`type`, `date`, `lifecycle`, a nested `provenance:` block…) is written
    back byte for byte instead of being re-serialized or dropped.
    """
    if not text.startswith("---\n"):
        return {}
    head, sep, _ = text[4:].partition("\n---\n")
    if not sep:
        return {}
    chunks: dict[str, str] = {}
    key = None
    for line in head.splitlines():
        match = _FRONTMATTER_KEY_RE.match(line)
        if match:
            key = match.group(1)
            chunks[key] = line
        elif key is not None:
            chunks[key] += "\n" + line
    return chunks


def _chunk_value(chunk: str) -> str:
    return chunk.split(":", 1)[1].strip()


def write_page(
    title: str,
    content: str,
    category: str = "concepts",
    *,
    tags: list[str] | None = None,
    sources: list[str] | None = None,
    summary: str | None = None,
    upsert: bool = True,
    path: str = "",
) -> dict[str, Any]:
    """Create a page from title + category, or update the existing page at `path`.

    On update, fields this writer does not manage are kept as they are, and
    omitted `tags` / `sources` / `summary` keep their current values.
    """
    if path:
        rel = path.strip().lstrip("/")
        if not rel.endswith(".md"):
            raise HTTPException(400, f"path must name a .md page: {rel}")
        target = _resolve(rel)
        if not target.is_file():
            raise HTTPException(404, f"no page at {rel}; omit path to create a new page")
    else:
        rel = f"{_category_slug(category)}/{_slug(title)}.md"
        target = _resolve(rel)
        if target.exists() and not upsert:
            raise HTTPException(409, f"page already exists: {rel}")
    existing = _frontmatter_chunks(target.read_text(encoding="utf-8")) if target.exists() else {}
    today = date.today().isoformat()
    # Preserve the original created: date across updates.
    created = _chunk_value(existing["created"]) if "created" in existing else today
    if path:
        category_line = existing.get("category") or f"category: {rel.split('/')[0]}"
    else:
        category_line = f"category: {_category_slug(category)}"

    def managed(key: str, given: Any, render: str) -> str:
        return existing[key] if given is None and key in existing else render

    front = "\n".join(
        [
            "---",
            _folded("title", title),
            category_line,
            managed("tags", tags, "tags: [" + ", ".join(tags or []) + "]"),
            managed("sources", sources, "sources: [" + ", ".join(sources or []) + "]"),
            managed("summary", summary, _folded("summary", summary) if summary else "summary:"),
            *(chunk for key, chunk in existing.items() if key not in _MANAGED_KEYS),
            f"created: {created}",
            f"updated: {today}",
            "---",
            "",
            "",
        ]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(front + content.rstrip() + "\n", encoding="utf-8")
    # Through the shared writer, so an API write is locked and parseable like
    # every other one. Previously this appended its own ad-hoc format.
    from obsidian_wiki import memory as mem

    mem.append_log(VAULT, "API_WRITE", {"page": rel, "title": title})
    return {"path": rel, "created": created, "updated": today}


def context_pack(
    topic: str,
    *,
    budget: int = 8000,
    recent: bool = False,
    public_only: bool = False,
    metadata_only: bool = False,
) -> dict[str, Any]:
    try:
        return build_context_pack(
            VAULT, topic, budget=budget, recent=recent,
            public_only=public_only, metadata_only=metadata_only,
        )
    except ContextError as exc:
        raise HTTPException(400, str(exc)) from exc


# --- HTTP -------------------------------------------------------------------

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _is_loopback(client_host: str | None) -> bool:
    try:
        return ipaddress.ip_address(client_host or "").is_loopback
    except ValueError:
        return False


def _host_name(host_header: str) -> str:
    """`localhost:8080` -> `localhost`, `[::1]:8080` -> `::1`."""
    if host_header.startswith("["):
        return host_header[1:].partition("]")[0]
    return host_header.rpartition(":")[0] if host_header.count(":") == 1 else host_header


def _authorized(request: Request) -> bool:
    """A valid key from anywhere, or — with WIKI_ALLOW_ANONYMOUS=1 — no key from
    this machine only.

    Anonymous access also requires a loopback Host header (a DNS-rebinding page
    in the user's browser connects from loopback but sends its own hostname) and
    no forwarding headers (a reverse proxy on this machine makes every remote
    client look like loopback).
    """
    header = request.headers.get("authorization", "")
    token = header[7:] if header.lower().startswith("bearer ") else ""
    if API_KEY and hmac.compare_digest(token, API_KEY):
        return True
    return (
        ANONYMOUS
        and _is_loopback(request.client.host if request.client else None)
        and _host_name(request.headers.get("host", "")) in _LOOPBACK_HOSTS
        and "x-forwarded-for" not in request.headers
        and "forwarded" not in request.headers
    )


def require_key(request: Request) -> None:
    if not _authorized(request):
        raise HTTPException(401, "missing or invalid API key")


class _RequireKey:
    """ASGI guard for the mounted MCP app: FastAPI dependencies never reach a mount."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http" and not _authorized(Request(scope)):
            response = JSONResponse({"detail": "missing or invalid API key"}, status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
            await response(scope, receive, send)
            return
        await self.inner(scope, receive, send)


class PageWrite(BaseModel):
    title: str
    category: str = "concepts"
    content: str
    # None keeps the current value when updating an existing page.
    tags: list[str] | None = None
    sources: list[str] | None = None
    summary: str | None = None
    upsert: bool = True
    path: str = ""


class PackRequest(BaseModel):
    topic: str = ""
    budget: int = 8000
    recent: bool = False
    public_only: bool = False
    metadata_only: bool = False


class ProfileWrite(BaseModel):
    action: str = "set"
    key: str = ""
    value: str = ""
    confidence: float = 0.6
    source: str = "agent"
    user_id: str = ""


class TodoWrite(BaseModel):
    action: str = "add"
    text: str = ""
    todo_id: str = ""
    origin: str = ""
    user_id: str = ""


class SyncRequest(BaseModel):
    verb: str = "API_WRITE"
    takeaways: str = ""
    fields: dict[str, str] | None = None


def _scope(user_id: str) -> str:
    """Validate a caller-supplied user_id before it becomes a filename."""
    from obsidian_wiki import memory as mem

    try:
        return mem.check_scope(user_id)
    except mem.MemoryError_ as exc:
        raise HTTPException(400, str(exc))


def memory_recap(project: str = "", max_words: int = 350, min_confidence: float = 0.0,
                 user_id: str = "") -> dict[str, Any]:
    """The owner profile, open threads, and recent activity as one block."""
    from obsidian_wiki import memory as mem

    return {
        "recap": mem.build_recap(
            VAULT, max_words=max_words, min_confidence=min_confidence,
            project=project or None, scope=_scope(user_id),
        )
    }


def memory_profile(action: str = "list", key: str = "", value: str = "",
                   confidence: float = 0.6, source: str = "agent",
                   user_id: str = "") -> dict[str, Any]:
    """Read or update durable facts about a person. `user_id` scopes them."""
    from obsidian_wiki import memory as mem

    scope = _scope(user_id)
    if action == "list":
        return {"facts": [fact.__dict__ for fact in mem.load_profile(VAULT, scope)]}
    if action == "set":
        if not key or not value:
            raise HTTPException(400, "key and value are required for action='set'")
        return {"fact": mem.set_fact(
            VAULT, key, value, confidence=confidence, source=source, scope=scope
        ).__dict__}
    if action == "forget":
        return {"removed": mem.forget_fact(VAULT, key, scope=scope)}
    raise HTTPException(400, f"action must be list, set, or forget; got {action!r}")


def memory_todo(action: str = "list", text: str = "", todo_id: str = "",
                origin: str = "", include_closed: bool = False,
                user_id: str = "") -> dict[str, Any]:
    """Read or update threads carried between sessions. `user_id` scopes them."""
    from obsidian_wiki import memory as mem

    scope = _scope(user_id)
    if action == "list":
        todos = mem.load_todos(VAULT, scope)
        if not include_closed:
            todos = [todo for todo in todos if todo.status == "open"]
        return {"todos": [dict(todo.__dict__, stale=todo.is_stale()) for todo in todos]}
    if action == "add":
        if not text:
            raise HTTPException(400, "text is required for action='add'")
        return {"todo": mem.add_todo(VAULT, text, origin=origin, scope=scope).__dict__}
    if action in ("done", "drop"):
        if not todo_id:
            raise HTTPException(400, f"todo_id is required for action={action!r}")
        status = "done" if action == "done" else "dropped"
        return {"todo": mem.set_todo_status(VAULT, todo_id, status, scope=scope).__dict__}
    raise HTTPException(400, f"action must be list, add, done, or drop; got {action!r}")


def memory_sync(verb: str = "API_WRITE", takeaways: str = "", **fields: str) -> dict[str, Any]:
    """Reconcile the index and hot cache after writes, under one lock."""
    from obsidian_wiki import memory as mem

    try:
        with mem.memory_lock(VAULT):
            line = mem.append_log(VAULT, verb, fields, lock=False)
            index = mem.rebuild_index(VAULT, lock=False)
            hot = mem.rebuild_hot(VAULT, lock=False, takeaways=takeaways or None)
    except mem.MemoryError_ as exc:
        if exc.code == "unmigrated":
            raise HTTPException(409, str(exc))
        raise HTTPException(400, str(exc))
    return {
        "log_line": line,
        "index": {"total": index.total, "added": list(index.added)},
        "hot": {"words": hot.words},
    }


def _tool(fn: Any) -> Any:
    """Expose an HTTP handler as an MCP tool, keeping its refusals readable.

    The SDK shows the agent only "Error executing tool <name>" for an
    unexpected exception; a `ToolError` keeps its message. The handlers raise
    HTTPException for expected refusals (missing page, bad path, conflict),
    so those become ToolErrors and the agent can correct its call.
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except HTTPException as exc:
            raise ToolError(f"{exc.status_code}: {exc.detail}") from exc
    return wrapper


mcp = MCPServer("obsidian-wiki")
mcp.tool(name="memory_search", description="Search the wiki. Returns ranked pages with summaries.")(_tool(search))
mcp.tool(name="memory_read", description="Read one wiki page as markdown, by vault-relative path.")(_tool(read_page))
mcp.tool(
    name="memory_write",
    description=(
        "Write a wiki page. To change an existing page, pass its `path` (as returned by "
        "memory_search or memory_read): the file is updated in place, its other frontmatter "
        "fields are kept, and omitted tags/sources/summary keep their values. Without `path`, "
        "a page is created at <category>/<slug of title>.md. Use category '_raw' for a rough capture."
    ),
)(_tool(write_page))
mcp.tool(name="memory_context_pack", description="Compile a token-bounded context pack on a topic.")(_tool(context_pack))
mcp.tool(name="memory_recap", description="Owner profile, open threads, and recent activity — call at session start.")(_tool(memory_recap))
mcp.tool(name="memory_profile", description="Read or update durable facts about a person (list|set|forget); user_id scopes them.")(_tool(memory_profile))
mcp.tool(name="memory_todo", description="Read or update threads carried between sessions (list|add|done|drop); user_id scopes them.")(_tool(memory_todo))
mcp.tool(name="memory_sync", description="Reconcile index.md and hot.md after writes, under one lock.")(_tool(memory_sync))


# Must be built before `mcp.session_manager` exists. Mounted at /mcp below, so
# its own path is "/". Stateless: no server-side session state to lose on restart.
# The SDK's own Host check defaults to localhost-only, which answers 421 to every
# remote agent; _RequireKey does that check instead, for anonymous requests only.
_mcp_app = mcp.streamable_http_app(
    streamable_http_path="/",
    stateless_http=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Without running the session manager here, /mcp accepts the first request
    # and then hangs.
    async with mcp.session_manager.run():
        yield


app = FastAPI(title="obsidian-wiki memory", lifespan=lifespan)
app.mount("/mcp", _RequireKey(_mcp_app))


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": VAULT.is_dir(), "vault": str(VAULT)}


@app.get("/v1/search", dependencies=[Depends(require_key)])
def http_search(q: str, limit: int = 8) -> dict[str, Any]:
    return search(q, limit)


@app.get("/v1/pages/{path:path}", dependencies=[Depends(require_key)])
def http_read(path: str) -> dict[str, Any]:
    return read_page(path)


@app.post("/v1/pages", dependencies=[Depends(require_key)])
def http_write(body: PageWrite) -> dict[str, Any]:
    return write_page(
        body.title, body.content, body.category,
        tags=body.tags, sources=body.sources, summary=body.summary,
        upsert=body.upsert, path=body.path,
    )


@app.post("/v1/context-pack", dependencies=[Depends(require_key)])
def http_pack(body: PackRequest) -> dict[str, Any]:
    return context_pack(
        body.topic, budget=body.budget, recent=body.recent,
        public_only=body.public_only, metadata_only=body.metadata_only,
    )


# --- operations console -----------------------------------------------------
# Read-only. The vault stays the source of truth: every number below is derived
# from the files on disk, from `.manifest.json`, or from git — no state of our
# own. Reuses lint_vault / run_doctor rather than shelling out to the CLI.

# `.manifest.json` is written by several skills and is not one shape: wiki-ingest
# keys sources under "sources", wiki-update under "projects", wiki-research under
# "research_sessions", and each names its page list differently. Read every known
# shape — a real vault usually holds more than one.
_MANIFEST_CONTAINERS = ("sources", "projects", "research_sessions")
_MANIFEST_PAGE_KEYS = ("pages_produced", "pages_created", "pages_in_vault")


def _count_md(rel: str) -> int:
    directory = VAULT / rel
    return len(list(directory.rglob("*.md"))) if directory.is_dir() else 0


def _git_status() -> dict[str, Any]:
    if not (VAULT / ".git").is_dir():
        return {"repo": False}
    porcelain = _git(VAULT, "status", "--porcelain")
    branch = _git(VAULT, "rev-parse", "--abbrev-ref", "HEAD")
    log = _git(VAULT, "log", "-5", "--pretty=%h %s")
    dirty = [line for line in porcelain.stdout.splitlines() if line.strip()]
    return {
        "repo": True,
        "branch": branch.stdout.strip(),
        "dirty": dirty[:50],
        "dirty_count": len(dirty),
        "recent": log.stdout.splitlines(),
    }


def status() -> dict[str, Any]:
    """Operational snapshot: page counts, staging/raw depth, source freshness."""
    pages = [p for p in iter_md(VAULT) if not p.relative_to(VAULT).parts[0].startswith("_")]
    categories: dict[str, int] = {}
    for page in pages:
        parts = page.relative_to(VAULT).parts
        # index.md / log.md / hot.md sit at the root and belong to no category.
        top = parts[0] if len(parts) > 1 else "(root)"
        categories[top] = categories.get(top, 0) + 1
    recent = sorted(pages, key=lambda p: p.stat().st_mtime, reverse=True)[:20]

    manifest_path = VAULT / ".manifest.json"
    sources: list[dict[str, Any]] = []
    last_ingest = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(500, f"unreadable .manifest.json: {exc}") from exc
        last_ingest = manifest.get("last_ingest")
        for container in _MANIFEST_CONTAINERS:
            for key, entry in _iter_entries(manifest.get(container)):
                produced = [
                    rel for field in _MANIFEST_PAGE_KEYS for rel in entry.get(field, [])
                ]
                sources.append({
                    "source_id": entry.get("source_id") or entry.get("session_id") or key or "",
                    "container": container,
                    "type": entry.get("type") or entry.get("skill") or "",
                    "ingested_at": (
                        entry.get("ingested_at")
                        or entry.get("last_synced")
                        or entry.get("completed")
                        or ""
                    ),
                    "pages_produced": len(produced),
                    # A produced page that no longer exists means the source needs re-ingesting.
                    "missing_pages": [rel for rel in produced if not (VAULT / rel).is_file()],
                })

    return {
        "vault": str(VAULT),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pages": len(pages),
        "categories": dict(sorted(categories.items())),
        "raw_count": _count_md("_raw"),
        "staging_count": _count_md("_staging"),
        "last_ingest": last_ingest,
        "sources": sources,
        "recent_pages": [
            {
                "path": str(p.relative_to(VAULT)),
                "modified": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
            }
            for p in recent
        ],
        "git": _git_status(),
    }


def vault_health() -> dict[str, Any]:
    """Doctor + lint, straight from the same functions the CLI calls."""
    from obsidian_wiki.cli import run_doctor

    return {"doctor": run_doctor(vault_override=str(VAULT)), "lint": lint_vault(VAULT)}


def staging() -> dict[str, Any]:
    """Staged pages, each with a unified diff against the live page (if any).

    difflib emits plain text and the UI inserts it as textContent, so
    agent-authored markdown never reaches the browser as HTML.
    """
    def lines(path: Path) -> list[str]:
        if not path.is_file():
            return []
        return path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)

    items = []
    for entry in list_staged(VAULT):
        item = entry.as_dict()
        item["diff"] = "".join(
            difflib.unified_diff(
                lines(VAULT / entry.live_path),
                lines(VAULT / entry.staged_path),
                "live",
                "staged",
            )
        )
        items.append(item)
    return {"count": len(items), "items": items}


_CONSOLE = Path(__file__).with_name("console.html")


@app.get("/v1/memory/recap", dependencies=[Depends(require_key)])
def http_recap(project: str = "", max_words: int = 350, min_confidence: float = 0.0,
               user_id: str = "") -> dict[str, Any]:
    return memory_recap(project=project, max_words=max_words,
                        min_confidence=min_confidence, user_id=user_id)


@app.get("/v1/memory/profile", dependencies=[Depends(require_key)])
def http_profile_list(user_id: str = "") -> dict[str, Any]:
    return memory_profile("list", user_id=user_id)


@app.post("/v1/memory/profile", dependencies=[Depends(require_key)])
def http_profile_set(body: ProfileWrite) -> dict[str, Any]:
    return memory_profile(
        body.action, key=body.key, value=body.value,
        confidence=body.confidence, source=body.source, user_id=body.user_id,
    )


@app.get("/v1/memory/todos", dependencies=[Depends(require_key)])
def http_todo_list(include_closed: bool = False, user_id: str = "") -> dict[str, Any]:
    return memory_todo("list", include_closed=include_closed, user_id=user_id)


@app.post("/v1/memory/todos", dependencies=[Depends(require_key)])
def http_todo_write(body: TodoWrite) -> dict[str, Any]:
    return memory_todo(body.action, text=body.text, todo_id=body.todo_id,
                       origin=body.origin, user_id=body.user_id)


@app.post("/v1/memory/sync", dependencies=[Depends(require_key)])
def http_memory_sync(body: SyncRequest) -> dict[str, Any]:
    return memory_sync(verb=body.verb, takeaways=body.takeaways, **(body.fields or {}))


@app.get("/ui", response_class=HTMLResponse)
def http_ui() -> str:
    """The console shell. Carries no vault data — every number is fetched with the key."""
    return _CONSOLE.read_text(encoding="utf-8")


@app.get("/v1/status", dependencies=[Depends(require_key)])
def http_status() -> dict[str, Any]:
    return status()


@app.get("/v1/health", dependencies=[Depends(require_key)])
def http_vault_health() -> dict[str, Any]:
    return vault_health()


@app.get("/v1/staging", dependencies=[Depends(require_key)])
def http_staging() -> dict[str, Any]:
    return staging()


def main() -> None:
    import uvicorn

    # Loopback by default: /ui is a browser console over the whole vault, and the
    # bare `python -m obsidian_wiki.server` case is a laptop or a dev box, not a
    # deliberate exposure. Containers need every interface, so the Dockerfile sets
    # WIKI_HOST=0.0.0.0 explicitly.
    host = os.environ.get("WIKI_HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=int(os.environ.get("WIKI_PORT", "8080")))


if __name__ == "__main__":
    main()
