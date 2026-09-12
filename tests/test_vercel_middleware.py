import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.main import VercelRouteRewriteASGIMiddleware

def test_vercel_middleware_route_query():
    app = FastAPI()

    @app.get("/api/status")
    def status_route():
        return {"status": "ok"}
    
    @app.get("/api/other")
    def other_route():
        return {"status": "other"}

    app.add_middleware(VercelRouteRewriteASGIMiddleware)
    client = TestClient(app)

    # Simulate a request with __route__=status query param
    resp = client.get("/api/index.py?__route__=status")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    
    # Empty __route__
    resp = client.get("/api/index.py?__route__=")
    assert resp.status_code == 404

    # Double /api/ prefix (e.g. __route__=/api/status)
    resp = client.get("/api/index.py?__route__=/api/status")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

def test_vercel_middleware_matched_path_header():
    app = FastAPI()

    @app.get("/api/status")
    def status_route():
        return {"status": "ok"}

    app.add_middleware(VercelRouteRewriteASGIMiddleware)
    client = TestClient(app)

    # Simulate a request with x-matched-path header
    resp = client.get("/api/index", headers={"x-matched-path": "/api/status"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

def test_vercel_middleware_normal_request():
    app = FastAPI()

    @app.get("/api/normal")
    def normal_route():
        return {"status": "normal"}

    app.add_middleware(VercelRouteRewriteASGIMiddleware)
    client = TestClient(app)

    resp = client.get("/api/normal")
    assert resp.status_code == 200
    assert resp.json() == {"status": "normal"}
