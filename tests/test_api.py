import time

from fastapi.testclient import TestClient

from agentml.server.app import create_app


def test_full_api_flow(settings, llm, churn_df):
    client = TestClient(create_app(settings, llm))
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["provider"] == "mock" and h["knowledge_chunks"] > 0
    assert "AgentML" in client.get("/").text

    up = client.post("/api/datasets", files={"file": ("my churn.csv", churn_df.to_csv(index=False), "text/csv")})
    assert up.json()["name"] == "my_churn.csv"
    assert "my_churn.csv" in [d["name"] for d in client.get("/api/datasets").json()]
    assert client.get("/api/datasets/my_churn.csv/preview").json()["rows"]
    assert client.post("/api/datasets", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 400

    run_id = client.post("/api/automl/runs", json={"dataset": "my_churn.csv", "goal": "predict churn",
                                                   "budget": "low"}).json()["run_id"]
    for _ in range(240):
        r = client.get(f"/api/automl/runs/{run_id}").json()
        if r["status"] in {"completed", "failed"}:
            break
        time.sleep(0.5)
    assert r["status"] == "completed", r.get("error")
    assert r["result"]["best_model"] and r["n_events"] > 5
    assert "Executive summary" in client.get(f"/api/automl/runs/{run_id}/report").text
    pred = client.post(f"/api/automl/runs/{run_id}/predict", json={"rows": [{"tenure_months": 3}]}).json()
    assert pred["predictions"][0] in {"yes", "no"}
    assert client.get(f"/api/automl/runs/{run_id}/model").status_code == 200
    assert client.get("/api/automl/runs/../../etc").status_code == 404

    assert client.post("/api/rag/ingest", json={"text": "AgentML Studio was created by Prasanna in 2026.",
                                                "source": "about"}).json()["added_chunks"] == 1
    ans = client.post("/api/rag/query", json={"question": "Who created AgentML Studio?"}).json()
    assert "Prasanna" in ans["answer"]

    chat = client.post("/api/chat", json={"message": "calculate 7*6", "session_id": "t"}).json()
    assert chat["tool_calls"] == 1 and "42" in chat["answer"]
