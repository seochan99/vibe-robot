"""API regression tests for scene preset isolation."""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.server import app


def test_scene_init_resets_runtime_added_objects():
    client = TestClient(app)

    init = client.post("/api/scene/init", json={"scene": "wind_paper"})
    assert init.status_code == 200

    add = client.post(
        "/api/scene/add-object",
        json={"obj_type": "cube", "position": [0.8, -0.16, 0.35], "name": "cube"},
    )
    assert add.status_code == 200

    objs_after_add = client.get("/api/scene/objects").json().get("objects", [])
    assert any(o.get("name") == "cube" for o in objs_after_add)

    reinit = client.post("/api/scene/init", json={"scene": "wind_paper"})
    assert reinit.status_code == 200

    objs_after_reinit = client.get("/api/scene/objects").json().get("objects", [])
    assert all(o.get("name") != "cube" for o in objs_after_reinit)
