# -*- coding: utf-8 -*-
import pytest
from fastapi.testclient import TestClient
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class DummyKeycloakOpenID:
    def well_known(self):
        return {"token_endpoint": "https://dummy/token"}

    def has_uma_access(self, token, permission):
        class AuthStatus:
            pass
        status = AuthStatus()
        # Для тестов: считаем валидным только токен "00000"
        if token == "00000":
            status.is_logged_in = True
            status.is_authorized = True
        else:
            status.is_logged_in = False
            status.is_authorized = False
        status.missing_permissions = set()
        return status

def dummy_get_keycloak_data():
    return DummyKeycloakOpenID(), "https://dummy/token"

# -------------------------------------------

@pytest.fixture
def init_test_client(monkeypatch) -> TestClient:
    monkeypatch.setenv("MODEL_PATH", "faked/model.pkl")
    monkeypatch.setenv("KEYCLOAK_URL", "https://dummy")
    monkeypatch.setenv("CLIENT_ID", "dummy")
    monkeypatch.setenv("CLIENT_SECRET", "dummy")

    def mock_make_inference(*args, **kwargs) -> dict[str, float]:
        return {"price": 1234.56}
    def mock_load_model(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr("model_utils.make_inference", mock_make_inference)
    monkeypatch.setattr("model_utils.load_model", mock_load_model)

    import keycloak_utils
    monkeypatch.setattr(keycloak_utils, "get_keycloak_data", dummy_get_keycloak_data)

    from main import app
    import main
    main.keycloak_openid = DummyKeycloakOpenID()

    return TestClient(app)


def test_healthcheck(init_test_client) -> None:
    response = init_test_client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_token_correctness(init_test_client) -> None:
    payload = {
        "carat": 0.23,
        "cut": "Ideal",
        "color": "E",
        "clarity": "SI2",
        "depth": 61.5,
        "table": 55.0,
        "x": 3.95,
        "y": 3.98,
        "z": 2.43
    }
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer 00000"},
        json=payload
    )
    assert response.status_code == 200
    assert "price" in response.json()
    assert response.json()["price"] == 1234.56


def test_token_not_correctness(init_test_client):
    payload = {
        "carat": 0.23,
        "cut": "Ideal",
        "color": "E",
        "clarity": "SI2",
        "depth": 61.5,
        "table": 55.0,
        "x": 3.95,
        "y": 3.98,
        "z": 2.43
    }
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer kedjkj"},
        json=payload
    )
    assert response.status_code == 401
    assert response.json() == {
        "detail": "Invalid authentication credentials"
    }


def test_token_absent(init_test_client):
    payload = {
        "carat": 0.23,
        "cut": "Ideal",
        "color": "E",
        "clarity": "SI2",
        "depth": 61.5,
        "table": 55.0,
        "x": 3.95,
        "y": 3.98,
        "z": 2.43
    }
    response = init_test_client.post(
        "/predictions",
        json=payload
    )
    assert response.status_code == 401
    assert response.json() == {
        "detail": "Not authenticated"
    }


def test_inference(init_test_client):
    payload = {
        "carat": 0.23,
        "cut": "Ideal",
        "color": "E",
        "clarity": "SI2",
        "depth": 61.5,
        "table": 55.0,
        "x": 3.95,
        "y": 3.98,
        "z": 2.43
    }
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer 00000"},
        json=payload
    )
    assert response.status_code == 200
    assert response.json()["price"] == 1234.56