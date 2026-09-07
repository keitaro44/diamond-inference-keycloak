# -*- coding: utf-8 -*-
import pytest
from fastapi.testclient import TestClient

# Импортируем модуль, но до этого подменим переменные окружения
# и функцию get_keycloak_data, чтобы она не пыталась соединиться с Keycloak
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def init_test_client(monkeypatch) -> TestClient:
    # 1. Подменяем переменные окружения (чтобы не было ошибки при импорте)
    monkeypatch.setenv("MODEL_PATH", "faked/model.pkl")
    monkeypatch.setenv("KEYCLOAK_URL", "https://dummy")
    monkeypatch.setenv("CLIENT_ID", "dummy")
    monkeypatch.setenv("CLIENT_SECRET", "dummy")

    # 2. Моки для функций сервиса
    def mock_make_inference(*args, **kwargs) -> dict[str, float]:
        return {"price": 1234.56}

    def mock_load_model(*args, **kwargs) -> None:
        return None

    # 3. Мок для get_keycloak_data – возвращает заглушку
    def mock_get_keycloak_data(*args, **kwargs):
        from keycloak import KeycloakOpenID
        openid = KeycloakOpenID(server_url="https://dummy", client_id="dummy", realm_name="inference", client_secret_key="dummy")
        return openid, "https://dummy/token"

    monkeypatch.setattr("model_utils.make_inference", mock_make_inference)
    monkeypatch.setattr("model_utils.load_model", mock_load_model)
    monkeypatch.setattr("keycloak_utils.get_keycloak_data", mock_get_keycloak_data)

    # 4. Теперь можно импортировать main без ошибок
    from main import app
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
