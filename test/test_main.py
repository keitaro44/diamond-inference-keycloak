# -*- coding: utf-8 -*-
import pytest
from fastapi.testclient import TestClient
import os
import sys

# Добавляем src в путь для импорта модулей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ---------- Заглушки ----------
class DummyKeycloakOpenID:
    def well_known(self):
        # Возвращаем минимальный словарь, чтобы не было ошибок
        return {"token_endpoint": "https://dummy/token"}

    def has_uma_access(self, token, permission):
        # Создаём объект, имитирующий успешную авторизацию
        class AuthStatus:
            is_logged_in = True
            is_authorized = True
            missing_permissions = set()
        return AuthStatus()

def dummy_get_keycloak_data():
    """Возвращает заглушку вместо реального KeycloakOpenID."""
    return DummyKeycloakOpenID(), "https://dummy/token"

# ---------------------------------

@pytest.fixture
def init_test_client(monkeypatch) -> TestClient:
    # 1. Подменяем переменные окружения (чтобы не было ошибок при импорте)
    monkeypatch.setenv("MODEL_PATH", "faked/model.pkl")
    monkeypatch.setenv("KEYCLOAK_URL", "https://dummy")
    monkeypatch.setenv("CLIENT_ID", "dummy")
    monkeypatch.setenv("CLIENT_SECRET", "dummy")

    # 2. Подменяем функции model_utils
    def mock_make_inference(*args, **kwargs) -> dict[str, float]:
        return {"price": 1234.56}
    def mock_load_model(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr("model_utils.make_inference", mock_make_inference)
    monkeypatch.setattr("model_utils.load_model", mock_load_model)

    # 3. Подменяем get_keycloak_data в модуле keycloak_utils ДО импорта main
    import keycloak_utils
    monkeypatch.setattr(keycloak_utils, "get_keycloak_data", dummy_get_keycloak_data)

    # 4. Теперь импортируем main (здесь уже будет использована заглушка)
    from main import app
    import main

    # 5. Также подменяем сам объект keycloak_openid в main на всякий случай
    main.keycloak_openid = DummyKeycloakOpenID()

    return TestClient(app)


# ---------- Тесты ----------
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