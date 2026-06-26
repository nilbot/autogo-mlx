"""Test FastAPI endpoints for the autogo-mlx local play server."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from autogo_mlx.play_server import app, find_all_checkpoints


@pytest.fixture
def client():
    return TestClient(app)


def test_get_models(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    # Ensure at least one model is present (e.g. synthetic_overfit.safetensors)
    assert len(models) > 0
    assert any("synthetic_overfit.safetensors" in m for m in models)


def test_game_flow_black(client):
    # Find a model to use
    checkpoints = find_all_checkpoints()
    assert len(checkpoints) > 0
    model_name = checkpoints[0].name

    # 1. Start a new game as Black (Human plays first)
    new_game_payload = {
        "model_name": model_name,
        "board_size": 9,
        "color": "black",
        "n_simulations": 8
    }
    response = client.post("/api/new_game", json=new_game_payload)
    assert response.status_code == 200
    state = response.json()
    
    assert "game_id" in state
    assert state["size"] == 9
    assert state["human_color"] == 1  # Black
    assert state["to_play"] == 1  # Black's turn to play first
    assert state["is_over"] is False
    assert len(state["legal_moves"]) > 0
    
    game_id = state["game_id"]

    # 2. Human plays a valid move
    # First, pick a legal move coordinates (e.g., first available)
    legal_moves = state["legal_moves"]
    # Usually (3, 3) or similar is legal. Let's pick a valid one from list.
    move_to_play = legal_moves[0]
    row, col = move_to_play
    
    move_payload = {
        "row": row,
        "col": col,
        "pass_move": False
    }
    response = client.post(f"/api/game/{game_id}/move", json=move_payload)
    assert response.status_code == 200
    state_after_move = response.json()
    
    # After human plays Black, the bot should play White automatically in response.
    # So it should be Black's turn again (to_play == 1) unless game is over
    assert state_after_move["to_play"] == 1
    # Check that bot played a move
    assert "bot_move" in state_after_move
    assert "bot_analysis" in state_after_move
    
    # 3. Get MCTS assist analysis
    response = client.get(f"/api/game/{game_id}/assist")
    assert response.status_code == 200
    assist = response.json()
    assert "root_value" in assist
    assert "moves" in assist
    assert isinstance(assist["moves"], list)

    # 4. Human passes
    response = client.post(f"/api/game/{game_id}/pass")
    assert response.status_code == 200
    state_after_pass = response.json()
    assert state_after_pass["to_play"] == 1  # Still Black's turn after bot responds

    # 5. Undo moves
    # Undoing should undo human pass + bot response, bringing it back to state_after_move
    response = client.post(f"/api/game/{game_id}/undo")
    assert response.status_code == 200
    state_after_undo = response.json()
    # Check that the board state is restored (i.e. same as after the first human move + bot response)
    # The last move before pass was the bot's move in state_after_move
    assert state_after_undo["last_move"] == state_after_move["last_move"]


def test_game_flow_white(client):
    # Find a model to use
    checkpoints = find_all_checkpoints()
    assert len(checkpoints) > 0
    model_name = checkpoints[0].name

    # Start a new game as White (Bot plays first move)
    new_game_payload = {
        "model_name": model_name,
        "board_size": 9,
        "color": "white",
        "n_simulations": 8
    }
    response = client.post("/api/new_game", json=new_game_payload)
    assert response.status_code == 200
    state = response.json()
    
    assert "game_id" in state
    assert state["human_color"] == 2  # White
    assert state["to_play"] == 2  # White's turn (since bot already played its Black move)
    assert "bot_move" in state
    assert "bot_analysis" in state
    assert state["bot_move"] is not None  # Bot should have made a move


def test_frontend_assets(client):
    # Verify index HTML
    response = client.get("/")
    assert response.status_code == 200
    assert "Tactical Sandbox" in response.text

    # Verify style.css
    response = client.get("/web/style.css")
    assert response.status_code == 200
    assert ".board-wrapper" in response.text
    assert "aspect-ratio" in response.text

    # Verify app.js
    response = client.get("/web/app.js")
    assert response.status_code == 200
    assert "gameState" in response.text
