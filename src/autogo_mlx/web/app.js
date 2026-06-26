// Client State Registry
let gameState = {
    gameId: null,
    boardSize: 9,
    size: 9,
    humanColor: 1, // 1 = BLACK, 2 = WHITE
    toPlay: 1,     // 1 = BLACK, 2 = WHITE
    isOver: false,
    board: Array(9).fill(null).map(() => Array(9).fill(0)), // 2D Array initialized to empty
    lastMove: null,
    legalMoves: [],
    botAnalysis: null,
    isThinking: false
};

// DOM References
const modelSelect = document.getElementById("model-select");
const ensembleSelect = document.getElementById("ensemble-select");
const sizeSelect = document.getElementById("size-select");
const blackBtn = document.getElementById("color-black-btn");
const whiteBtn = document.getElementById("color-white-btn");
const simsSlider = document.getElementById("sims-slider");
const simsVal = document.getElementById("sims-val");
const teacherToggle = document.getElementById("teacher-toggle");
const newGameBtn = document.getElementById("new-game-btn");
const passBtn = document.getElementById("pass-btn");
const undoBtn = document.getElementById("undo-btn");
const statusText = document.getElementById("status-text");
const statusDot = document.getElementById("status-dot");
const turnDisplay = document.getElementById("turn-display");
const scoreDisplay = document.getElementById("score-display");
const evalFill = document.getElementById("eval-bar-fill");
const evalText = document.getElementById("eval-bar-text");
const boardContainer = document.getElementById("go-board-container");
const teacherPanel = document.getElementById("teacher-options-panel");
const passNotification = document.getElementById("pass-notification");

// Initialize application
document.addEventListener("DOMContentLoaded", () => {
    fetchModels();
    setupEventListeners();
    updateTeacherPanelVisibility();
    renderBoard();
});

// Event Listeners Registration
function setupEventListeners() {
    newGameBtn.addEventListener("click", initializeGame);
    passBtn.addEventListener("click", passTurn);
    undoBtn.addEventListener("click", undoMove);
    
    simsSlider.addEventListener("input", (e) => {
        simsVal.textContent = e.target.value;
    });
    
    blackBtn.addEventListener("click", () => setActiveColor("black"));
    whiteBtn.addEventListener("click", () => setActiveColor("white"));
    
    teacherToggle.addEventListener("change", () => {
        updateTeacherPanelVisibility();
        renderBoard(); // Rerender to toggle MCTS layer
    });
    
    document.querySelectorAll('input[name="teacher-mode"]').forEach(radio => {
        radio.addEventListener("change", renderBoard);
    });
}

function setActiveColor(color) {
    if (color === "black") {
        blackBtn.classList.add("active");
        whiteBtn.classList.remove("active");
        gameState.humanColor = 1;
    } else {
        whiteBtn.classList.add("active");
        blackBtn.classList.remove("active");
        gameState.humanColor = 2;
    }
}

function updateTeacherPanelVisibility() {
    if (teacherToggle.checked) {
        teacherPanel.classList.add("expanded");
    } else {
        teacherPanel.classList.remove("expanded");
    }
}

// Fetch available model checkpoints
async function fetchModels() {
    try {
        const res = await fetch("/api/models");
        const models = await res.json();
        
        modelSelect.innerHTML = "";
        if (models.length === 0) {
            modelSelect.innerHTML = `<option value="">No checkpoints found</option>`;
            return;
        }
        
        models.forEach(m => {
            const opt = document.createElement("option");
            opt.value = m;
            opt.textContent = m;
            // Default to iter11.safetensors or iter10 if available
            if (m === "iter11.safetensors" || m === "iter10.safetensors") {
                opt.selected = true;
            }
            modelSelect.appendChild(opt);
        });
    } catch (err) {
        console.error("Failed to fetch models:", err);
        statusText.textContent = "Error: Could not load model checkpoints.";
    }
}

// Initialize game on backend
async function initializeGame() {
    const model = modelSelect.value;
    if (!model) {
        alert("Please select a checkpoint first!");
        return;
    }
    
    setThinkingState(true, "Initializing match...");
    
    try {
        const res = await fetch("/api/new_game", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model_name: model,
                board_size: parseInt(sizeSelect.value),
                color: gameState.humanColor === 1 ? "black" : "white",
                n_simulations: parseInt(simsSlider.value),
                ensemble_mode: ensembleSelect.value
            })
        });
        
        const data = await res.json();
        updateGameState(data);
        setThinkingState(false, data.message || "Match started. Your turn!");
    } catch (err) {
        console.error("New game error:", err);
        setThinkingState(false, "Failed to start match.");
    }
}

// Handle playing moves
async function handleCellClick(row, col) {
    if (gameState.isOver || gameState.isThinking) return;
    
    // Check if it's the human's turn
    if (gameState.toPlay !== gameState.humanColor) {
        statusText.textContent = "It is not your turn yet.";
        return;
    }
    
    // Validate move locally
    const isLegal = gameState.legalMoves.some(m => m[0] === row && m[1] === col);
    if (!isLegal) {
        statusText.textContent = "Illegal move selected.";
        return;
    }
    
    setThinkingState(true, "Neural Network MCTS pondering...");
    
    try {
        const res = await fetch(`/api/game/${gameState.gameId}/move`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ row, col })
        });
        
        const data = await res.json();
        updateGameState(data);
        setThinkingState(false, data.message || "Your turn.");
    } catch (err) {
        console.error("Play move error:", err);
        setThinkingState(false, "Failed to play move.");
    }
}

async function passTurn() {
    if (!gameState.gameId || gameState.isOver || gameState.isThinking) return;
    
    setThinkingState(true, "Passing turn. AI thinking...");
    
    try {
        const res = await fetch(`/api/game/${gameState.gameId}/pass`, {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });
        
        const data = await res.json();
        updateGameState(data);
        setThinkingState(false, data.message || "Your turn.");
    } catch (err) {
        console.error("Pass error:", err);
        setThinkingState(false, "Failed to pass.");
    }
}

async function undoMove() {
    if (!gameState.gameId || gameState.isThinking) return;
    
    setThinkingState(true, "Rewinding game state...");
    
    try {
        const res = await fetch(`/api/game/${gameState.gameId}/undo`, {
            method: "POST"
        });
        
        const data = await res.json();
        updateGameState(data);
        setThinkingState(false, data.message || "Restored turn.");
    } catch (err) {
        console.error("Undo error:", err);
        setThinkingState(false, "Failed to undo.");
    }
}

// Request assist values (MCTS evaluation of current position)
async function requestAssistAnalysis() {
    if (!gameState.gameId || gameState.isOver || gameState.isThinking) return;
    
    try {
        const res = await fetch(`/api/game/${gameState.gameId}/assist`);
        const data = await res.json();
        gameState.botAnalysis = data;
        renderBoard();
        updateWinProbability(data.root_value);
    } catch (err) {
        console.error("Assist analysis error:", err);
    }
}

// Update local state and trigger UI updates
function updateGameState(data) {
    gameState.gameId = data.game_id;
    gameState.board = data.board;
    gameState.size = data.size;
    gameState.toPlay = data.to_play;
    gameState.lastMove = data.last_move;
    gameState.legalMoves = data.legal_moves;
    gameState.isOver = data.is_over;
    
    // Toggle pass notification overlay
    if (data.last_move_was_pass) {
        passNotification.classList.remove("hidden");
        const dismissPassBanner = () => {
            passNotification.classList.add("hidden");
            document.removeEventListener("mousedown", dismissPassBanner);
            document.removeEventListener("keydown", dismissPassBanner);
        };
        // Dismiss on any keyboard key or mouse click anywhere
        document.addEventListener("mousedown", dismissPassBanner);
        document.addEventListener("keydown", dismissPassBanner);
    } else {
        passNotification.classList.add("hidden");
    }
    
    if (data.bot_analysis) {
        gameState.botAnalysis = data.bot_analysis;
        updateWinProbability(data.bot_analysis.root_value);
    } else {
        // If no analysis returned, fetch it if Teacher Mode is on
        gameState.botAnalysis = null;
        if (teacherToggle.checked && !gameState.isOver) {
            requestAssistAnalysis();
        }
    }
    
    renderBoard();
    updateHUDDisplays(data);
}

function setThinkingState(isThinking, message) {
    gameState.isThinking = isThinking;
    statusText.textContent = message;
    
    if (isThinking) {
        statusDot.className = "status-indicator thinking";
    } else {
        statusDot.className = "status-indicator ready";
    }
}

function updateHUDDisplays(data) {
    turnDisplay.textContent = gameState.toPlay === 1 ? "BLACK" : "WHITE";
    
    if (gameState.isOver) {
        scoreDisplay.textContent = data.result || "Game Ended";
    } else {
        scoreDisplay.textContent = data.score > 0 ? `B+${data.score.toFixed(1)}` : `W+${Math.abs(data.score).toFixed(1)}`;
    }
}

function updateWinProbability(rootValue) {
    // root_value represents win probability from self-perspective (toPlay).
    // Let's normalize it to Black's perspective for the evaluation bar:
    // If toPlay = BLACK (1), B_prob = rootValue
    // If toPlay = WHITE (2), B_prob = 1.0 - rootValue
    let blackProb = gameState.toPlay === 1 ? rootValue : (1.0 - rootValue);
    let pct = (blackProb * 100).toFixed(1);
    
    evalFill.style.height = `${pct}%`;
    evalText.textContent = `${pct}%`;
}

// Render Board Grid Cells
function renderBoard() {
    if (!gameState.board || gameState.board.length === 0) return;
    
    const size = gameState.size;
    
    // Set grid columns/rows dynamically
    boardContainer.style.gridTemplateColumns = `repeat(${size}, 1fr)`;
    boardContainer.style.gridTemplateRows = `repeat(${size}, 1fr)`;
    boardContainer.innerHTML = "";
    
    // Calculate total visits to scale heatmap colors
    let maxVisits = 0;
    if (teacherToggle.checked && gameState.botAnalysis && gameState.botAnalysis.moves) {
        maxVisits = Math.max(...gameState.botAnalysis.moves.map(m => m.visits), 1);
    }

    for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
            const val = gameState.board[r][c];
            
            const cell = document.createElement("div");
            cell.className = "board-cell";
            
            // Add grid line classes for edge trimming
            if (r === 0) cell.classList.add("cell-top");
            if (r === size - 1) cell.classList.add("cell-bottom");
            if (c === 0) cell.classList.add("cell-left");
            if (c === size - 1) cell.classList.add("cell-right");
            
            // Add star points
            if (isStarPoint(size, r, c)) {
                const sp = document.createElement("div");
                sp.className = "board-star-point";
                cell.appendChild(sp);
            }
            
            // Render existing stones
            if (val !== 0) {
                cell.classList.add("occupied");
                const stone = document.createElement("div");
                stone.className = `go-stone ${val === 1 ? "stone-black" : "stone-white"}`;
                
                // Highlight last played move
                if (gameState.lastMove && gameState.lastMove[0] === r && gameState.lastMove[1] === c) {
                    stone.classList.add("last-played-move");
                }
                cell.appendChild(stone);
            } else {
                // Empty cell: setup ghost stone for legal hover
                const isLegal = gameState.legalMoves.some(m => m[0] === r && m[1] === c);
                if (isLegal && !gameState.isOver) {
                    const ghost = document.createElement("div");
                    ghost.className = `go-stone stone-ghost ${gameState.humanColor === 1 ? "stone-black" : "stone-white"}`;
                    ghost.style.display = "none";
                    cell.appendChild(ghost);
                    
                    cell.addEventListener("mouseenter", () => {
                        if (!gameState.isThinking) ghost.style.display = "block";
                    });
                    cell.addEventListener("mouseleave", () => { ghost.style.display = "none"; });
                    cell.addEventListener("click", () => handleCellClick(r, c));
                }
            }
            
            // Render Teacher Mode overlays
            if (teacherToggle.checked && val === 0 && gameState.botAnalysis && gameState.botAnalysis.moves) {
                const analysis = gameState.botAnalysis.moves.find(m => m.row === r && m.col === c);
                if (analysis) {
                    const activeRadio = document.querySelector('input[name="teacher-mode"]:checked');
                    const activeMode = activeRadio ? activeRadio.value : "priors";
                    
                    // Render heatmap styling based on selected teacher mode
                    if (activeMode === "visits" && analysis.visits > 0) {
                        const ratio = analysis.visits / maxVisits;
                        cell.style.backgroundColor = `rgba(240, 98, 51, ${ratio * 0.35})`;
                        cell.classList.add("visited-heatmap");
                    } else if (activeMode === "q" && analysis.visits > 0) {
                        // Normalize Q-value (-1..1 to 0..1) for emerald green heatmap
                        const qValNormalized = (analysis.q_value + 1.0) / 2.0;
                        cell.style.backgroundColor = `rgba(52, 199, 89, ${qValNormalized * 0.3})`;
                        cell.classList.add("visited-heatmap");
                    } else if (activeMode === "priors" && analysis.prior > 0.01) {
                        // Cool Blue heatmap for priors
                        cell.style.backgroundColor = `rgba(0, 122, 255, ${analysis.prior * 0.3})`;
                        cell.classList.add("visited-heatmap");
                    }
                    
                    const overlay = document.createElement("div");
                    overlay.className = "teacher-overlay";
                    
                    if (activeMode === "priors" && analysis.prior > 0.005) {
                        const p = document.createElement("span");
                        p.className = "teacher-prior";
                        p.textContent = `${(analysis.prior * 100).toFixed(0)}%`;
                        overlay.appendChild(p);
                    }
                    
                    if (activeMode === "visits" && analysis.visits > 0) {
                        const v = document.createElement("span");
                        v.className = "teacher-visits";
                        v.textContent = analysis.visits;
                        overlay.appendChild(v);
                    }
                    
                    if (activeMode === "q" && analysis.visits > 0) {
                        const q = document.createElement("span");
                        q.className = "teacher-qval";
                        q.textContent = analysis.q_value.toFixed(2);
                        overlay.appendChild(q);
                    }
                    
                    cell.appendChild(overlay);
                }
            }
            
            boardContainer.appendChild(cell);
        }
    }
}

// Star points coordinate check
function isStarPoint(size, r, c) {
    if (size === 9) {
        return (r === 2 || r === 6 || r === 4) && (c === 2 || c === 6 || c === 4);
    }
    // Fallback or other sizes if supported
    return false;
}
