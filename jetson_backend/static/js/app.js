'use strict';

// ── API Client ──────────────────────────────────────────────

const api = {
    async get(path) {
        const res = await fetch(path);
        if (!res.ok) throw new Error(`GET ${path}: ${res.status}`);
        return res.json();
    },

    async post(path, body) {
        const res = await fetch(path, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!res.ok) throw new Error(`POST ${path}: ${res.status}`);
        return res.json();
    },
};

// ── DOM References ──────────────────────────────────────────

const $ = (id) => document.getElementById(id);

const dom = {
    jetsonDot:       $('jetson-dot'),
    piDot:           $('pi-dot'),
    modeDescription: $('mode-description'),
    featuresGrid:    $('features-grid'),
    motorSpeed:      $('motor-speed'),
    voiceInput:      $('voice-input'),
    voiceSend:       $('voice-send'),
    log:             $('log'),
};

// ── State ───────────────────────────────────────────────────

let currentMode = null;

// ── Helpers ─────────────────────────────────────────────────

function escapeHtml(text) {
    const el = document.createElement('span');
    el.textContent = text;
    return el.innerHTML;
}

// ── Logging ─────────────────────────────────────────────────

function addLog(message, type = 'info') {
    const time = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `<span class="log-time">${time}</span>${escapeHtml(message)}`;
    dom.log.prepend(entry);

    while (dom.log.children.length > 50) {
        dom.log.lastChild.remove();
    }
}

// ── Status ──────────────────────────────────────────────────

function setDot(el, connected) {
    el.className = `dot ${connected ? 'connected' : 'disconnected'}`;
}

async function refreshState() {
    try {
        const { state } = await api.get('/state');
        setDot(dom.jetsonDot, true);
        setDot(dom.piDot, state.pi_connected);

        if (state.mode !== currentMode) {
            applyMode(state.mode, state.mode_description, state.mode_config);
        }
    } catch {
        setDot(dom.jetsonDot, false);
        setDot(dom.piDot, false);
    }
}

// ── Mode ────────────────────────────────────────────────────

function applyMode(mode, description, config) {
    currentMode = mode;

    document.querySelectorAll('.mode-card').forEach(card =>
        card.classList.toggle('active', card.dataset.mode === mode)
    );

    dom.modeDescription.textContent = description;
    dom.motorSpeed.textContent = `Speed: ${config.motor_speed}`;
    renderFeatures(config.features || []);
}

async function selectMode(mode) {
    try {
        addLog(`Switching to ${mode} mode…`);
        const data = await api.post('/mode', { mode });
        applyMode(mode, data.backend.description, data.backend.config);
        setDot(dom.piDot, data.raspberry_pi?.success ?? false);
        addLog(`Mode set to ${mode}`, 'success');
    } catch (err) {
        addLog(`Mode switch failed: ${err.message}`, 'error');
    }
}

// ── Features ────────────────────────────────────────────────

function renderFeatures(features) {
    dom.featuresGrid.innerHTML = '';

    features.forEach(f => {
        const btn = document.createElement('button');
        btn.className = 'feature-btn';
        btn.textContent = f.name;
        btn.addEventListener('click', () => executeAction(f.action, f.name));
        dom.featuresGrid.appendChild(btn);
    });
}

// ── Actions ─────────────────────────────────────────────────

async function executeAction(action, label) {
    const name = label || action;
    try {
        addLog(`Executing: ${name}…`);
        const data = await api.post('/robot/action', { action });

        if (data.status === 'blocked') {
            addLog(`Blocked: ${data.message}`, 'error');
        } else {
            addLog(`${name} — ${data.status}`, 'success');
        }
    } catch (err) {
        addLog(`Action failed: ${err.message}`, 'error');
    }
}

// ── Voice Commands ──────────────────────────────────────────

async function sendVoiceCommand() {
    const command = dom.voiceInput.value.trim();
    if (!command) return;

    try {
        addLog(`Voice: "${command}"`);
        const data = await api.post('/voice-command', { command });

        if (data.voice.mode_changed) {
            addLog(`Voice switched mode to ${data.voice.new_mode}`, 'success');
            await refreshState();
        } else {
            addLog(data.voice.message, data.voice.recognized ? 'success' : 'error');
        }

        dom.voiceInput.value = '';
    } catch (err) {
        addLog(`Voice command failed: ${err.message}`, 'error');
    }
}

// ── Keyboard Shortcuts ──────────────────────────────────────

function handleKeyboard(e) {
    if (document.activeElement === dom.voiceInput) return;

    const keyMap = {
        ArrowUp:    'drive_forward',
        ArrowDown:  'drive_backward',
        ArrowLeft:  'turn_left',
        ArrowRight: 'turn_right',
        ' ':        'stop',
    };

    const action = keyMap[e.key];
    if (action) {
        e.preventDefault();
        executeAction(action);
    }
}

// ── Init ────────────────────────────────────────────────────

function init() {
    document.querySelectorAll('.mode-card').forEach(card =>
        card.addEventListener('click', () => selectMode(card.dataset.mode))
    );

    document.querySelectorAll('.dpad-btn').forEach(btn =>
        btn.addEventListener('click', () => executeAction(btn.dataset.action))
    );

    dom.voiceSend.addEventListener('click', sendVoiceCommand);
    dom.voiceInput.addEventListener('keydown', e => {
        if (e.key === 'Enter') sendVoiceCommand();
    });

    document.addEventListener('keydown', handleKeyboard);

    addLog('Baymax Control Panel loaded');
    refreshState();
    setInterval(refreshState, 5000);
}

document.addEventListener('DOMContentLoaded', init);
