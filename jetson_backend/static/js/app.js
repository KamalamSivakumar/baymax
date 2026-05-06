'use strict';

// ── API Client ────────────────────────────────────────────────────────────────

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

const $ = (id) => document.getElementById(id);
const dom = {
    jetsonDot:    $('jetson-dot'),   piDot:       $('pi-dot'),
    ledDot:       $('led-dot'),      modeDescription: $('mode-description'),
    featuresGrid: $('features-grid'),motorSpeed:  $('motor-speed'),
    voiceInput:   $('voice-input'),  voiceSend:   $('voice-send'),
    chatInput:    $('chat-input'),   chatSend:    $('chat-send'),
    chatReset:    $('chat-reset'),   chatMessages:$('chat-messages'),
    cameraFeed:   $('camera-feed'),  cameraToggle:$('camera-toggle'),
    log:          $('log'),
};

let currentMode = null, cameraEnabled = false, ws = null;

function escapeHtml(text) {
    const el = document.createElement('span');
    el.textContent = text;
    return el.innerHTML;
}

function addLog(message, type = 'info') {
    const time = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `<span class="log-time">${time}</span>${escapeHtml(message)}`;
    dom.log.prepend(entry);
    while (dom.log.children.length > 50) dom.log.lastChild.remove();
}

// WebSocket
function connectWebSocket() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.addEventListener('open', () => { setDot(dom.jetsonDot, true); addLog('WebSocket connected', 'success'); });
    ws.addEventListener('message', (ev) => {
        try { const msg = JSON.parse(ev.data); if (msg.type === 'state_update') handleStateUpdate(msg.state); } catch {}
    });
    ws.addEventListener('close', () => { setDot(dom.jetsonDot, false); addLog('WebSocket disconnected — retrying in 5s', 'error'); setTimeout(connectWebSocket, 5000); });
    ws.addEventListener('error', () => ws.close());
}

function setDot(el, connected) { el.className = `dot ${connected ? 'connected' : 'disconnected'}`; }

const LED_COLORS = { red:'#e05', green:'#2d2', yellow:'#da0', all_on:'#fff', off:'#333' };

function handleStateUpdate(state) {
    setDot(dom.piDot, state.pi_connected);
    dom.ledDot.style.background = LED_COLORS[state.led_state] || '#333';
    dom.ledDot.title = `LED: ${state.led_state || 'off'}`;
    if (state.mode !== currentMode) applyMode(state.mode, state.mode_description, state.mode_config);
}

async function refreshState() {
    try {
        const { state } = await api.get('/state');
        setDot(dom.jetsonDot, true);
        handleStateUpdate(state);
    } catch { setDot(dom.jetsonDot, false); setDot(dom.piDot, false); }
}

// Mode
function applyMode(mode, description, config) {
    currentMode = mode;
    document.querySelectorAll('.mode-card').forEach(c => c.classList.toggle('active', c.dataset.mode === mode));
    dom.modeDescription.textContent = description || '';
    dom.motorSpeed.textContent = `Speed: ${config?.motor_speed ?? '–'}`;
    renderFeatures(config?.features || []);
}

async function selectMode(mode) {
    try {
        addLog(`Switching to ${mode} mode…`);
        const data = await api.post('/mode', { mode });
        applyMode(mode, data.backend?.description, data.backend?.config);
        setDot(dom.piDot, data.raspberry_pi?.success ?? false);
        addLog(`Mode set to ${mode}`, 'success');
    } catch (err) { addLog(`Mode switch failed: ${err.message}`, 'error'); }
}

function renderFeatures(features) {
    dom.featuresGrid.innerHTML = '';
    features.forEach(f => {
        const btn = document.createElement('button');
        btn.className = 'feature-btn'; btn.textContent = f.name;
        btn.addEventListener('click', () => executeAction(f.action, f.name));
        dom.featuresGrid.appendChild(btn);
    });
}

async function executeAction(action, label) {
    try {
        addLog(`Executing: ${label || action}…`);
        const data = await api.post('/robot/action', { action });
        addLog(`${label || action} — ${data.status}`, data.status === 'blocked' ? 'error' : 'success');
    } catch (err) { addLog(`Action failed: ${err.message}`, 'error'); }
}

// Eye / LED
async function setEyeExpression(expression) {
    try { await api.post('/eyes', { expression }); addLog(`Eye: ${expression}`, 'success'); }
    catch (err) { addLog(`Eye failed: ${err.message}`, 'error'); }
}

async function setLedColor(color) {
    try { await api.post('/led', { color }); addLog(`LED: ${color}`, 'success'); }
    catch (err) { addLog(`LED failed: ${err.message}`, 'error'); }
}

// Camera
function toggleCamera() {
    cameraEnabled = !cameraEnabled;
    if (cameraEnabled) {
        dom.cameraFeed.src = `http://${location.hostname}:8080/stream`;
        dom.cameraToggle.textContent = 'Disable';
        addLog('Camera feed enabled');
    } else {
        dom.cameraFeed.src = '';
        dom.cameraToggle.textContent = 'Enable';
        addLog('Camera feed disabled');
    }
}

// Chat
function appendChatBubble(text, role) {
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${role}`;
    bubble.textContent = text;
    dom.chatMessages.appendChild(bubble);
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
}

async function sendChat() {
    const message = dom.chatInput.value.trim();
    if (!message) return;
    dom.chatInput.value = '';
    appendChatBubble(message, 'user');
    addLog(`Chat: "${message}"`);
    try {
        const data = await api.post('/llm/chat', { message });
        appendChatBubble(data.response, 'assistant');
        if (data.tool_calls?.length) addLog(`LLM used tools: ${data.tool_calls.map(t => t.name).join(', ')}`, 'success');
        await refreshState();
    } catch (err) { appendChatBubble(`Error: ${err.message}`, 'assistant'); addLog(`Chat error: ${err.message}`, 'error'); }
}

async function resetChat() {
    try { await api.post('/llm/reset', {}); dom.chatMessages.innerHTML = ''; addLog('Conversation reset', 'success'); }
    catch (err) { addLog(`Reset failed: ${err.message}`, 'error'); }
}

// Voice
async function sendVoiceCommand() {
    const command = dom.voiceInput.value.trim();
    if (!command) return;
    try {
        addLog(`Voice: "${command}"`);
        const data = await api.post('/voice-command', { command });
        const v = data.voice;
        if (v.mode_changed) { addLog(`Voice switched mode to ${v.new_mode}`, 'success'); await refreshState(); }
        else if (v.llm_used) { appendChatBubble(command, 'user'); appendChatBubble(v.message, 'assistant'); addLog('LLM handled voice command', 'success'); }
        else addLog(v.message, v.recognized ? 'success' : 'error');
        dom.voiceInput.value = '';
    } catch (err) { addLog(`Voice command failed: ${err.message}`, 'error'); }
}

function handleKeyboard(e) {
    const active = document.activeElement;
    if (active === dom.voiceInput || active === dom.chatInput) return;
    const keyMap = { ArrowUp:'drive_forward', ArrowDown:'drive_backward', ArrowLeft:'turn_left', ArrowRight:'turn_right', ' ':'stop' };
    const action = keyMap[e.key];
    if (action) { e.preventDefault(); executeAction(action); }
}

function init() {
    document.querySelectorAll('.mode-card').forEach(c => c.addEventListener('click', () => selectMode(c.dataset.mode)));
    document.querySelectorAll('.dpad-btn').forEach(b => b.addEventListener('click', () => executeAction(b.dataset.action)));
    document.querySelectorAll('.eye-btn').forEach(b => b.addEventListener('click', () => setEyeExpression(b.dataset.expr)));
    document.querySelectorAll('.led-btn').forEach(b => b.addEventListener('click', () => setLedColor(b.dataset.color)));
    dom.cameraToggle.addEventListener('click', toggleCamera);
    dom.chatSend.addEventListener('click', sendChat);
    dom.chatInput.addEventListener('keydown', e => { if (e.key === 'Enter') sendChat(); });
    dom.chatReset.addEventListener('click', resetChat);
    dom.voiceSend.addEventListener('click', sendVoiceCommand);
    dom.voiceInput.addEventListener('keydown', e => { if (e.key === 'Enter') sendVoiceCommand(); });
    document.addEventListener('keydown', handleKeyboard);
    addLog('Baymax Control Panel loaded');
    connectWebSocket();
    refreshState();
}

document.addEventListener('DOMContentLoaded', init);