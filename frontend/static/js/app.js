/* ── TherapistAI WebSocket client ─────────────────────────────────
   Single eternal session per user — no new-chat button, ever.
   User ID is minted once and stored in localStorage.
──────────────────────────────────────────────────────────────── */

// Use 127.0.0.1 explicitly — macOS sometimes resolves localhost → ::1 (IPv6) which breaks WS
const WS_HOST = (location.hostname === "localhost" || location.hostname === "127.0.0.1")
  ? "127.0.0.1:8001"
  : location.host;

// ── Persistent user ID (one per browser) ─────────────────────────
function getUserId() {
  let id = localStorage.getItem("therapist_user_id");
  if (!id) {
    id = "u_" + crypto.randomUUID().replace(/-/g, "").slice(0, 20);
    localStorage.setItem("therapist_user_id", id);
  }
  return id;
}

function getUserName() { return localStorage.getItem("therapist_user_name") || ""; }
function setUserName(n) { localStorage.setItem("therapist_user_name", n); }

// ── DOM refs ──────────────────────────────────────────────────────
const messagesEl   = document.getElementById("messages");
const inputEl      = document.getElementById("msg-input");
const sendBtn      = document.getElementById("send-btn");
const statusDot    = document.getElementById("status-dot");
const statusText   = document.getElementById("status-text");
const sidebar      = document.getElementById("sidebar");
const sidebarToggle= document.getElementById("sidebar-toggle");
const menuBtn      = document.getElementById("menu-btn");
const moodBadge    = document.getElementById("mood-badge");
const topicsList   = document.getElementById("topics-list");
const peopleList   = document.getElementById("people-list");
const intentLabel  = document.getElementById("intent-label");
const intentScore  = document.getElementById("intent-score");
const intentDump   = document.getElementById("intent-state-dump");
const nameOverlay  = document.getElementById("name-overlay");
const nameInput    = document.getElementById("name-input");
const nameSubmit   = document.getElementById("name-submit");

let ws = null;
let reconnectTimer = null;
const userId = getUserId();

// ── Sidebar toggle ─────────────────────────────────────────────────
function toggleSidebar() {
  sidebar.classList.toggle("collapsed");
  sidebarToggle.textContent = sidebar.classList.contains("collapsed") ? "▶" : "◀";
}
sidebarToggle.addEventListener("click", toggleSidebar);
menuBtn.addEventListener("click", () => {
  sidebar.classList.remove("collapsed");
  sidebarToggle.textContent = "◀";
});

// ── Name prompt (first visit) ────────────────────────────────────
function maybeShowNamePrompt() {
  if (!getUserName()) {
    nameOverlay.classList.remove("hidden");
    nameInput.focus();
  }
}

function submitName() {
  const name = nameInput.value.trim();
  setUserName(name);
  nameOverlay.classList.add("hidden");
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "set_name", name }));
  }
}

nameSubmit.addEventListener("click", submitName);
nameInput.addEventListener("keydown", e => { if (e.key === "Enter") submitName(); });

// ── WebSocket ─────────────────────────────────────────────────────
function connect() {
  clearTimeout(reconnectTimer);
  setStatus("connecting");

  ws = new WebSocket(`ws://${WS_HOST}/ws/${userId}`);

  ws.addEventListener("open", () => {
    setStatus("connected");
    sendBtn.disabled = false;
    maybeShowNamePrompt();
  });

  ws.addEventListener("message", e => {
    const data = JSON.parse(e.data);
    handleMessage(data);
  });

  ws.addEventListener("close", () => {
    setStatus("disconnected");
    sendBtn.disabled = true;
    reconnectTimer = setTimeout(connect, 3000);
  });

  ws.addEventListener("error", () => ws.close());
}

function setStatus(s) {
  statusDot.className = "status-dot " + s;
  statusText.textContent = { connecting: "Connecting…", connected: "Connected", disconnected: "Reconnecting…" }[s];
}

// ── Message handling ──────────────────────────────────────────────
function handleMessage(data) {
  switch (data.type) {
    case "history":
      removeTypingIndicator();
      data.messages.forEach(m => appendBubble(m.role, m.content, m.timestamp, false));
      scrollBottom();
      break;

    case "message":
      removeTypingIndicator();
      appendBubble(data.role, data.content, null, true);
      break;

    case "intent":
      updateIntentPanel(data);
      break;

    case "ack_name":
      // silently accepted
      break;
  }
}

// ── Bubbles ────────────────────────────────────────────────────────
function appendBubble(role, content, timestamp, animate) {
  const isCrisis = content.toLowerCase().includes("samaritans") ||
                   content.toLowerCase().includes("crisis lifeline");

  const msg = document.createElement("div");
  msg.className = "msg " + role + (isCrisis ? " crisis" : "");
  if (!animate) msg.style.animation = "none";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.innerHTML = markdownLite(content);

  const meta = document.createElement("div");
  meta.className = "msg-meta";
  const ts = timestamp ? new Date(timestamp) : new Date();
  meta.textContent = role === "therapist" ? "TherapistAI · " + fmtTime(ts) : fmtTime(ts);

  msg.appendChild(bubble);
  msg.appendChild(meta);
  messagesEl.appendChild(msg);
  scrollBottom();
}

function showTypingIndicator() {
  if (document.getElementById("typing")) return;
  const msg = document.createElement("div");
  msg.className = "msg therapist typing-indicator";
  msg.id = "typing";
  msg.innerHTML = `<div class="msg-bubble"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>`;
  messagesEl.appendChild(msg);
  scrollBottom();
}

function removeTypingIndicator() {
  const el = document.getElementById("typing");
  if (el) el.remove();
}

// ── Intent panel ───────────────────────────────────────────────────
function updateIntentPanel(data) {
  intentLabel.textContent = data.label;
  intentScore.textContent = (data.score * 100).toFixed(0) + "%";

  const state = data.state || {};

  // mood badge
  const mood = state.current_mood || "neutral";
  moodBadge.textContent = mood;
  moodBadge.className = "badge " + mood;

  // topics
  const topics = state.active_topics || [];
  if (topics.length) {
    topicsList.innerHTML = topics.map(t => `<li>${t}</li>`).join("");
  } else {
    topicsList.innerHTML = `<li class="empty">Nothing yet</li>`;
  }

  // people
  const people = state.mentioned_people || [];
  if (people.length) {
    peopleList.innerHTML = people.map(p => `<li>${p}</li>`).join("");
  } else {
    peopleList.innerHTML = `<li class="empty">No one yet</li>`;
  }

  // state dump
  intentDump.textContent = JSON.stringify(state, null, 2);
}

// ── Send ────────────────────────────────────────────────────────────
function send() {
  const text = inputEl.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

  appendBubble("user", text, null, true);
  ws.send(JSON.stringify({ type: "message", content: text }));
  inputEl.value = "";
  autoResize();
  showTypingIndicator();
}

sendBtn.addEventListener("click", send);
inputEl.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

// ── Auto-resize textarea ───────────────────────────────────────────
function autoResize() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
}
inputEl.addEventListener("input", autoResize);

// ── Helpers ────────────────────────────────────────────────────────
function scrollBottom() {
  requestAnimationFrame(() => {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  });
}

function fmtTime(d) {
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function markdownLite(text) {
  return text
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/^---$/gm, "<hr>")
    .replace(/\n/g, "<br>");
}

// ── Boot ───────────────────────────────────────────────────────────
connect();
