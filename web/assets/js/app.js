/* DeskBot shared web helpers — API access, API-key persistence, formatting,
   toast, and the single source of truth for the top nav. Loaded by every
   dashboard via /assets/js/app.js. All exports are globals (pages use plain
   inline scripts). */
'use strict';

const $ = (id) => document.getElementById(id);
const API_BASE = '/api/v1';

/* ---- API key (persists across pages via localStorage) ---- */
function getApiKey() {
  const el = document.getElementById('api-key-input');
  if (el && el.value.trim()) return el.value.trim();
  try { return localStorage.getItem('deskbot.apikey') || ''; } catch { return ''; }
}
function setApiKey(v) {
  try { localStorage.setItem('deskbot.apikey', v || ''); } catch { /* ignore */ }
}
function authHeaders() {
  const h = {};
  const k = getApiKey();
  if (k) h['Authorization'] = 'Bearer ' + k;
  return h;
}

/* ---- HTTP helpers (root-relative urls; pages prefix their own API area) ----
   apiGet / apiPost throw on non-2xx with `.status` and parsed `.body`.
   Pass {safe:true} to swallow errors: apiGet returns null, apiPost returns
   the parsed body (or a synthetic {status:'error'}/ {status:'ok'}). */
async function apiGet(url, opts) {
  opts = opts || {};
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) {
    if (opts.safe) return null;
    const err = new Error('HTTP ' + r.status);
    err.status = r.status;
    try { err.body = await r.json(); } catch { err.body = null; }
    throw err;
  }
  try { return await r.json(); } catch { return null; }
}

async function apiPost(url, body, opts) {
  opts = opts || {};
  const headers = { 'Content-Type': 'application/json' };
  if (opts.auth !== false) Object.assign(headers, authHeaders());
  const r = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body || {}) });
  let data = null;
  try { data = await r.json(); } catch { data = null; }
  if (!r.ok) {
    if (opts.safe) return data || { status: 'error', _ok: false };
    const err = new Error('HTTP ' + r.status);
    err.status = r.status;
    err.body = data;
    throw err;
  }
  return data || { status: 'ok', _ok: true };
}

async function apiDelete(url) {
  const r = await fetch(url, { method: 'DELETE' });
  return r.ok;
}

/* ---- Formatting / DOM helpers ---- */
function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}
function badge(text, cls) { return `<span class="badge ${cls || 'neutral'}">${esc(text)}</span>`; }
function card(title, inner) { return `<div class="card"><div class="card-title">${title}</div>${inner}</div>`; }
function infoRow(label, value) { return `<div class="card-row"><span class="label">${esc(label)}</span><span class="value">${value}</span></div>`; }

function fmtUptime(s) {
  if (!s) return '-';
  if (s < 60) return Math.round(s) + 's';
  const m = Math.floor(s / 60);
  if (m < 60) return m + 'm';
  const h = Math.floor(s / 3600);
  if (h < 24) return h + 'h ' + Math.floor((s % 3600) / 60) + 'm';
  return Math.floor(h / 24) + 'd ' + (h % 24) + 'h';
}
function fmtTime(iso) {
  if (!iso) return '-';
  try { return new Date(iso).toLocaleString(); } catch { return iso; }
}
function fmtNum(n, digits) {
  digits = digits == null ? 4 : digits;
  if (n === null || n === undefined) return '-';
  if (typeof n !== 'number' || !isFinite(n)) return '-';
  return n.toFixed(digits);
}
function relTime(iso) {
  if (!iso) return '-';
  try {
    const diff = (Date.now() - new Date(iso).getTime()) / 1000;
    if (diff < 0) return 'just now';
    if (diff < 60) return Math.round(diff) + 's ago';
    if (diff < 3600) return Math.round(diff / 60) + 'm ago';
    if (diff < 86400) return Math.round(diff / 3600) + 'h ago';
    return Math.round(diff / 86400) + 'd ago';
  } catch { return iso; }
}

/* ---- Toast (single #toast element; created if missing) ---- */
function toast(msg, ok) {
  let t = document.getElementById('toast');
  if (!t) { t = document.createElement('div'); t.id = 'toast'; document.body.appendChild(t); }
  t.textContent = msg;
  t.className = 'show ' + (ok === false ? 'err' : 'ok');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.className = ''; }, 2800);
}

/* ---- Shared nav (one source of truth for dashboard links) ---- */
const NAV_ITEMS = [
  { key: 'dashboard',  label: 'Dashboard',        href: '/' },
  { key: 'learning',   label: 'Local Brain',      href: '/learning' },
  { key: 'teaching',   label: 'Teaching',         href: '/teaching' },
  { key: 'calibration',label: 'Calibration',      href: '/calibration' },
  { key: 'config',     label: 'Config Validator', href: '/config' },
  { key: 'settings',   label: 'Hardware Test',    href: '/settings' },
];

function _navActiveFor(el) {
  const explicit = el.getAttribute('data-active');
  if (explicit) return explicit;
  const path = location.pathname.replace(/\/$/, '') || '/';
  const match = NAV_ITEMS.find((i) => i.href === path);
  return match ? match.key : '';
}

function renderNav(el, current) {
  const active = current || _navActiveFor(el);
  el.innerHTML = NAV_ITEMS.map((i) =>
    i.key === active
      ? `<span class="nav-cur">${i.label}</span>`
      : `<a href="${i.href}">${i.label}</a>`
  ).join('');
}

function _init() {
  // Nav
  document.querySelectorAll('nav.app-nav').forEach((el) => renderNav(el));
  // API key input: seed from storage, persist on change
  const keyInput = document.getElementById('api-key-input');
  if (keyInput) {
    try { if (!keyInput.value) keyInput.value = localStorage.getItem('deskbot.apikey') || ''; } catch { /* ignore */ }
    keyInput.addEventListener('input', () => setApiKey(keyInput.value.trim()));
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', _init);
} else {
  _init();
}