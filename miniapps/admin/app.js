// State
const state = {
  items: [],
  offset: 0,
  limit: 50,
  sort: "generations",
  order: "desc",
  total: 0,
  loading: false,
  search: "",
  stats: null,
};

// DOM Elements
const elements = {
  statusBadge: document.getElementById("statusBadge"),
  statsGrid: document.getElementById("statsGrid"),
  statUsers: document.getElementById("statUsers"),
  statGenerations: document.getElementById("statGenerations"),
  statPhones: document.getElementById("statPhones"),
  statEvents: document.getElementById("statEvents"),
  searchInput: document.getElementById("searchInput"),
  sortSelect: document.getElementById("sortSelect"),
  refreshBtn: document.getElementById("refreshBtn"),
  usersList: document.getElementById("usersList"),
  loadMoreBtn: document.getElementById("loadMoreBtn"),
  loadMoreContainer: document.getElementById("loadMoreContainer"),
  usersCount: document.getElementById("usersCount"),
  emptyState: document.getElementById("emptyState"),
  errorState: document.getElementById("errorState"),
  errorTitle: document.getElementById("errorTitle"),
  errorText: document.getElementById("errorText"),
  retryBtn: document.getElementById("retryBtn"),
  loadingState: document.getElementById("loadingState"),
  modalOverlay: document.getElementById("modalOverlay"),
  userModal: document.getElementById("userModal"),
  modalTitle: document.getElementById("modalTitle"),
  modalBody: document.getElementById("modalBody"),
  modalClose: document.getElementById("modalClose"),
  confirmOverlay: document.getElementById("confirmOverlay"),
  confirmText: document.getElementById("confirmText"),
  confirmCancel: document.getElementById("confirmCancel"),
  confirmOk: document.getElementById("confirmOk"),
  toast: document.getElementById("toast"),
  toastMessage: document.getElementById("toastMessage"),
};

// Telegram WebApp
const telegram = window.Telegram && window.Telegram.WebApp;
if (telegram) {
  telegram.ready();
  telegram.expand();
}

const initData = telegram ? telegram.initData : "";
const apiBase = (() => {
  const raw = window.ADMIN_CONFIG && window.ADMIN_CONFIG.apiBaseUrl;
  if (!raw) return "";
  return raw.replace(/\/$/, "");
})();

// API helpers
function buildUrl(endpoint, params = {}) {
  const url = new URL(`${apiBase || window.location.origin}${endpoint}`);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      url.searchParams.set(key, String(value));
    }
  });
  return url.toString();
}

async function apiRequest(endpoint, options = {}) {
  const url = buildUrl(endpoint, options.params);
  const response = await fetch(url, {
    method: options.method || "GET",
    headers: {
      "X-Telegram-Init-Data": initData,
      "Content-Type": "application/json",
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error || `HTTP ${response.status}`);
  }
  
  return response.json();
}

// UI Helpers
function setStatus(text, mode) {
  elements.statusBadge.className = `status-badge ${mode}`;
  elements.statusBadge.querySelector(".status-text").textContent = text;
}

function showToast(message, type = "default") {
  elements.toastMessage.textContent = message;
  elements.toast.className = `toast ${type}`;
  elements.toast.classList.remove("hidden");
  setTimeout(() => elements.toast.classList.add("hidden"), 3000);
}

function formatNumber(num) {
  if (num >= 1000000) return (num / 1000000).toFixed(1) + "M";
  if (num >= 1000) return (num / 1000).toFixed(1) + "K";
  return String(num);
}

function formatUserName(user) {
  // Use display_name from API which prefers real name over username
  if (user.display_name) return user.display_name;
  if (user.full_name) return user.full_name;
  if (user.first_name || user.last_name) {
    return [user.first_name, user.last_name].filter(Boolean).join(" ");
  }
  if (user.username) return `@${user.username}`;
  return `User ${user.user_id}`;
}

function getInitials(user) {
  const name = user.full_name || user.first_name || user.display_name;
  if (name && !name.startsWith("@") && !name.startsWith("User ")) {
    const parts = name.split(" ");
    return parts.map(p => p[0]).join("").toUpperCase().slice(0, 2);
  }
  if (user.username) return user.username[0].toUpperCase();
  return "U";
}

// Render Functions
function renderStats() {
  if (!state.stats) return;
  elements.statUsers.textContent = formatNumber(state.stats.total_users || 0);
  elements.statGenerations.textContent = formatNumber(state.stats.total_generations || 0);
  elements.statPhones.textContent = formatNumber(state.stats.users_with_phone || 0);
  // Show event paid uses (free not counted per user request)
  elements.statEvents.textContent = formatNumber(state.stats.event_paid_used || 0);
}

function renderUserCard(user) {
  const hasPhone = user.phone ? '<span class="user-phone">📱</span>' : '';
  const hasEvent = user.has_event_record ? '<span class="user-event-badge">🎄</span>' : '';
  const triesClass = user.tries_remaining > 0 ? 'success' : 'warning';
  
  // Event stats - only show real values if user has event record
  let eventDisplay, eventClass;
  if (user.has_event_record) {
    eventDisplay = `${user.event_paid_remaining}/${user.event_paid_limit}`;
    eventClass = user.event_paid_remaining > 0 ? 'success' : 'warning';
  } else {
    eventDisplay = '—';
    eventClass = '';
  }
  
  return `
    <div class="user-card" data-user-id="${user.user_id}">
      <div class="user-header">
        <div class="user-info">
          <div class="user-name">${formatUserName(user)}${hasPhone}${hasEvent}</div>
          <div class="user-id">ID: ${user.user_id}${user.username ? ` · @${user.username}` : ''}</div>
        </div>
      </div>
      <div class="user-stats">
        <div class="user-stat highlight">
          <div class="user-stat-value">${user.generations}</div>
          <div class="user-stat-label">Генераций</div>
        </div>
        <div class="user-stat ${triesClass}">
          <div class="user-stat-value">${user.tries_remaining}/${user.tries_limit}</div>
          <div class="user-stat-label">Попыток</div>
        </div>
        <div class="user-stat ${eventClass}">
          <div class="user-stat-value">${eventDisplay}</div>
          <div class="user-stat-label">Ивент</div>
        </div>
        <div class="user-stat">
          <div class="user-stat-value">${user.site_clicks}</div>
          <div class="user-stat-label">Клики</div>
        </div>
      </div>
    </div>
  `;
}

function renderUsersList() {
  const filtered = filterUsers(state.items);
  
  if (filtered.length === 0 && state.items.length === 0) {
    elements.usersList.innerHTML = "";
    elements.emptyState.classList.remove("hidden");
    elements.loadMoreContainer.classList.add("hidden");
    return;
  }
  
  elements.emptyState.classList.add("hidden");
  
  if (filtered.length === 0) {
    elements.usersList.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">🔍</div>
        <div class="empty-title">Ничего не найдено</div>
        <div class="empty-text">Попробуйте изменить запрос</div>
      </div>
    `;
    elements.loadMoreContainer.classList.add("hidden");
    return;
  }
  
  elements.usersList.innerHTML = filtered.map(renderUserCard).join("");
  elements.loadMoreContainer.classList.remove("hidden");
  elements.usersCount.textContent = `${state.items.length} из ${state.total}`;
  elements.loadMoreBtn.disabled = state.items.length >= state.total;
  
  // Attach click handlers
  document.querySelectorAll(".user-card").forEach(card => {
    card.addEventListener("click", () => {
      const userId = parseInt(card.dataset.userId);
      const user = state.items.find(u => u.user_id === userId);
      if (user) openUserModal(user);
    });
  });
}

function filterUsers(users) {
  if (!state.search) return users;
  const query = state.search.toLowerCase();
  return users.filter(user => {
    const name = formatUserName(user).toLowerCase();
    const id = String(user.user_id);
    const phone = (user.phone || "").toLowerCase();
    const username = (user.username || "").toLowerCase();
    return name.includes(query) || id.includes(query) || phone.includes(query) || username.includes(query);
  });
}

// Modal Functions
function openUserModal(user) {
  elements.modalTitle.textContent = formatUserName(user);
  const usernameDisplay = user.username ? `@${user.username}` : '';
  const eventBadge = user.has_event_record ? ' 🎄' : '';
  
  // Event section - only show if user has event record
  let eventSection = '';
  if (user.has_event_record) {
    eventSection = `
    <div class="section-title">Ивент попытки 🎄</div>
    <div class="form-row">
      <div class="form-group">
        <label class="form-label">Осталось попыток</label>
        <input type="number" class="form-input" id="inputEventRemaining" value="${user.event_paid_remaining}" min="0" max="${user.event_paid_limit}" />
        <div class="form-hint">Использовано: ${user.event_paid_used} из ${user.event_paid_limit}</div>
      </div>
      <div class="form-group">
        <label class="form-label">Лимит попыток</label>
        <input type="number" class="form-input" id="inputEventLimit" value="${user.event_paid_limit}" disabled />
        <div class="form-hint">Фиксированный лимит</div>
      </div>
    </div>
    <button class="btn btn-secondary" id="saveEventTriesBtn" style="width: 100%;">Сохранить ивент попытки</button>
    `;
  } else {
    eventSection = `
    <div class="section-title">Ивент попытки</div>
    <div class="form-hint" style="text-align: center; padding: 16px; background: var(--bg-secondary); border-radius: var(--radius-md);">
      Пользователь ещё не заходил в ивент
    </div>
    `;
  }
  
  elements.modalBody.innerHTML = `
    <div class="user-detail-header">
      <div class="user-avatar">${getInitials(user)}</div>
      <div class="user-detail-info">
        <h3>${formatUserName(user)}${eventBadge}</h3>
        <p>ID: ${user.user_id}${usernameDisplay ? ` · ${usernameDisplay}` : ''}</p>
        <p><a href="${user.telegram_link}" target="_blank">Открыть профиль в Telegram</a></p>
        ${user.phone ? `<p>📱 ${user.phone}</p>` : ''}
      </div>
    </div>
    
    <div class="section-title">Основные попытки</div>
    <div class="form-row">
      <div class="form-group">
        <label class="form-label">Осталось попыток</label>
        <input type="number" class="form-input" id="inputTriesRemaining" value="${user.tries_remaining}" min="0" />
        <div class="form-hint">Использовано: ${user.tries_used} из ${user.tries_limit}</div>
      </div>
      <div class="form-group">
        <label class="form-label">Лимит попыток</label>
        <input type="number" class="form-input" id="inputTriesLimit" value="${user.tries_limit}" min="1" />
      </div>
    </div>
    <button class="btn btn-primary" id="saveTriesBtn" style="width: 100%;">Сохранить попытки</button>
    
    ${eventSection}
    
    <div class="section-title">Статистика</div>
    <div class="user-stats" style="margin-bottom: 12px;">
      <div class="user-stat">
        <div class="user-stat-value">${user.generations}</div>
        <div class="user-stat-label">Генераций</div>
      </div>
      <div class="user-stat">
        <div class="user-stat-value">${user.site_clicks}</div>
        <div class="user-stat-label">Клики сайт</div>
      </div>
      <div class="user-stat">
        <div class="user-stat-value">${user.social_clicks}</div>
        <div class="user-stat-label">Клики соц.</div>
      </div>
    </div>
    
    <div class="delete-zone">
      <button class="btn btn-danger" id="deleteUserBtn">
        🗑️ Удалить пользователя
      </button>
      <div class="form-hint" style="margin-top: 8px; text-align: center;">
        Полностью удалит все данные пользователя. Он сможет начать заново.
      </div>
    </div>
  `;
  
  // Attach handlers
  document.getElementById("saveTriesBtn").addEventListener("click", () => saveUserTries(user.user_id));
  if (user.has_event_record) {
    document.getElementById("saveEventTriesBtn").addEventListener("click", () => saveEventTries(user.user_id, user.event_paid_limit));
  }
  document.getElementById("deleteUserBtn").addEventListener("click", () => confirmDeleteUser(user));
  
  elements.modalOverlay.classList.remove("hidden");
}

function closeModal() {
  elements.modalOverlay.classList.add("hidden");
}

function closeConfirm() {
  elements.confirmOverlay.classList.add("hidden");
}

// Actions
async function saveUserTries(userId) {
  const triesRemaining = parseInt(document.getElementById("inputTriesRemaining").value);
  const triesLimit = parseInt(document.getElementById("inputTriesLimit").value);
  
  if (isNaN(triesRemaining) || isNaN(triesLimit) || triesRemaining < 0 || triesLimit < 1) {
    showToast("Проверьте введённые значения", "error");
    return;
  }
  
  try {
    await apiRequest(`/admin/api/users/${userId}/tries`, {
      method: "POST",
      body: { tries_remaining: triesRemaining, tries_limit: triesLimit },
    });
    showToast("Попытки обновлены", "success");
    closeModal();
    fetchUsers({ reset: true });
  } catch (error) {
    showToast(`Ошибка: ${error.message}`, "error");
  }
}

async function saveEventTries(userId, eventLimit) {
  const eventRemaining = parseInt(document.getElementById("inputEventRemaining").value);
  
  if (isNaN(eventRemaining) || eventRemaining < 0 || eventRemaining > eventLimit) {
    showToast("Проверьте введённые значения", "error");
    return;
  }
  
  // Calculate paid_used from remaining: paid_used = limit - remaining
  const paidUsed = eventLimit - eventRemaining;
  
  try {
    await apiRequest(`/admin/api/users/${userId}/events/default`, {
      method: "POST",
      body: { paid_used: paidUsed },
    });
    showToast("Ивент попытки обновлены", "success");
    closeModal();
    fetchUsers({ reset: true });
  } catch (error) {
    showToast(`Ошибка: ${error.message}`, "error");
  }
}

function confirmDeleteUser(user) {
  elements.confirmText.textContent = `Удалить пользователя ${formatUserName(user)}? Это действие нельзя отменить.`;
  elements.confirmOverlay.classList.remove("hidden");
  
  elements.confirmOk.onclick = async () => {
    try {
      await apiRequest(`/admin/api/users/${user.user_id}`, { method: "DELETE" });
      showToast("Пользователь удалён", "success");
      closeConfirm();
      closeModal();
      fetchUsers({ reset: true });
    } catch (error) {
      showToast(`Ошибка: ${error.message}`, "error");
      closeConfirm();
    }
  };
}

// Data Fetching
async function fetchStats() {
  try {
    state.stats = await apiRequest("/admin/api/stats");
    renderStats();
  } catch (error) {
    console.error("Failed to fetch stats:", error);
  }
}

async function fetchUsers({ reset = false } = {}) {
  if (state.loading) return;
  
  if (!initData) {
    setStatus("Нет доступа", "error");
    elements.errorTitle.textContent = "Откройте в Telegram";
    elements.errorText.textContent = "Для доступа к API этой панели управления требуются initData Telegram.";
    elements.errorState.classList.remove("hidden");
    elements.loadingState.classList.add("hidden");
    elements.usersList.classList.add("hidden");
    return;
  }
  
  state.loading = true;
  setStatus("Загрузка...", "loading");
  
  if (reset) {
    state.offset = 0;
    state.items = [];
    elements.usersList.innerHTML = "";
    elements.loadingState.classList.remove("hidden");
  }
  
  try {
    const data = await apiRequest("/admin/api/users", {
      params: {
        offset: state.offset,
        limit: state.limit,
        sort: state.sort,
        order: state.order,
      },
    });
    
    state.total = data.total || 0;
    
    if (Array.isArray(data.items)) {
      if (reset) {
        state.items = data.items;
      } else {
        state.items = [...state.items, ...data.items];
      }
      state.offset = state.items.length;
    }
    
    elements.loadingState.classList.add("hidden");
    elements.errorState.classList.add("hidden");
    elements.usersList.classList.remove("hidden");
    
    renderUsersList();
    setStatus("Готово", "ready");
    
  } catch (error) {
    setStatus("Ошибка", "error");
    elements.loadingState.classList.add("hidden");
    
    if (state.items.length === 0) {
      elements.errorTitle.textContent = "Не удалось загрузить";
      elements.errorText.textContent = error.message || "Проверьте подключение";
      elements.errorState.classList.remove("hidden");
    } else {
      showToast(`Ошибка: ${error.message}`, "error");
    }
  } finally {
    state.loading = false;
  }
}

// Event Listeners
elements.refreshBtn.addEventListener("click", () => {
  fetchStats();
  fetchUsers({ reset: true });
});

elements.loadMoreBtn.addEventListener("click", () => {
  fetchUsers();
});

elements.sortSelect.addEventListener("change", (e) => {
  state.sort = e.target.value;
  fetchUsers({ reset: true });
});

elements.searchInput.addEventListener("input", (e) => {
  state.search = e.target.value;
  renderUsersList();
});

elements.modalClose.addEventListener("click", closeModal);
elements.modalOverlay.addEventListener("click", (e) => {
  if (e.target === elements.modalOverlay) closeModal();
});

elements.confirmCancel.addEventListener("click", closeConfirm);
elements.confirmOverlay.addEventListener("click", (e) => {
  if (e.target === elements.confirmOverlay) closeConfirm();
});

elements.retryBtn.addEventListener("click", () => {
  fetchStats();
  fetchUsers({ reset: true });
});

// Keyboard handling moved to broadcast section below

// Auto-refresh interval (every 30 seconds)
let autoRefreshInterval = null;

function startAutoRefresh() {
  if (autoRefreshInterval) return;
  autoRefreshInterval = setInterval(() => {
    // Only refresh if modal is closed and not currently loading
    if (elements.modalOverlay.classList.contains("hidden") && !state.loading) {
      fetchStats();
      fetchUsers({ reset: true });
    }
  }, 30000); // 30 seconds
}

function stopAutoRefresh() {
  if (autoRefreshInterval) {
    clearInterval(autoRefreshInterval);
    autoRefreshInterval = null;
  }
}

// Handle visibility change - pause refresh when tab is hidden
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    stopAutoRefresh();
  } else {
    startAutoRefresh();
    // Refresh immediately when tab becomes visible
    if (!state.loading) {
      fetchStats();
      fetchUsers({ reset: true });
    }
  }
});

// Broadcast functionality
const broadcastElements = {
  btn: document.getElementById("broadcastBtn"),
  overlay: document.getElementById("broadcastOverlay"),
  close: document.getElementById("broadcastClose"),
  cancel: document.getElementById("broadcastCancel"),
  send: document.getElementById("broadcastSend"),
  preview: document.getElementById("broadcastPreview"),
  previewBtnText: document.querySelector(".preview-btn-text"),
  previewSection: document.getElementById("previewSection"),
  previewContent: document.getElementById("previewContent"),
  editMode: document.getElementById("broadcastEditMode"),
  text: document.getElementById("broadcastText"),
  photo: document.getElementById("broadcastPhoto"),
  photoPreview: document.getElementById("photoPreview"),
  photoPreviewImg: document.getElementById("photoPreviewImg"),
  photoPlaceholder: document.getElementById("photoPlaceholder"),
  photoRemove: document.getElementById("photoRemove"),
  photoUploadArea: document.getElementById("photoUploadArea"),
  userCount: document.getElementById("broadcastUserCount"),
};

let broadcastPhotoBase64 = null;
let broadcastPhotoDataUrl = null;
let isPreviewMode = false;

function openBroadcastModal() {
  broadcastElements.text.value = "";
  broadcastPhotoBase64 = null;
  broadcastPhotoDataUrl = null;
  broadcastElements.photoPreview.classList.add("hidden");
  broadcastElements.photoPlaceholder.classList.remove("hidden");
  broadcastElements.userCount.textContent = state.total || 0;
  // Reset to edit mode
  setPreviewMode(false);
  broadcastElements.overlay.classList.remove("hidden");
}

function closeBroadcastModal() {
  broadcastElements.overlay.classList.add("hidden");
  setPreviewMode(false);
}

function handlePhotoSelect(file) {
  if (!file || !file.type.startsWith("image/")) return;
  
  const reader = new FileReader();
  reader.onload = (e) => {
    const dataUrl = e.target.result;
    broadcastPhotoDataUrl = dataUrl;
    broadcastPhotoBase64 = dataUrl.split(",")[1];
    broadcastElements.photoPreviewImg.src = dataUrl;
    broadcastElements.photoPreview.classList.remove("hidden");
    broadcastElements.photoPlaceholder.classList.add("hidden");
  };
  reader.readAsDataURL(file);
}

function removePhoto() {
  broadcastPhotoBase64 = null;
  broadcastPhotoDataUrl = null;
  broadcastElements.photo.value = "";
  broadcastElements.photoPreview.classList.add("hidden");
  broadcastElements.photoPlaceholder.classList.remove("hidden");
}

function sanitizeHtmlForPreview(html) {
  // Only allow safe Telegram HTML tags: b, i, u, s, a, code, pre
  // First escape any potentially dangerous content
  const tempDiv = document.createElement("div");
  tempDiv.textContent = html;
  let escaped = tempDiv.innerHTML;
  
  // Now restore allowed Telegram HTML tags
  const allowedTags = [
    { open: /&lt;b&gt;/gi, close: /&lt;\/b&gt;/gi, openTag: "<b>", closeTag: "</b>" },
    { open: /&lt;strong&gt;/gi, close: /&lt;\/strong&gt;/gi, openTag: "<strong>", closeTag: "</strong>" },
    { open: /&lt;i&gt;/gi, close: /&lt;\/i&gt;/gi, openTag: "<i>", closeTag: "</i>" },
    { open: /&lt;em&gt;/gi, close: /&lt;\/em&gt;/gi, openTag: "<em>", closeTag: "</em>" },
    { open: /&lt;u&gt;/gi, close: /&lt;\/u&gt;/gi, openTag: "<u>", closeTag: "</u>" },
    { open: /&lt;s&gt;/gi, close: /&lt;\/s&gt;/gi, openTag: "<s>", closeTag: "</s>" },
    { open: /&lt;code&gt;/gi, close: /&lt;\/code&gt;/gi, openTag: "<code>", closeTag: "</code>" },
    { open: /&lt;pre&gt;/gi, close: /&lt;\/pre&gt;/gi, openTag: "<pre>", closeTag: "</pre>" },
  ];
  
  allowedTags.forEach(tag => {
    escaped = escaped.replace(tag.open, tag.openTag);
    escaped = escaped.replace(tag.close, tag.closeTag);
  });
  
  // Handle <a href="..."> links specially
  escaped = escaped.replace(/&lt;a\s+href=&quot;([^&]+)&quot;&gt;/gi, '<a href="$1" target="_blank">');
  escaped = escaped.replace(/&lt;a\s+href='([^']+)'&gt;/gi, '<a href="$1" target="_blank">');
  escaped = escaped.replace(/&lt;\/a&gt;/gi, "</a>");
  
  return escaped;
}

function setPreviewMode(enabled) {
  isPreviewMode = enabled;
  
  if (enabled) {
    // Switch to preview mode
    broadcastElements.editMode.classList.add("hidden");
    broadcastElements.previewSection.classList.remove("hidden");
    broadcastElements.previewBtnText.textContent = "Редактировать";
    broadcastElements.preview.classList.add("active");
    
    // Render preview content
    renderPreviewContent();
  } else {
    // Switch to edit mode
    broadcastElements.editMode.classList.remove("hidden");
    broadcastElements.previewSection.classList.add("hidden");
    broadcastElements.previewBtnText.textContent = "Предпросмотр";
    broadcastElements.preview.classList.remove("active");
  }
}

function renderPreviewContent() {
  const text = broadcastElements.text.value.trim();
  
  if (!text && !broadcastPhotoDataUrl) {
    broadcastElements.previewContent.innerHTML = `
      <div class="preview-empty">Введите текст или добавьте фото для предпросмотра</div>
    `;
    return;
  }
  
  let previewHtml = '<div class="preview-message">';
  
  // Add photo if present
  if (broadcastPhotoDataUrl) {
    previewHtml += `<img src="${broadcastPhotoDataUrl}" alt="Фото" />`;
  }
  
  // Add text with HTML rendering
  if (text) {
    const sanitizedText = sanitizeHtmlForPreview(text);
    previewHtml += sanitizedText;
  }
  
  previewHtml += '</div>';
  
  broadcastElements.previewContent.innerHTML = previewHtml;
}

function togglePreview() {
  setPreviewMode(!isPreviewMode);
}

async function sendBroadcast() {
  const text = broadcastElements.text.value.trim();
  
  if (!text && !broadcastPhotoBase64) {
    showToast("Введите текст или добавьте фото", "error");
    return;
  }
  
  broadcastElements.send.disabled = true;
  broadcastElements.send.innerHTML = `
    <div class="spinner" style="width: 16px; height: 16px; border-width: 2px;"></div>
    Отправка...
  `;
  
  try {
    const result = await apiRequest("/admin/api/broadcast", {
      method: "POST",
      body: {
        text: text,
        photo_base64: broadcastPhotoBase64 || "",
      },
    });
    closeBroadcastModal();
    showToast(`✅ Отправлено: ${result.sent} из ${result.total} (заблокировано: ${result.blocked})`, "success");
  } catch (error) {
    showToast(`❌ ${error.message}`, "error");
  } finally {
    broadcastElements.send.disabled = false;
    broadcastElements.send.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
        <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
      </svg>
      Отправить
    `;
  }
}

// Broadcast event listeners
broadcastElements.btn.addEventListener("click", openBroadcastModal);
broadcastElements.close.addEventListener("click", closeBroadcastModal);
broadcastElements.cancel.addEventListener("click", closeBroadcastModal);
broadcastElements.overlay.addEventListener("click", (e) => {
  if (e.target === broadcastElements.overlay) closeBroadcastModal();
});
broadcastElements.send.addEventListener("click", sendBroadcast);
broadcastElements.preview.addEventListener("click", togglePreview);

broadcastElements.photo.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) handlePhotoSelect(file);
});

broadcastElements.photoRemove.addEventListener("click", (e) => {
  e.stopPropagation();
  removePhoto();
});

// Drag and drop support
broadcastElements.photoUploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  broadcastElements.photoUploadArea.classList.add("dragover");
});

broadcastElements.photoUploadArea.addEventListener("dragleave", () => {
  broadcastElements.photoUploadArea.classList.remove("dragover");
});

broadcastElements.photoUploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  broadcastElements.photoUploadArea.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) handlePhotoSelect(file);
});

// Update keyboard handling to include broadcast and dashboard modals
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (!dashboardElements.overlay.classList.contains("hidden")) {
      closeDashboardModal();
    } else if (!broadcastElements.overlay.classList.contains("hidden")) {
      closeBroadcastModal();
    } else if (!elements.confirmOverlay.classList.contains("hidden")) {
      closeConfirm();
    } else if (!elements.modalOverlay.classList.contains("hidden")) {
      closeModal();
    }
  }
});

// Dashboard functionality
const dashboardElements = {
  btn: document.getElementById("dashboardBtn"),
  overlay: document.getElementById("dashboardOverlay"),
  close: document.getElementById("dashboardClose"),
  loading: document.getElementById("dashboardLoading"),
  empty: document.getElementById("dashboardEmpty"),
  chart: document.getElementById("dashboardChart"),
  filters: document.querySelectorAll(".filter-btn"),
};

let dashboardChart = null;
let currentPeriod = "4h";

function openDashboardModal() {
  dashboardElements.overlay.classList.remove("hidden");
  fetchDashboardData(currentPeriod);
}

function closeDashboardModal() {
  dashboardElements.overlay.classList.add("hidden");
}

async function fetchDashboardData(period) {
  currentPeriod = period;
  
  // Update active filter button
  dashboardElements.filters.forEach(btn => {
    btn.classList.toggle("active", btn.dataset.period === period);
  });
  
  // Show loading
  dashboardElements.loading.classList.remove("hidden");
  dashboardElements.empty.classList.add("hidden");
  
  try {
    const data = await apiRequest("/admin/api/dashboard", {
      params: { period },
    });
    
    dashboardElements.loading.classList.add("hidden");
    
    if (!data.data || data.data.length === 0) {
      dashboardElements.empty.classList.remove("hidden");
      if (dashboardChart) {
        dashboardChart.destroy();
        dashboardChart = null;
      }
      return;
    }
    
    renderDashboardChart(data.data, data.aggregation);
    
  } catch (error) {
    dashboardElements.loading.classList.add("hidden");
    showToast(`Ошибка загрузки дашборда: ${error.message}`, "error");
  }
}

function renderDashboardChart(data, aggregation) {
  const ctx = dashboardElements.chart.getContext("2d");
  
  // Prepare data - only include points with activity (no zeros)
  const labels = data.map(point => {
    const date = new Date(point.time);
    if (aggregation === "daily") {
      return date.toLocaleDateString("ru-RU", { day: "numeric", month: "short" });
    } else {
      return date.toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });
    }
  });
  
  const activeUsers = data.map(point => point.active_users);
  const totalActions = data.map(point => point.total_actions);
  
  // Destroy existing chart
  if (dashboardChart) {
    dashboardChart.destroy();
  }
  
  // Create new chart with spanGaps: false to show gaps where there's no data
  dashboardChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Active Users",
          data: activeUsers,
          borderColor: "#6366f1",
          backgroundColor: "rgba(99, 102, 241, 0.1)",
          borderWidth: 2,
          fill: true,
          tension: 0.3,
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: "#6366f1",
          spanGaps: false,
        },
        {
          label: "Total Actions",
          data: totalActions,
          borderColor: "#22c55e",
          backgroundColor: "rgba(34, 197, 94, 0.1)",
          borderWidth: 2,
          fill: true,
          tension: 0.3,
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: "#22c55e",
          spanGaps: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        intersect: false,
        mode: "index",
      },
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          backgroundColor: "rgba(0, 0, 0, 0.8)",
          titleColor: "#fff",
          bodyColor: "#fff",
          padding: 12,
          cornerRadius: 8,
          displayColors: true,
          callbacks: {
            title: function(context) {
              const idx = context[0].dataIndex;
              const point = data[idx];
              const date = new Date(point.time);
              if (aggregation === "daily") {
                return date.toLocaleDateString("ru-RU", { 
                  weekday: "short", 
                  day: "numeric", 
                  month: "long" 
                });
              } else {
                return date.toLocaleString("ru-RU", { 
                  day: "numeric", 
                  month: "short", 
                  hour: "2-digit", 
                  minute: "2-digit" 
                });
              }
            },
            label: function(context) {
              const label = context.dataset.label;
              const value = context.parsed.y;
              if (label === "Active Users") {
                return ` Активных пользователей: ${value}`;
              } else {
                return ` Генераций: ${value}`;
              }
            },
          },
        },
      },
      scales: {
        x: {
          grid: {
            display: false,
          },
          ticks: {
            color: "#888",
            maxRotation: 45,
            minRotation: 0,
          },
        },
        y: {
          beginAtZero: true,
          grid: {
            color: "rgba(136, 136, 136, 0.1)",
          },
          ticks: {
            color: "#888",
            stepSize: 1,
            callback: function(value) {
              if (Number.isInteger(value)) {
                return value;
              }
            },
          },
        },
      },
    },
  });
}

// Dashboard event listeners
dashboardElements.btn.addEventListener("click", openDashboardModal);
dashboardElements.close.addEventListener("click", closeDashboardModal);
dashboardElements.overlay.addEventListener("click", (e) => {
  if (e.target === dashboardElements.overlay) closeDashboardModal();
});

dashboardElements.filters.forEach(btn => {
  btn.addEventListener("click", () => {
    const period = btn.dataset.period;
    if (period !== currentPeriod) {
      fetchDashboardData(period);
    }
  });
});

// Init
fetchStats();
fetchUsers({ reset: true });
startAutoRefresh();
