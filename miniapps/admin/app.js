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
    return name.includes(query) || id.includes(query) || phone.includes(query);
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

// Keyboard handling
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (!elements.confirmOverlay.classList.contains("hidden")) {
      closeConfirm();
    } else if (!elements.modalOverlay.classList.contains("hidden")) {
      closeModal();
    }
  }
});

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

// Init
fetchStats();
fetchUsers({ reset: true });
startAutoRefresh();
