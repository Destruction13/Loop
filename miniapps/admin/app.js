const state = {
  items: [],
  offset: 0,
  limit: 50,
  sort: "generations",
  order: "desc",
  total: 0,
  loading: false,
};

const elements = {
  tableBody: document.getElementById("tableBody"),
  loadMore: document.getElementById("loadMore"),
  refresh: document.getElementById("refresh"),
  sort: document.getElementById("sort"),
  limit: document.getElementById("limit"),
  orderToggle: document.getElementById("orderToggle"),
  statusDot: document.getElementById("statusDot"),
  statusText: document.getElementById("statusText"),
  count: document.getElementById("count"),
  stateBox: document.getElementById("state"),
  stateTitle: document.getElementById("stateTitle"),
  stateBody: document.getElementById("stateBody"),
};

const telegram = window.Telegram && window.Telegram.WebApp;
if (telegram) {
  telegram.ready();
  telegram.expand();
}

const initData = telegram ? telegram.initData : "";
const apiBase = (() => {
  const raw = window.ADMIN_CONFIG && window.ADMIN_CONFIG.apiBaseUrl;
  if (!raw) {
    return "";
  }
  return raw.replace(/\/$/, "");
})();

function buildUrl() {
  const params = new URLSearchParams({
    offset: String(state.offset),
    limit: String(state.limit),
    sort: state.sort,
    order: state.order,
  });
  // Use absolute path to avoid issues with <base href>
  // If apiBase is set, use it; otherwise use absolute path from root
  if (apiBase) {
    return `${apiBase}/admin/api/users?${params.toString()}`;
  }
  return `/admin/api/users?${params.toString()}`;
}

function setStatus(text, mode) {
  elements.statusText.textContent = text;
  elements.statusDot.className = `status-dot ${mode}`;
}

function setState(visible, title, body) {
  if (visible) {
    elements.stateTitle.textContent = title;
    elements.stateBody.textContent = body;
    elements.stateBox.classList.remove("hidden");
  } else {
    elements.stateBox.classList.add("hidden");
  }
}

function formatName(row) {
  if (row.username) {
    return `@${row.username}`;
  }
  if (row.full_name) {
    return row.full_name;
  }
  return `User ${row.user_id}`;
}

function renderTable() {
  elements.tableBody.innerHTML = "";
  if (!state.items.length) {
    return;
  }
  for (const row of state.items) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${formatName(row)}</td>
      <td><a href="${row.telegram_link}" target="_blank" rel="noopener">Profile</a></td>
      <td class="num">${row.generations}</td>
      <td class="num">${row.site_clicks}</td>
      <td class="num">${row.social_clicks}</td>
      <td>${row.phone || "-"}</td>
    `;
    elements.tableBody.appendChild(tr);
  }
}

function updateFooter() {
  const shown = state.items.length;
  const total = state.total;
  elements.count.textContent = total
    ? `${shown} of ${total} users`
    : `${shown} users`;
  elements.loadMore.disabled = shown >= total;
}

async function fetchUsers({ reset = false } = {}) {
  if (state.loading) {
    return;
  }
  if (!initData) {
    setStatus("Init data missing", "error");
    setState(
      true,
      "Open inside Telegram",
      "This dashboard requires Telegram initData to access the API."
    );
    return;
  }
  state.loading = true;
  setStatus("Loading...", "loading");
  if (reset) {
    state.offset = 0;
    state.items = [];
    renderTable();
  }
  try {
    const response = await fetch(buildUrl(), {
      headers: {
        "X-Telegram-Init-Data": initData,
      },
    });
    if (!response.ok) {
      const text = response.status === 403 ? "Access denied" : "Request failed";
      throw new Error(text);
    }
    const data = await response.json();
    state.total = data.total || 0;
    if (Array.isArray(data.items)) {
      if (reset) {
        state.items = data.items;
      } else {
        state.items = state.items.concat(data.items);
      }
      state.offset = state.items.length;
    }
    renderTable();
    updateFooter();
    if (!state.items.length) {
      setState(true, "No users yet", "The database did not return any users.");
      setStatus("Empty", "idle");
    } else {
      setState(false);
      setStatus("Up to date", "ready");
    }
  } catch (error) {
    setStatus("Error", "error");
    setState(
      true,
      "Unable to load data",
      error.message || "Check API availability and admin access."
    );
  } finally {
    state.loading = false;
  }
}

elements.loadMore.addEventListener("click", () => {
  fetchUsers();
});

elements.refresh.addEventListener("click", () => {
  fetchUsers({ reset: true });
});

elements.sort.addEventListener("change", (event) => {
  state.sort = event.target.value;
  fetchUsers({ reset: true });
});

elements.limit.addEventListener("change", (event) => {
  state.limit = Number(event.target.value);
  fetchUsers({ reset: true });
});

elements.orderToggle.addEventListener("click", () => {
  state.order = state.order === "desc" ? "asc" : "desc";
  elements.orderToggle.textContent =
    state.order === "desc" ? "Descending" : "Ascending";
  fetchUsers({ reset: true });
});

setState(true, "Loading...", "Fetching data from the API.");
fetchUsers({ reset: true });
