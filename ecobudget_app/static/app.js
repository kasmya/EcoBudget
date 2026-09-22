const $ = (id) => document.getElementById(id);
const CAT_COLORS = {
  "Transport": "#60a5fa", "Food": "#f87171", "Home energy": "#fbbf24",
  "Waste": "#34d399", "Goods": "#a78bfa",
};
let FACTORS = {};            // category -> [factor rows]
let catChart, dayChart;
const state = { month: new Date().toISOString().slice(0, 7) };

const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok && r.status !== 204) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.status === 204 ? null : r.json();
};
const fmt = (n, d = 1) => Number(n).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
function toast(msg) {
  const t = $("toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), 2200);
}

/* ---------- factors + form ---------- */
async function loadFactors() {
  const data = await api("/api/factors");
  FACTORS = data.factors;
  const cat = $("fCategory");
  cat.innerHTML = data.categories.map(c => `<option>${c}</option>`).join("");
  cat.onchange = fillActivities;
  fillActivities();
}
function fillActivities() {
  const c = $("fCategory").value;
  const act = $("fActivity");
  act.innerHTML = FACTORS[c].map(f => `<option value="${f.activity}">${f.activity}</option>`).join("");
  act.onchange = updateHint;
  updateHint();
}
function currentFactor() {
  const c = $("fCategory").value, a = $("fActivity").value;
  return (FACTORS[c] || []).find(f => f.activity === a);
}
function updateHint() {
  const f = currentFactor();
  if (!f) return;
  $("fHint").textContent = `Amount in ${f.unit} - factor ${f.kg_co2e_per_unit} kg CO₂e / ${f.unit}`;
  updateEstimate();
}
function updateEstimate() {
  const f = currentFactor(); const q = parseFloat($("fQuantity").value);
  $("estimate").innerHTML = (f && q > 0) ? `= <b>${fmt(q * f.kg_co2e_per_unit, 2)} kg CO₂e</b>` : "";
}
$("fQuantity").addEventListener("input", updateEstimate);

async function addEntry() {
  const f = currentFactor(); const q = parseFloat($("fQuantity").value);
  if (!f || !(q > 0)) { toast("Enter an amount greater than 0"); return; }
  await api("/api/entries", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ activity: f.activity, quantity: q, entry_date: $("fDate").value || undefined, note: $("fNote").value || undefined }),
  });
  toast(`Logged ${f.activity}: ${fmt(q * f.kg_co2e_per_unit, 2)} kg CO₂e`);
  $("fQuantity").value = ""; $("fNote").value = ""; updateEstimate();
  await refresh();
}

/* ---------- summary + charts ---------- */
async function loadSummary() {
  const s = await api(`/api/summary?month=${state.month}`);
  $("kTotal").textContent = fmt(s.total_kg, 1);
  $("kEntries").textContent = `${s.entry_count} entries logged`;
  $("kBudget").textContent = fmt(s.budget_kg, 0);
  $("kBasis").textContent = s.budget_basis.split("(")[0].trim();
  $("kRemain").textContent = fmt(s.remaining_kg, 1);
  $("budgetInput").value = Math.round(s.budget_kg);
  const pill = $("pctPill");
  pill.textContent = `${s.pct_of_budget}% of budget`;
  pill.className = "pill " + (s.over_budget ? "over" : "ok");
  $("kStatus").innerHTML = s.over_budget
    ? `<span style="color:var(--danger)">over budget</span>`
    : `<span style="color:var(--accent)">on track</span>`;
  const fill = $("progFill");
  fill.style.width = Math.min(100, s.pct_of_budget) + "%";
  fill.className = "fill" + (s.over_budget ? " over" : "");
  $("legLeft").textContent = `${fmt(s.total_kg, 1)} kg logged`;
  $("legRight").textContent = `budget ${fmt(s.budget_kg, 0)} kg`;

  drawCategory(s.by_category);
  drawDay(s.by_day);
  drawCatTable(s.by_category, s.total_kg);
  drawTop(s.top_activities);
  $("catSub").textContent = `${s.by_category.length} categories`;
}

function drawCategory(rows) {
  const ctx = $("catChart");
  if (catChart) catChart.destroy();
  if (!rows.length) { ctx.getContext("2d").clearRect(0, 0, ctx.width, ctx.height); return; }
  catChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: rows.map(r => r.category),
      datasets: [{ data: rows.map(r => r.kg), backgroundColor: rows.map(r => CAT_COLORS[r.category] || "#888"),
        borderColor: "#161d19", borderWidth: 3 }],
    },
    options: { plugins: { legend: { position: "right", labels: { color: "#cfe0d7", boxWidth: 12, padding: 12 } } },
      cutout: "62%", responsive: true, maintainAspectRatio: false },
  });
}
function drawDay(rows) {
  const ctx = $("dayChart");
  if (dayChart) dayChart.destroy();
  dayChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: rows.map(r => r.day.slice(5)),
      datasets: [{ label: "kg CO₂e", data: rows.map(r => r.kg),
        backgroundColor: "rgba(74,222,128,.65)", borderRadius: 6, maxBarThickness: 42 }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { x: { grid: { display: false }, ticks: { color: "#9db3a8" } },
        y: { grid: { color: "#26332c" }, ticks: { color: "#9db3a8" }, beginAtZero: true } },
      responsive: true, maintainAspectRatio: false,
    },
  });
}
function drawCatTable(rows, total) {
  $("catTable").innerHTML = rows.map(r => {
    const pct = total ? (100 * r.kg / total) : 0;
    return `<tr>
      <td><span class="cat-${r.category.replace(/ /g, ' ')}">●</span> ${r.category}</td>
      <td class="num bar-cell"><div class="b" style="width:${pct}%"></div><span>${fmt(r.kg, 1)}</span></td>
      <td class="num">${pct.toFixed(0)}%</td><td class="num">${r.n}</td></tr>`;
  }).join("") || `<tr><td colspan="4" class="empty">No entries yet</td></tr>`;
}
function drawTop(rows) {
  $("topTable").innerHTML = rows.map(r =>
    `<tr><td>${r.activity}</td><td><span class="tag">${r.category}</span></td><td class="num">${fmt(r.kg, 1)}</td></tr>`
  ).join("") || `<tr><td colspan="3" class="empty">No entries yet</td></tr>`;
}

/* ---------- entries ---------- */
async function loadEntries() {
  const d = await api(`/api/entries?month=${state.month}`);
  $("entriesSub").textContent = `${d.entries.length} entries in ${state.month}`;
  $("entriesTable").innerHTML = d.entries.map(e => `<tr>
    <td>${e.entry_date}</td><td>${e.activity}</td><td><span class="tag">${e.category}</span></td>
    <td class="num">${e.quantity} ${e.unit}</td><td class="num">${fmt(e.kg_co2e, 2)}</td>
    <td>${e.note || ""}</td>
    <td class="num"><button class="tiny" data-del="${e.id}">Delete</button></td></tr>`).join("")
    || `<tr><td colspan="7" class="empty">No entries this month. Log one above.</td></tr>`;
  $("entriesTable").querySelectorAll("[data-del]").forEach(b =>
    b.onclick = async () => { await api(`/api/entries/${b.dataset.del}`, { method: "DELETE" }); toast("Entry deleted"); refresh(); });
}

/* ---------- benchmarks ---------- */
async function loadBenchmarks() {
  const d = await api(`/api/benchmarks?month=${state.month}`);
  $("kAnnual").textContent = fmt(d.your_projected_annual_tonnes, 2);
  const you = { country: "You (projected)", co2_per_capita_tonnes: d.your_projected_annual_tonnes, you: true };
  const rows = [...d.countries, you].sort((a, b) => b.co2_per_capita_tonnes - a.co2_per_capita_tonnes);
  const max = Math.max(...rows.map(r => r.co2_per_capita_tonnes), 1);
  $("benchList").innerHTML = rows.map(r => `
    <div class="bench-row ${r.you ? "you" : ""}">
      <div>${r.country}</div>
      <div class="track"><div class="f" style="width:${Math.max(2, 100 * r.co2_per_capita_tonnes / max)}%"></div></div>
      <div class="num" style="text-align:right">${fmt(r.co2_per_capita_tonnes, 2)} t</div>
    </div>`).join("");
}

/* ---------- orchestration ---------- */
async function refresh() {
  await Promise.all([loadSummary(), loadEntries(), loadBenchmarks()]);
}
async function saveBudget() {
  const v = parseFloat($("budgetInput").value);
  if (!(v > 0)) { toast("Budget must be positive"); return; }
  await api("/api/budget", { method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ monthly_kg_co2e: v }) });
  toast("Budget updated"); refresh();
}

function init() {
  $("month").value = state.month;
  $("fDate").value = new Date().toISOString().slice(0, 10);
  $("month").onchange = () => { state.month = $("month").value; refresh(); };
  $("addBtn").onclick = addEntry;
  $("saveBudget").onclick = saveBudget;
  loadFactors().then(refresh);
}
init();
