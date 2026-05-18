const form = document.querySelector("#profile-form");
const state = document.querySelector("#state");
const cards = document.querySelector("#cards");
const dialog = document.querySelector("#trial-dialog");
const closeDialog = document.querySelector("#close-dialog");

let latestProfile = null;

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(form);
  latestProfile = {
    age: Number(data.get("age")) || null,
    sex: data.get("sex") || null,
    condition: data.get("condition"),
    location: data.get("location") || null,
    intervention_preferences: splitList(data.get("interventions")),
    phase_preferences: [],
    notes: data.get("notes") || null,
  };

  state.textContent = "Retrieving trials and generating explanations...";
  state.hidden = false;
  cards.innerHTML = "";

  try {
    const response = await fetch("/api/recommendations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(latestProfile),
    });

    if (!response.ok) {
      throw new Error("Recommendation service returned an error.");
    }

    const payload = await response.json();
    renderResults(payload.results);
  } catch (error) {
    state.textContent = "Something went wrong while generating recommendations. Please try again.";
  }
});

closeDialog.addEventListener("click", () => dialog.close());

function splitList(value) {
  return String(value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function renderResults(results) {
  if (!results.length) {
    state.textContent = "No strong matches were found. Try broadening the condition or location.";
    return;
  }

  state.hidden = true;
  cards.innerHTML = results.map(cardTemplate).join("");
  document.querySelectorAll("[data-detail]").forEach((button) => {
    button.addEventListener("click", () => {
      const index = Number(button.dataset.detail);
      showDetail(results[index]);
    });
  });

  document.querySelectorAll("[data-ask]").forEach((button) => {
    button.addEventListener("click", async () => {
      const index = Number(button.dataset.ask);
      await askAssistant(results[index]);
    });
  });
}

function cardTemplate(item, index) {
  const trial = item.trial;
  const score = Math.round(item.score * 100);
  const pills = [...trial.conditions.slice(0, 2), trial.status, ...trial.phases.slice(0, 1)]
    .filter(Boolean)
    .map((label) => `<span class="pill">${escapeHtml(label)}</span>`)
    .join("");

  return `
    <article class="card">
      <div class="card-head">
        <div>
          <p class="eyebrow">${escapeHtml(trial.nct_id)}</p>
          <h3>${escapeHtml(trial.title)}</h3>
          <div class="pill-row">${pills}</div>
        </div>
        <div>
          <div class="score">${score}%</div>
          <p class="score-text">match</p>
        </div>
      </div>
      <p>${escapeHtml(item.explanation.patient_friendly_summary)}</p>
      <div class="insight">
        <strong>Why this matched</strong>
        <p>${escapeHtml(item.explanation.ranking_rationale)}</p>
      </div>
      <p class="meta">${escapeHtml((trial.locations || []).slice(0, 2).join(" | ") || "Location not listed")}</p>
      <div class="card-actions">
        <button class="secondary" data-detail="${index}" type="button">View details</button>
        <button data-ask="${index}" type="button">Explain eligibility</button>
      </div>
    </article>
  `;
}

function showDetail(item, assistantAnswer = "", assistantProvider = "") {
  const trial = item.trial;
  document.querySelector("#dialog-nct").textContent = trial.nct_id;
  document.querySelector("#dialog-title").textContent = trial.title;
  document.querySelector("#dialog-body").innerHTML = `
    <div>
      <strong>Patient-friendly summary</strong>
      <p>${escapeHtml(item.explanation.patient_friendly_summary)}</p>
    </div>
    <div>
      <strong>Eligibility signals</strong>
      <p>${escapeHtml(item.explanation.eligibility_notes.join(" ") || "No structured eligibility signals were available.")}</p>
    </div>
    <div>
      <strong>Trial data</strong>
      <p>${escapeHtml(trial.eligibility_criteria || "Eligibility criteria are not listed in this record.")}</p>
    </div>
    ${
      assistantAnswer
        ? `<div class="insight">
            <div class="assistant-head">
              <strong>AI assistant</strong>
              ${assistantProvider ? `<span class="provider-badge">${providerLabel(assistantProvider)}</span>` : ""}
            </div>
            <div class="assistant-answer">${formatAssistantText(assistantAnswer)}</div>
          </div>`
        : ""
    }
    <a href="${escapeHtml(trial.source_url || "#")}" target="_blank" rel="noreferrer">Open source record</a>
  `;
  dialog.showModal();
}

async function askAssistant(item) {
  showDetail(item, "Generating a grounded eligibility explanation...", "pending");
  const response = await fetch("/api/assistant/answer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: "Explain the main eligibility considerations for this patient in plain language.",
      trial: item.trial,
      patient_profile: latestProfile,
    }),
  });
  const payload = await response.json();
  showDetail(item, payload.answer, payload.provider);
}

function providerLabel(provider) {
  if (provider === "openai") {
    return "Answered by OpenAI";
  }
  if (provider === "pending") {
    return "Generating";
  }
  return "Local fallback";
}

function formatAssistantText(value) {
  const lines = String(value ?? "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  const html = [];
  let inList = false;

  for (const line of lines) {
    if (line.startsWith("### ")) {
      if (inList) {
        html.push("</ul>");
        inList = false;
      }
      html.push(`<h4>${escapeHtml(line.slice(4))}</h4>`);
      continue;
    }

    if (line.startsWith("- ")) {
      if (!inList) {
        html.push("<ul>");
        inList = true;
      }
      html.push(`<li>${formatInlineMarkdown(line.slice(2))}</li>`);
      continue;
    }

    if (inList) {
      html.push("</ul>");
      inList = false;
    }
    html.push(`<p>${formatInlineMarkdown(line)}</p>`);
  }

  if (inList) {
    html.push("</ul>");
  }

  return html.join("");
}

function formatInlineMarkdown(value) {
  return escapeHtml(value).replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
