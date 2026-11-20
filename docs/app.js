const el = id => document.getElementById(id);
const picker = el("case");
const toggles = [...document.querySelectorAll("input")];
const labels = ["Retina", "Transcript", "EHR"];

function render(experiment) {
  const record = experiment.cases[picker.selectedIndex];
  const key = toggles.map(input => Number(input.checked)).join("");
  const result = record.subsets[key];
  el("score").textContent = result ? result.score.toFixed(3) : "Abstained";
  el("weights").replaceChildren();
  if (result) {
    result.weights.forEach((weight, i) => {
      const row = document.createElement("div");
      const meter = document.createElement("meter");
      meter.value = weight;
      meter.setAttribute("aria-label", labels[i]);
      row.append(meter, " " + labels[i] + " " + (weight * 100).toFixed(0) + "%");
      el("weights").append(row);
    });
  } else el("weights").textContent = "Select at least one input to score this case.";
  const canvas = el("image");
  canvas.hidden = !toggles[0].checked;
  const ctx = canvas.getContext("2d");
  record.image.forEach((row, y) => row.forEach((pixel, x) => {
    const v = Math.round(Math.max(0, Math.min(255, 128 + 55 * pixel)));
    ctx.fillStyle = "rgb(" + [v,v,v].join(",") + ")";
    ctx.fillRect(x, y, 1, 1);
  }));
  el("speech").textContent = toggles[1].checked ? "Transcript · " + record.tokens : "Transcript withheld";
  el("visits").replaceChildren();
  if (toggles[2].checked) record.visits.forEach(visit => {
    const row = el("visits").insertRow();
    visit.forEach(value => row.insertCell().textContent = value.toFixed(2));
  });
  const metrics = experiment.metrics[key];
  el("metric").textContent = metrics ? "At this input combination, held-out accuracy was " +
    (100 * metrics.accuracy).toFixed(1) + "% and Brier score was " + metrics.brier.toFixed(3) + "." : "";
}

async function start() {
  const response = await fetch("experiment.json");
  if (!response.ok) throw Error("Experiment unavailable");
  const data = await response.json();
  data.cases.forEach(c => picker.add(new Option(c.id, c.id)));
  picker.disabled = false;
  el("run").textContent = data.train_records + " generated records across " + data.train_sites +
    " simulated training sites, " + data.rounds + " FedAvg rounds and " + data.test_records +
    " independent synthetic test records.";
  [picker, ...toggles].forEach(control => control.addEventListener("change", () => render(data)));
  render(data);
}
start().catch(() => { el("score").textContent = "Unavailable"; el("weights").textContent = "Serve this folder over HTTP with experiment.json present."; });
