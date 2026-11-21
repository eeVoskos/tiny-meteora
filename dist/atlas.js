import { SceneAnimation } from "./scene-life.js";
import { categories, isLanguage, placeKinds, translate, translatePage } from "./i18n.js";

const $ = id => document.getElementById(id);
const viewport = $("viewport");
const search = $("search");
const smallScreen = matchMedia("(max-width: 700px)");
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
const categoryIcons = { monastery: "temple", town: "pin", viewpoint: "eye", civic: "square" };
const browseOrder = ["town", "monastery", "viewpoint", "civic"];
// Shared links encode these layers in their original order.
const layerDefaults = { monastery: true, town: true, viewpoint: false, trails: false, civic: false, life: false };
const layers = { ...layerDefaults };
const lightingChoices = [...document.querySelectorAll('input[name="lighting"]')];
const languageButtons = [...document.querySelectorAll("[data-language]")];
const languageKey = "tiny-meteora-language";
const timesOfDay = new Set(lightingChoices.map(input => input.value));
const icons = {
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5m0-8h.01"/>',
  share: '<path d="M12 15V3m-4 4 4-4 4 4m3 5v7a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2v-7"/>',
  search: '<circle cx="10.5" cy="10.5" r="5.5"/><path d="m15 15 4 4"/>',
  temple: '<path d="M3 10h18L12 4 3 10Zm2 9h14M7 10v9m5-9v9m5-9v9M3 21h18"/>',
  church: '<path d="M12 2v6m-2-4h4M5 12l7-5 7 5v9H5zM10 21v-4a2 2 0 0 1 4 0v4M3 21h18"/>',
  pin: '<path d="M19 10c0 5-7 10-7 10s-7-5-7-10a7 7 0 1 1 14 0Z"/><circle cx="12" cy="10" r="2"/>',
  trail: '<path d="m2 17 4-7 2 3 4-7 4 7 2-3 4 7M4 21h3m3 0h3m3 0h3"/>',
  eye: '<path d="M2 12s4-6 10-6 10 6 10 6-4 6-10 6S2 12 2 12Z"/><circle cx="12" cy="12" r="2.5"/>',
  square: '<rect x="4" y="4" width="16" height="16" rx="2"/><circle cx="12" cy="12" r="3"/>',
  bird: '<path d="M2 8c4 .2 7 2.3 10 6 3-3.7 6-5.8 10-6M9.5 12l2.5 4 2.5-4"/>',
  sunrise: '<path d="M3 18h18M7 18a5 5 0 0 1 10 0M12 3v7m-3-4 3-3 3 3M4 12l2 2m14-2-2 2"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v3m0 14v3M2 12h3m14 0h3M5 5l2 2m10 10 2 2M5 19l2-2M17 7l2-2"/>',
  sunset: '<path d="M3 18h18M7 18a5 5 0 0 1 10 0M12 3v7m-3-3 3 3 3-3M4 12l2 2m14-2-2 2"/>',
  moon: '<path d="M19.8 15.3A8.5 8.5 0 0 1 8.7 4.2a8.6 8.6 0 1 0 11.1 11.1Z"/><path d="M16 4v4m-2-2h4"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  minus: '<path d="M5 12h14"/>',
  home: '<path d="M4 11 12 4l8 7M6 10v10h12V10m-8 10v-5h4v5"/>',
  chevron: '<path d="m9 5 7 7-7 7"/>',
  close: '<path d="m6 6 12 12M18 6 6 18"/>',
};
function icon(name) {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name]}</svg>`;
}

for (const element of document.querySelectorAll("[data-icon]")) {
  element.insertAdjacentHTML("afterbegin", icon(element.dataset.icon));
}

let config;
let sceneLife;
let language = "en";
let timeOfDay = "day";
let selected = null;
let frame = 0;
let motionFrame = 0;
let toastTimer;
let lastPlaceControl = null;
const view = { x: 0, y: 0, z: 1, width: 0, height: 0 };
const markerButtons = new Map();
const visibleTiles = new Map();
const pointers = new Map();
const maxZoom = 4;
const clamp = (value, lower, upper) => Math.min(upper, Math.max(lower, value));
const fold = text => text.normalize("NFD").replace(/\p{M}/gu, "").toLocaleLowerCase().replaceAll("ς", "σ");
const t = key => translate(key, language);
const placeName = place => language === "el" ? place.greek : place.name;
const placeSummary = place => language === "el" ? place.summary_el : place.summary;
const placeKind = place => language === "el"
  ? (placeKinds[place.kind] ?? categories[place.type][1])
  : (place.kind ?? categories[place.type][0]);

function savedLanguage() {
  try {
    const saved = localStorage.getItem(languageKey);
    if (isLanguage(saved)) return saved;
  } catch {
    // The map can still switch languages when browser storage is unavailable.
  }
  return "en";
}

function setLanguage(next) {
  language = next;
  translatePage(language);
  for (const button of languageButtons) {
    button.setAttribute("aria-pressed", button.dataset.language === language);
  }
  if (!config) return;
  for (const place of config.places) updateMarkerLabel(markerButtons.get(place.id), place);
  updateSelection();
}

function chooseLanguage(next) {
  if (next === language) return;
  try {
    localStorage.setItem(languageKey, next);
  } catch {
    // The current view still uses this choice.
  }
  if (location.hash) {
    const address = new URL(location.href);
    const params = new URLSearchParams(address.hash.slice(1));
    params.set("lang", next);
    address.hash = params;
    history.replaceState(null, "", address);
  }
  setLanguage(next);
}

function minZoom() {
  // Cover both axes so the viewport never extends beyond the map.
  return Math.max(view.width / config.width, view.height / config.height);
}

function clampZoom(zoom) {
  const minimum = minZoom();
  return clamp(zoom, minimum, Math.max(minimum, maxZoom));
}

function homeView() {
  const z = minZoom();
  const preferred = config.width * (smallScreen.matches ? .57 : .51);
  if (!z) return { x: preferred, y: config.height / 2, z };
  const peaks = config.places.filter(place => place.type === "monastery").map(place => place.x);
  const radius = Math.max(0, view.width/2 - 20) / z;
  const lower = Math.max(...peaks) - radius;
  const upper = Math.min(...peaks) + radius;
  return { x: clamp(preferred, Math.min(lower, upper), Math.max(lower, upper)), y: config.height / 2, z };
}

function constrainAxis(center, extent, viewportSize) {
  const radius = viewportSize / view.z / 2;
  return radius * 2 >= extent ? extent / 2 : clamp(center, radius, extent - radius);
}

function positionView(x, y, z = view.z) {
  view.z = clampZoom(z);
  view.x = constrainAxis(x, config.width, view.width);
  view.y = constrainAxis(y, config.height, view.height);
  if (!frame) frame = requestAnimationFrame(renderView);
}

function measure() {
  const wasMeasured = view.width > 0;
  const oldHome = wasMeasured ? homeView() : null;
  const wasHomeZoom = !oldHome || Math.abs(view.z - oldHome.z) < 0.005;
  view.width = viewport.clientWidth;
  view.height = viewport.clientHeight;
  const home = homeView();
  if (!wasMeasured) positionView(home.x, home.y, home.z);
  else positionView(view.x, view.y, wasHomeZoom ? home.z : view.z);
}

function showTiles(left, top) {
  const { width, height, levels, tileSize } = config;
  const ratio = Math.min(devicePixelRatio || 1, 2);
  const level = levels.findLast(factor => factor * view.z * ratio <= 1.1) ?? levels[0];
  const span = tileSize * level;
  const firstX = Math.max(0, Math.floor(-left / (view.z * span)));
  const firstY = Math.max(0, Math.floor(-top / (view.z * span)));
  const lastX = Math.min(Math.ceil(width / span) - 1, Math.floor((view.width - left) / (view.z * span)));
  const lastY = Math.min(Math.ceil(height / span) - 1, Math.floor((view.height - top) / (view.z * span)));
  const wanted = new Set();
  for (let y = firstY; y <= lastY; y++) {
    for (let x = firstX; x <= lastX; x++) {
      const key = `${level}/${x}-${y}`;
      wanted.add(key);
      if (visibleTiles.has(key)) continue;
      const tile = document.createElement("img");
      tile.alt = "";
      tile.draggable = false;
      tile.decoding = "async";
      Object.assign(tile.style, {
        left: `${span * x}px`,
        top: `${span * y}px`,
        width: `${Math.min(span, width - span * x)}px`,
        height: `${Math.min(span, height - span * y)}px`,
      });
      tile.src = `./assets/scan-tiles/${key}.webp`;
      $("terrain-tiles").append(tile);
      visibleTiles.set(key, tile);
    }
  }
  for (const [key, tile] of visibleTiles) {
    if (wanted.has(key)) continue;
    tile.remove();
    visibleTiles.delete(key);
  }
}

function renderView() {
  frame = 0;
  const left = view.width / 2 - view.x * view.z;
  const top = view.height / 2 - view.y * view.z;
  $("world").style.transform = `translate(${left}px, ${top}px) scale(${view.z})`;
  showTiles(left, top);

  const visibleMarkers = [];
  for (const place of config.places) {
    const button = markerButtons.get(place.id);
    const x = left + place.x * view.z;
    const y = top + place.y * view.z;
    button.hidden = (!layers[place.type] && selected !== place) ||
      x < 10 || x > view.width - 10 || y < 20 || y > view.height - 3;
    if (button.hidden) continue;
    button.style.left = `${x}px`;
    button.style.top = `${y}px`;
    if (place.featured || place.type === "monastery") visibleMarkers.push({ place, button, x, y });
  }
  positionLabels(visibleMarkers);

  const edgeX = clamp(-left / (view.z * config.width), 0, 1);
  const edgeY = clamp(-top / (view.z * config.height), 0, 1);
  Object.assign($("minimap-box").style, {
    left: `${edgeX * 100}%`,
    top: `${edgeY * 100}%`,
    width: `${Math.min(1 - edgeX, view.width / view.z / config.width) * 100}%`,
    height: `${Math.min(1 - edgeY, view.height / view.z / config.height) * 100}%`,
  });
  const distances = [20, 50, 100, 200, 500, 1000, 2000];
  const scale = distances.find(distance => distance * view.z / config.metersPerPixel >= 45) ?? distances.at(-1);
  const scaleBar = $("scale").firstElementChild;
  const measurement = scale < 1000 ? scale : scale / 1000;
  scaleBar.textContent = `${measurement.toLocaleString(language)} ${t(scale < 1000 ? "meters" : "kilometers")}`;
  scaleBar.style.width = `${scale * view.z / config.metersPerPixel}px`;
  $("zoom-in").disabled = view.z >= maxZoom;
  $("zoom-out").disabled = view.z <= minZoom() + 1e-9;
  sceneLife?.redraw();
}

function positionLabels(markers) {
  const monasteries = markers.filter(({place}) => place.type === "monastery");
  const occupied = monasteries.map(({x, y}) => ({
    left: x - 13, top: y - 24, right: x + 13, bottom: y + 2,
  }));
  const overlap = (a, b) => Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left) + 3) *
    Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) + 3);

  const sized = monasteries.map(marker => ({...marker,
    width: marker.button.lastElementChild.offsetWidth, height: marker.button.lastElementChild.offsetHeight,
  }));
  for (const {button, x, y, width, height} of sized) {
    const positions = [[x - width/2, y - 29 - height], [x + 16, y - 19 - height/2],
      [x - width - 16, y - 19 - height/2], [x - width/2, y + 5],
      [x - width/2, y - 31 - 2*height], [x + 15, y + 8], [x - width - 15, y + 8]];
    let best, lowest = Infinity;
    for (const [px, py] of positions) {
      const left = clamp(px, 5, Math.max(5, view.width - width - 5));
      const top = clamp(py, 5, Math.max(5, view.height - height - 5));
      const box = { left, top, right: left + width, bottom: top + height };
      const score = occupied.reduce((sum, other) => sum + overlap(box, other), 0);
      if (score < lowest) {
        best = box;
        lowest = score;
      }
      if (!score) break;
    }
    button.style.setProperty("--label-left", `${best.left - x}px`);
    button.style.setProperty("--label-bottom", `${y - best.bottom}px`);
    occupied.push(best);
  }

  for (const {place, button, x, y} of markers) {
    if (place.type === "monastery") continue;
    const box = { left: x - 65, top: y - 56, right: x + 65, bottom: y - 30 };
    const fits = box.left > 0 && box.right < view.width && box.top > 0 &&
      !occupied.some(other => overlap(box, other));
    button.classList.toggle("crowded", !fits);
    if (fits) occupied.push(box);
  }
}

function stopAnimation() {
  cancelAnimationFrame(motionFrame);
  motionFrame = 0;
}

function flyTo(target) {
  stopAnimation();
  if (reduceMotion.matches) {
    positionView(target.x, target.y, target.z);
    return;
  }
  const start = { ...view };
  const from = performance.now();
  function step(now) {
    const elapsed = clamp((now - from) / 380, 0, 1);
    const t = 1 - (1 - elapsed) ** 3;
    positionView(start.x + (target.x - start.x) * t,
      start.y + (target.y - start.y) * t, start.z + (target.z - start.z) * t);
    motionFrame = elapsed < 1 ? requestAnimationFrame(step) : 0;
  }
  motionFrame = requestAnimationFrame(step);
}

function zoomAt(factor, x = view.width / 2, y = view.height / 2) {
  stopAnimation();
  const z = clampZoom(view.z * factor);
  const ratio = (1 / view.z - 1 / z);
  positionView(view.x + (x - view.width / 2) * ratio, view.y + (y - view.height / 2) * ratio, z);
}

function toggleBrowse(open, moveFocus = false) {
  if (!smallScreen.matches) return;
  $("sidebar").classList.toggle("open", open);
  $("browse").setAttribute("aria-expanded", open);
  if (moveFocus) (open ? search : $("browse")).focus();
}

function isChurch(place) {
  return place.kind === "Church" || place.kind === "Chapel";
}

function placeIcon(place) {
  return icon(isChurch(place) ? "church" : categoryIcons[place.type]);
}

function updateMarkerLabel(button, place) {
  const name = placeName(place);
  button.setAttribute("aria-label", name);
  button.lastElementChild.textContent = name;
}

function addMarkers() {
  for (const place of config.places) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `marker ${place.type}${place.featured ? " featured" : ""}`;
    button.hidden = true;
    button.innerHTML = `<span class="marker-dot">${placeIcon(place)}</span><span class="marker-name" aria-hidden="true"></span>`;
    updateMarkerLabel(button, place);
    button.addEventListener("click", () => openPlace(place, button));
    $("markers").append(button);
    markerButtons.set(place.id, button);
  }
}

function updateList() {
  const terms = fold(search.value.trim()).split(/\s+/).filter(Boolean);
  const results = config.places.filter(place => {
    if (!terms.length) return layers[place.type];
    const labels = place.kind ? [place.kind, placeKinds[place.kind]] : categories[place.type];
    const searchable = fold([place.name, place.greek, ...labels, place.aliases].join(" "));
    return terms.every(term => searchable.includes(term));
  });
  $("list-title").textContent = t(terms.length ? "searchResults" : "landmarks");
  $("list-count").textContent = results.length;
  $("place-list").replaceChildren();
  for (const place of results) {
    const entry = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = `place-row ${place.type}`;
    if (selected === place) button.setAttribute("aria-current", "true");
    button.innerHTML = `<span class="category-icon">${placeIcon(place)}</span>
      <strong class="list-name"></strong>${icon("chevron")}`;
    button.querySelector(".list-name").textContent = placeName(place);
    button.addEventListener("click", () => openPlace(place, button));
    entry.append(button);
    $("place-list").append(entry);
  }
  if (!results.length) {
    const empty = document.createElement("li");
    empty.className = "empty-list";
    empty.textContent = t(terms.length ? "noMatches" : "enableLayer");
    $("place-list").append(empty);
  }
}

function updateSelection() {
  for (const [id, button] of markerButtons) {
    button.classList.toggle("selected", selected?.id === id);
  }
  $("place-card").hidden = !selected;
  if (selected) {
    $("card-type").textContent = placeKind(selected);
    $("card-title").textContent = placeName(selected);
    $("card-summary").textContent = placeSummary(selected);
    $("card-link").href = `https://www.openstreetmap.org/?mlat=${selected.lat}&mlon=${selected.lon}#map=18/${selected.lat}/${selected.lon}`;
    $("card-satellite").href = `https://www.google.com/maps/@?api=1&map_action=map&center=${selected.lat},${selected.lon}&zoom=19&basemap=satellite`;
  }
  updateList();
  if (!frame) frame = requestAnimationFrame(renderView);
}

function openPlace(place, trigger) {
  const moveFocus = document.activeElement === trigger || smallScreen.matches;
  lastPlaceControl = trigger;
  selected = place;
  toggleBrowse(false);
  updateSelection();
  if (moveFocus) $("close-place").focus({ preventScroll: true });
  const z = landmarkZoom(place);
  flyTo({ x: place.x, y: place.y + (smallScreen.matches ? 50 : 22) / z, z });
}

function landmarkZoom(place) {
  const target = isChurch(place) ? 2.0 : place.type === "civic" ? .92 : place.type === "monastery" ? .76 : .54;
  return Math.max(view.z, target * (smallScreen.matches ? .87 : 1));
}

function closePlace(restoreFocus = true) {
  if (!selected) return;
  const marker = markerButtons.get(selected.id);
  const layerIsVisible = layers[selected.type];
  selected = null;
  updateSelection();
  if (!restoreFocus) return;
  if (lastPlaceControl?.isConnected && !lastPlaceControl.hidden) lastPlaceControl.focus();
  else if (!marker.hidden && layerIsVisible) marker.focus();
  else viewport.focus();
}

function resetView() {
  closePlace(false);
  flyTo(homeView());
  history.replaceState(null, "", location.pathname + location.search);
}

function syncLayers() {
  for (const button of document.querySelectorAll("[data-layer]")) {
    button.setAttribute("aria-pressed", layers[button.dataset.layer]);
  }
  const trail = $("trail-overlay");
  if (layers.trails && !trail.hasAttribute("src")) trail.src = trail.dataset.src;
  trail.hidden = !layers.trails;
  syncScene();
  updateList();
  if (!frame) frame = requestAnimationFrame(renderView);
}

function syncScene() {
  viewport.dataset.time = timeOfDay;
  for (const input of lightingChoices) input.checked = input.value === timeOfDay;
  const glow = $("window-lights");
  const useGlow = timeOfDay === "night" || timeOfDay === "sunset";
  if (useGlow && !glow.hasAttribute("src")) glow.src = glow.dataset.src;
  glow.hidden = !useGlow;
  sceneLife.set(layers.life, timeOfDay);
}

function restoreLink() {
  const hash = new URLSearchParams(location.hash.slice(1));
  const linkedLanguage = hash.get("lang");
  if (isLanguage(linkedLanguage) && linkedLanguage !== language) setLanguage(linkedLanguage);
  if (!hash.has("v")) return;
  const [x, y, z, ...rest] = hash.get("v").split(",").map(Number);
  if (rest.length || ![x, y, z].every(Number.isFinite) || z <= 0) return;
  const settings = hash.get("layers");
  Object.assign(layers, layerDefaults);
  if (settings && /^[01]{4,6}$/.test(settings)) {
    Object.keys(layers).slice(0, settings.length).forEach((key, i) => {
      layers[key] = settings[i] === "1";
    });
  }
  const savedTime = hash.get("time");
  timeOfDay = timesOfDay.has(savedTime) ? savedTime : "day";
  selected = config.places.find(place => place.id === hash.get("place")) ?? null;
  stopAnimation();
  if (hash.get("map") === config.version) positionView(x, y, z);
  else if (selected) positionView(selected.x, selected.y, landmarkZoom(selected));
  else {
    const home = homeView();
    positionView(home.x, home.y, home.z);
  }
  syncLayers();
  updateSelection();
}

async function share() {
  const address = new URL(location.href);
  const params = new URLSearchParams();
  params.set("map", config.version);
  params.set("v", `${view.x.toFixed(1)},${view.y.toFixed(1)},${view.z.toFixed(3)}`);
  params.set("layers", Object.values(layers).map(enabled => enabled ? "1" : "0").join(""));
  params.set("lang", language);
  if (timeOfDay !== "day") params.set("time", timeOfDay);
  if (selected) params.set("place", selected.id);
  address.hash = params;
  history.replaceState(null, "", address);
  try {
    await navigator.clipboard.writeText(address.href);
    $("toast").hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { $("toast").hidden = true; }, 2400);
  } catch {
    $("share-url").value = address.href;
    $("share-dialog").showModal();
    $("share-url").select();
  }
}

function localPoint(event) {
  const rect = viewport.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function gesture() {
  const [a, b] = pointers.values();
  if (!b) return { ...a, distance: 0 };
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, distance: Math.hypot(a.x - b.x, a.y - b.y) };
}

function bindMap() {
  viewport.addEventListener("pointerdown", event => {
    if (event.button !== 0 || event.target.closest(".map-control, .marker")) return;
    if (pointers.size >= 2) return;
    stopAnimation();
    toggleBrowse(false);
    pointers.set(event.pointerId, localPoint(event));
    viewport.setPointerCapture(event.pointerId);
    viewport.classList.add("dragging");
  });
  viewport.addEventListener("pointermove", event => {
    if (!pointers.has(event.pointerId)) return;
    const old = gesture();
    pointers.set(event.pointerId, localPoint(event));
    const next = gesture();
    const z = old.distance > 0 ? clampZoom(view.z * next.distance / old.distance) : view.z;
    const worldX = view.x + (old.x - view.width / 2) / view.z;
    const worldY = view.y + (old.y - view.height / 2) / view.z;
    positionView(worldX - (next.x - view.width / 2) / z,
      worldY - (next.y - view.height / 2) / z, z);
  });
  for (const eventName of ["pointerup", "pointercancel", "lostpointercapture"]) {
    viewport.addEventListener(eventName, event => {
      pointers.delete(event.pointerId);
      if (!pointers.size) viewport.classList.remove("dragging");
    });
  }
  viewport.addEventListener("wheel", event => {
    if (event.target.closest(".place-card")) return;
    event.preventDefault();
    const point = localPoint(event);
    const multiplier = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? view.height : 1;
    zoomAt(Math.exp(-event.deltaY * multiplier * 0.0018), point.x, point.y);
  }, { passive: false });
  viewport.addEventListener("dblclick", event => {
    if (event.target.closest(".map-control, .marker")) return;
    const point = localPoint(event);
    zoomAt(1.45, point.x, point.y);
  });

  $("minimap").addEventListener("click", event => {
    if (event.detail === 0) {
      resetView();
      return;
    }
    const rect = $("minimap").getBoundingClientRect();
    flyTo({ x: (event.clientX - rect.left) / rect.width * config.width,
      y: (event.clientY - rect.top) / rect.height * config.height, z: view.z });
  });
}

function bindKeyboard() {
  document.addEventListener("keydown", event => {
    if (document.querySelector("dialog[open]") || event.ctrlKey || event.altKey || event.metaKey) return;
    if (event.key === "Escape") {
      if (search === document.activeElement && search.value) {
        search.value = "";
        updateList();
      } else if (selected) closePlace();
      else toggleBrowse(false, $("sidebar").classList.contains("open"));
      return;
    }
    if (event.target.closest("input, textarea, select")) return;
    if (event.key === "/") {
      event.preventDefault();
      toggleBrowse(true);
      search.focus();
    }
    if (event.key === "+" || event.key === "=") zoomAt(1.3);
    if (event.key === "-") zoomAt(1 / 1.3);
    if (event.key === "0") resetView();
    if (document.activeElement !== viewport) return;
    const movement = { ArrowLeft: [-80, 0], ArrowRight: [80, 0], ArrowUp: [0, -80], ArrowDown: [0, 80] }[event.key];
    if (movement) {
      event.preventDefault();
      stopAnimation();
      positionView(view.x + movement[0] / view.z, view.y + movement[1] / view.z);
    }
  });
}

async function init() {
  const response = await fetch("./assets/map.json");
  if (!response.ok) throw new Error(`Map data returned HTTP ${response.status}`);
  config = await response.json();
  config.places.sort((a, b) => browseOrder.indexOf(a.type) - browseOrder.indexOf(b.type));
  const world = $("world");
  world.style.width = `${config.width}px`;
  world.style.height = `${config.height}px`;
  $("minimap").style.aspectRatio = `${config.width} / ${config.height}`;
  sceneLife = new SceneAnimation($("scene-life"), config, view, reduceMotion);
  viewport.style.setProperty("--north", `${config.northAngle}deg`);
  addMarkers();
  measure();
  syncLayers();
  restoreLink();
  new ResizeObserver(measure).observe(viewport);
  addEventListener("hashchange", restoreLink);
  bindMap();
  bindKeyboard();
  $("home").addEventListener("click", resetView);
  $("brand").addEventListener("click", event => { event.preventDefault(); resetView(); });
  $("zoom-in").addEventListener("click", () => zoomAt(1.35));
  $("zoom-out").addEventListener("click", () => zoomAt(1 / 1.35));
  $("browse").addEventListener("click", () => {
    const open = !$("sidebar").classList.contains("open");
    toggleBrowse(open);
  });
  $("close-place").addEventListener("click", () => closePlace());
  $("share-button").addEventListener("click", share);
  search.addEventListener("input", updateList);
  search.addEventListener("keydown", event => {
    if (event.key === "Enter" && search.value.trim()) $("place-list").querySelector("button")?.click();
  });
  for (const input of lightingChoices) {
    input.addEventListener("change", () => {
      timeOfDay = input.value;
      syncScene();
    });
  }
  for (const button of document.querySelectorAll("[data-layer]")) {
    button.addEventListener("click", () => {
      const layer = button.dataset.layer;
      layers[layer] = !layers[layer];
      if (!layers[layer] && selected?.type === layer) closePlace(false);
      syncLayers();
    });
  }
}

$("about-button").addEventListener("click", () => $("about").showModal());
$("close-about").addEventListener("click", () => $("about").close());
$("close-share").addEventListener("click", () => $("share-dialog").close());
$("retry").addEventListener("click", () => location.reload());
for (const button of languageButtons) {
  button.addEventListener("click", () => chooseLanguage(button.dataset.language));
}
const linkedLanguage = new URLSearchParams(location.hash.slice(1)).get("lang");
setLanguage(isLanguage(linkedLanguage) ? linkedLanguage : savedLanguage());
document.querySelector(".terrain").decode().catch(() => { $("map-error").hidden = false; });
init().catch(error => {
  console.error(error);
  $("map-error").hidden = false;
});
