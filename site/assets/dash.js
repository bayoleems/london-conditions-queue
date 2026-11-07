/*
 * A small stand-in for the dashboard runtime, so the page's chart code runs as a plain static site.
 *
 *   dash.loader(id, {fn})        how a table is read out of data/ui_dashboard.json
 *   dash.calc(id, {inputs, fn})  a value worked out from other tables
 *   dash.data(id)                {status, data} for a table or calc
 *   dash.draw(name, fn)          registers a drawing and runs it on every element marked data-draw="name"
 *   dash.colors                  series colours for the current light or dark theme
 *
 * Drawings run once at load (showing placeholders), again when the data arrives, and on resize or theme change.
 */
(function () {
  "use strict";

  const LIGHT = ["#2563c9", "#d9622b", "#1f9d74", "#8a5cd1", "#0f97a6", "#c94882", "#6b7280", "#b8860b"];
  const DARK = ["#6aa2f5", "#f08a54", "#45c296", "#a98af0", "#38bfcf", "#e47aa7", "#9ca3af", "#e0b443"];
  const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");

  const state = { status: "loading", raw: null, loaders: {}, calcs: {}, draws: {}, cache: {} };

  const dash = {
    get colors() {
      return darkQuery.matches ? DARK : LIGHT;
    },
    loader(id, spec) {
      state.loaders[id] = spec.fn;
    },
    calc(id, spec) {
      state.calcs[id] = spec;
    },
    data(id) {
      if (state.status !== "ok") return { status: state.status, data: [] };
      if (!(id in state.cache)) {
        let result;
        try {
          if (state.calcs[id]) {
            const { inputs, fn } = state.calcs[id];
            result = { status: "ok", data: fn(...inputs.map((input) => dash.data(input).data)) };
          } else if (state.loaders[id]) {
            result = { status: "ok", data: state.loaders[id]({ answer: state.raw }) || [] };
          } else {
            result = { status: "missing", data: [] };
          }
        } catch (err) {
          console.error(`Couldn't read "${id}"`, err);
          result = { status: "error", data: [] };
        }
        state.cache[id] = result;
      }
      return state.cache[id];
    },
    draw(name, fn) {
      if (fn) state.draws[name] = fn;
      const drawing = state.draws[name];
      if (!drawing) return;
      document.querySelectorAll(`[data-draw="${name}"]`).forEach((el) => {
        try {
          drawing(el);
        } catch (err) {
          console.error(`Drawing "${name}" failed`, err);
        }
      });
    },
    // Filters and links aren't used on the static site
    params: () => ({}),
    setParams() {},
    resetParams() {},
    setLink() {},
    refresh() {},
  };
  window.dash = dash;

  const drawAll = () => Object.keys(state.draws).forEach((name) => dash.draw(name));
  const applyThemeColours = () =>
    document.querySelectorAll(".cq").forEach((el) => {
      el.style.setProperty("--cq-c0", dash.colors[0]);
      el.style.setProperty("--cq-c1", dash.colors[1]);
    });

  fetch("data/ui_dashboard.json")
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((json) => {
      state.raw = json;
      state.status = "ok";
      state.cache = {};
      drawAll();
    })
    .catch((err) => {
      console.error("Couldn't load data/ui_dashboard.json", err);
      state.status = "error";
      drawAll();
    });

  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(drawAll, 150);
  });
  darkQuery.addEventListener("change", () => {
    applyThemeColours();
    drawAll();
  });
})();
