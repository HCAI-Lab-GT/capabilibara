/* Site interactions: navigation, scroll reveals, taxonomy grid, and compact figure animations. */
import { MODELS } from "./animations/socialtda-data.js";

(function () {
  "use strict";

  var root = document.documentElement;
  root.classList.remove("no-js");

  function ready(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else fn();
  }

  function setNavHeight() {
    var navbar = document.querySelector(".paper-nav");
    if (!navbar) return;
    root.style.setProperty("--nav-height", Math.round(navbar.getBoundingClientRect().height) + "px");
  }

  function initNavbar() {
    document.querySelectorAll(".navbar-burger").forEach(function (burger) {
      burger.addEventListener("click", function () {
        var target = document.getElementById(burger.dataset.target);
        var active = !burger.classList.contains("is-active");
        burger.classList.toggle("is-active", active);
        burger.setAttribute("aria-expanded", active ? "true" : "false");
        if (target) target.classList.toggle("is-active", active);
      });
    });

    document.querySelectorAll(".navbar-menu .navbar-item").forEach(function (item) {
      item.addEventListener("click", function () {
        document.querySelectorAll(".navbar-burger").forEach(function (burger) {
          burger.classList.remove("is-active");
          burger.setAttribute("aria-expanded", "false");
        });
        document.querySelectorAll(".navbar-menu").forEach(function (menu) {
          menu.classList.remove("is-active");
        });
      });
    });
  }

  function initTaxonomyGrid() {
    var grid = document.querySelector("[data-tax-grid]");
    if (!grid || grid.childElementCount) return;

    // Numbered cells match the numbered example chips under the grid.
    var exampleCells = {
      "4-6": 1, "7-14": 2, "11-19": 3, "2-3": 4, "15-9": 5, "19-16": 6
    };

    for (var r = 0; r < 24; r += 1) {
      for (var c = 0; c < 24; c += 1) {
        var cell = document.createElement("span");
        cell.className = "tax-cell";
        var delay = ((r * 19 + c * 13) % 130) * 6;
        cell.style.setProperty("--delay", delay + "ms");
        var n = exampleCells[r + "-" + c];
        if (n) {
          cell.classList.add("is-example");
          cell.textContent = n;
        }
        cell.setAttribute("aria-hidden", "true");
        grid.appendChild(cell);
      }
    }
  }

  function initScrollAnimations() {
    var nodes = [].slice.call(document.querySelectorAll(
      ".taxonomy-viz, .contrast-viz, .cluster-viz, .influence-viz"
    ));

    if (!("IntersectionObserver" in window)) {
      nodes.forEach(function (node) {
        node.classList.add("is-visible");
      });
      return;
    }

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.25, rootMargin: "0px 0px -8% 0px" });

    nodes.forEach(function (node) { observer.observe(node); });
  }

  /* Render the model roster table from MODELS. Rows = models; columns are
     identity facts only (corpus, scale, status) — the roster is qualitative
     by design and never carries result numbers. The markup ships a static
     copy as a no-JS fallback; this re-renders from the module so roster
     changes happen in one place. */
  function esc(s) {
    return String(s).replace(/[&<>]/g, function (c) {
      return c === "&" ? "&amp;" : c === "<" ? "&lt;" : "&gt;";
    });
  }

  function initModelsTable() {
    var head = document.getElementById("models-head");
    var body = document.getElementById("models-body");
    if (!head || !body || !MODELS || !MODELS.length) return;

    var badges = {
      "primary": '<span class="status-badge status-done">Primary study</span>',
      "in-progress": '<span class="status-badge status-pending">In progress</span>',
      "planned": '<span class="status-badge status-pending">Planned</span>'
    };

    head.innerHTML = '<tr><th scope="col">Model</th><th scope="col">Training corpus</th>' +
      '<th scope="col">Scale</th><th scope="col">Status</th></tr>';

    var bodyHtml = "";
    MODELS.forEach(function (m) {
      bodyHtml += '<tr><th scope="row" class="metric-cell">' + esc(m.name) + '</th>' +
        '<td class="value-cell">' + esc(m.corpus) + '</td>' +
        '<td class="value-cell">' + esc(m.scale) + '</td>' +
        '<td class="value-cell">' + (badges[m.status] || badges["planned"]) + '</td></tr>';
    });
    body.innerHTML = bodyHtml;
  }

  function initBibtexCopy() {
    var button = document.querySelector("[data-copy-bibtex]");
    var code = document.getElementById("bibtex-code");
    if (!button || !code) return;
    if (!navigator.clipboard) {
      button.hidden = true;
      return;
    }
    button.addEventListener("click", function () {
      navigator.clipboard.writeText(code.textContent).then(function () {
        var original = button.textContent;
        button.textContent = "Copied";
        button.classList.add("is-copied");
        button.setAttribute("aria-pressed", "true");
        window.setTimeout(function () {
          button.textContent = original;
          button.classList.remove("is-copied");
          button.setAttribute("aria-pressed", "false");
        }, 1600);
      });
    });
  }

  ready(function () {
    setNavHeight();
    initNavbar();
    initTaxonomyGrid();
    initScrollAnimations();
    initModelsTable();
    initBibtexCopy();

    window.addEventListener("resize", setNavHeight);

    if (window.AOS) {
      var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      window.AOS.init({ once: true, duration: 650, easing: "ease-out-cubic", disable: reduce });
    } else {
      /* AOS failed to load (CDN blocked or offline): its stylesheet hides
         [data-aos] elements, so strip the attributes to reveal everything. */
      document.querySelectorAll("[data-aos]").forEach(function (el) {
        el.removeAttribute("data-aos");
        el.removeAttribute("data-aos-delay");
      });
    }
  });
})();
