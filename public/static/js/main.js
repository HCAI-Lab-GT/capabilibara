/* Site interactions: theme, navigation, scroll reveals, taxonomy grid, scrollspy,
   progress bar, and compact figure animations. */
import { MODELS } from "./animations/socialtda-data.js";

(function () {
  "use strict";

  var root = document.documentElement;
  root.classList.remove("no-js");

  function ready(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else fn();
  }

  /* ---------- Theme ---------- */
  function applyTheme(dark) {
    root.setAttribute("data-theme", dark ? "dark" : "light");
    var toggle = document.getElementById("theme-toggle");
    if (toggle) {
      toggle.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
    }
  }

  function initTheme() {
    // The inline head script already set data-theme before paint; mirror it.
    var media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
    if (!root.getAttribute("data-theme")) {
      applyTheme(media && media.matches);
    }

    var toggle = document.getElementById("theme-toggle");
    if (toggle) {
      toggle.addEventListener("click", function () {
        var dark = root.getAttribute("data-theme") !== "dark";
        applyTheme(dark);
        try { localStorage.setItem("theme", dark ? "dark" : "light"); } catch (e) {}
      });
    }

    // Follow OS changes only while the user has not chosen explicitly.
    if (media && media.addEventListener) {
      media.addEventListener("change", function (e) {
        var saved = null;
        try { saved = localStorage.getItem("theme"); } catch (err) {}
        if (!saved) applyTheme(e.matches);
      });
    }
  }

  /* ---------- Navbar ---------- */
  function setNavHeight() {
    var navbar = document.querySelector(".paper-nav");
    if (!navbar) return;
    root.style.setProperty("--nav-height", Math.round(navbar.getBoundingClientRect().height) + "px");
  }

  function closeMenu() {
    document.querySelectorAll(".navbar-burger").forEach(function (burger) {
      burger.classList.remove("is-active");
      burger.setAttribute("aria-expanded", "false");
    });
    document.querySelectorAll(".navbar-menu").forEach(function (menu) {
      menu.classList.remove("is-active");
    });
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
      item.addEventListener("click", closeMenu);
    });
  }

  /* ---------- Scroll reveal ---------- */
  function initReveal() {
    var nodes = [].slice.call(document.querySelectorAll("[data-reveal]"));
    nodes.forEach(function (el) {
      var delay = parseInt(el.getAttribute("data-reveal-delay") || "0", 10);
      if (delay) el.style.setProperty("--reveal-delay", delay + "ms");
    });
    if (!("IntersectionObserver" in window)) {
      nodes.forEach(function (el) { el.classList.add("is-revealed"); });
      return;
    }
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-revealed");
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
    nodes.forEach(function (el) { observer.observe(el); });
  }

  /* ---------- Taxonomy grid ---------- */
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

  /* ---------- Section-level scroll reveals for figures ---------- */
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

  /* ---------- Scrollspy: highlight the nav item for the section in view ---------- */
  function initScrollSpy() {
    var links = [].slice.call(document.querySelectorAll(".navbar-menu .navbar-item[href^='#']"));
    if (!links.length || !("IntersectionObserver" in window)) return;

    var byId = {};
    links.forEach(function (link) {
      var id = link.getAttribute("href").slice(1);
      var section = document.getElementById(id);
      if (section) byId[id] = link;
    });

    function setActive(id) {
      links.forEach(function (link) {
        var on = link.getAttribute("href") === "#" + id;
        link.classList.toggle("is-active", on);
        if (on) link.setAttribute("aria-current", "true");
        else link.removeAttribute("aria-current");
      });
    }

    var currentId = null;
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) currentId = entry.target.id;
      });
      if (currentId) setActive(currentId);
    }, { rootMargin: "-40% 0px -55% 0px", threshold: 0 });

    Object.keys(byId).forEach(function (id) { spy.observe(document.getElementById(id)); });

    window.addEventListener("hashchange", function () {
      var id = location.hash.slice(1);
      if (id && byId[id]) setActive(id);
    });
  }

  /* ---------- Reading progress bar under the navbar ---------- */
  function initProgressBar() {
    var bar = document.getElementById("scroll-progress");
    if (!bar) return;
    var ticking = false;
    function update() {
      ticking = false;
      var doc = document.documentElement;
      var max = doc.scrollHeight - window.innerHeight;
      var frac = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
      bar.style.transform = "scaleX(" + frac + ")";
    }
    window.addEventListener("scroll", function () {
      if (!ticking) { ticking = true; requestAnimationFrame(update); }
    }, { passive: true });
    update();
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

  /* ---------- Models table scroll affordance ---------- */
  function initTableFade() {
    var wrap = document.querySelector(".models-table-wrap");
    if (!wrap) return;
    function update() {
      var max = wrap.scrollWidth - wrap.clientWidth;
      var overflow = max > 4;
      wrap.classList.toggle("is-scrollable", overflow);
      wrap.classList.toggle("is-at-end", overflow && wrap.scrollLeft >= max - 4);
    }
    wrap.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    update();
  }

  ready(function () {
    setNavHeight();
    initTheme();
    initNavbar();
    initReveal();
    initTaxonomyGrid();
    initScrollAnimations();
    initScrollSpy();
    initProgressBar();
    initModelsTable();
    initBibtexCopy();
    initTableFade();

    window.addEventListener("resize", setNavHeight);
  });
})();
