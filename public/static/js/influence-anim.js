/* Animated influence-function explainer — the geometry of an influence score.
 A benchmark question defines a direction in model space; each training
 document's gradient nudge scores by its alignment with that direction
 (positive = aligned, negative = opposed); averaging thousands of noisy
 per-document scores yields a readable signal.

 Auto-looping GSAP timeline, played/paused by an IntersectionObserver.
 Falls back to the static SVG (.infl-static) when GSAP is unavailable or
 the user prefers reduced motion. */

var NS = "http://www.w3.org/2000/svg";
var VB_W = 460, VB_H = 300;

var C = {
  query: "#762A83",
  pos: "#2F6FA8",
  neg: "#B35806",
  ink: "#171717",
  faint: "#8a8790",
  hairline: "#ded9ce"
};

var OX = 150, OY = 150; // origin: "the model"

function el(tag, attrs, parent) {
  var e = document.createElementNS(NS, tag);
  if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function txt(parent, x, y, s, attrs) {
  var t = el("text", Object.assign({ x: x, y: y }, attrs || {}), parent);
  t.textContent = s;
  return t;
}
function arrowHead(parent, x, y, angleDeg, color, size) {
  var s = size || 7;
  var head = el("polygon", {
    points: s + ",0 " + (-s * 0.6) + "," + (-s * 0.62) + " " + (-s * 0.6) + "," + (s * 0.62),
    fill: color,
    transform: "translate(" + x + " " + y + ") rotate(" + angleDeg + ")"
  }, parent);
  return head;
}
function docGlyph(parent, x, y, w, h) {
  var fold = 7, g = el("g", {}, parent);
  el("path", {
    d: "M" + x + " " + (y + 3) + " Q" + x + " " + y + " " + (x + 3) + " " + y +
       " H" + (x + w - fold) + " L" + (x + w) + " " + (y + fold) +
       " V" + (y + h - 3) + " Q" + (x + w) + " " + (y + h) + " " + (x + w - 3) + " " + (y + h) +
       " H" + (x + 3) + " Q" + x + " " + (y + h) + " " + x + " " + (y + h - 3) + " Z",
    fill: "#ffffff", stroke: C.hairline, "stroke-width": 1.1, "stroke-linejoin": "round"
  }, g);
  el("path", { d: "M" + (x + w - fold) + " " + y + " V" + (y + fold) + " H" + (x + w) + " Z", fill: "#efe9dd" }, g);
  var ly = y + 8;
  while (ly < y + h - 5) {
    el("rect", { x: x + 5, y: ly, width: w - 10, height: 1.8, rx: 0.9, fill: "#d9d3c6" }, g);
    ly += 5.5;
  }
  return g;
}
// a vector from the origin with a matching arrowhead; returns parts for animation
function vec(parent, angleDeg, len, color, width) {
  var rad = angleDeg * Math.PI / 180;
  var x2 = OX + len * Math.cos(rad), y2 = OY + len * Math.sin(rad);
  var g = el("g", {}, parent);
  var line = el("line", {
    x1: OX, y1: OY, x2: x2, y2: y2,
    stroke: color, "stroke-width": width, "stroke-linecap": "round"
  }, g);
  var head = arrowHead(g, x2, y2, angleDeg, color, width >= 2 ? 7 : 4.5);
  return { g: g, line: line, head: head, x2: x2, y2: y2 };
}
function drawIn(gsap, tl, line, dur, at) {
  var len = Math.hypot(line.x2.baseVal.value - line.x1.baseVal.value,
                       line.y2.baseVal.value - line.y1.baseVal.value);
  line.setAttribute("stroke-dasharray", len);
  line.setAttribute("stroke-dashoffset", len);
  tl.to(line, { attr: { "stroke-dashoffset": 0 }, duration: dur, ease: "power2.out" }, at);
}

function build(container) {
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var gsap = window.gsap;
  if (reduce || !gsap) return; // static SVG stays

  var svg = el("svg", {
    class: "infl-anim", viewBox: "0 0 " + VB_W + " " + VB_H, role: "img",
    "aria-label": "Animated: a benchmark question defines a direction. A training document whose nudge aligns with it scores positive influence, one whose nudge opposes it scores negative, and averaging thousands of noisy documents yields a readable signal."
  });
  container.appendChild(svg);
  container.classList.add("has-anim");

  // static furniture: axis, origin, caption slot
  el("line", { x1: 30, y1: OY, x2: 430, y2: OY, stroke: C.hairline, "stroke-width": 1, "stroke-dasharray": "4 3" }, svg);
  el("circle", { cx: OX, cy: OY, r: 4, fill: C.ink }, svg);
  txt(svg, 141, 141, "the model", { class: "ia-label", "text-anchor": "end" });
  var caption = txt(svg, VB_W / 2, 287, "", { class: "ia-caption", "text-anchor": "middle", opacity: 0 });

  // beat 1 — the question's direction
  var g1 = el("g", { opacity: 0 }, svg);
  var qv = vec(g1, 0, 220, C.query, 2.4);
  var qLabel = txt(g1, 352, 134, "the question's direction", { class: "ia-chip", fill: C.query, "text-anchor": "middle" });

  // beat 2 — an aligned document
  var g2 = el("g", { opacity: 0 }, svg);
  var posDoc = docGlyph(g2, 287, 32, 26, 32);
  var pv = vec(g2, -35, 150, C.faint, 2.2); // tip ≈ (273, 64)
  var pDrop = el("line", { x1: pv.x2, y1: pv.y2, x2: pv.x2, y2: OY, stroke: C.faint, "stroke-width": 1, "stroke-dasharray": "4 3", opacity: 0.8 }, g2);
  var pProj = el("line", { x1: OX, y1: OY, x2: pv.x2, y2: OY, stroke: C.pos, "stroke-width": 2.5, "stroke-linecap": "round" }, g2);
  var pChip = txt(g2, 212, 174, "+ positive influence", { class: "ia-chip", fill: C.pos, "text-anchor": "middle" });

  // beat 3 — an opposed document
  var g3 = el("g", { opacity: 0 }, svg);
  var negDoc = docGlyph(g3, 30, 218, 26, 32);
  var nv = vec(g3, 145, 110, C.faint, 2.2); // tip ≈ (60, 213)
  var nDrop = el("line", { x1: nv.x2, y1: nv.y2, x2: nv.x2, y2: OY, stroke: C.faint, "stroke-width": 1, "stroke-dasharray": "4 3", opacity: 0.8 }, g3);
  var nProj = el("line", { x1: OX, y1: OY, x2: nv.x2, y2: OY, stroke: C.neg, "stroke-width": 2.5, "stroke-linecap": "round" }, g3);
  var nChip = txt(g3, 110, 264, "− negative influence", { class: "ia-chip", fill: C.neg, "text-anchor": "middle" });

  // beat 4 — many noisy documents average into a signal
  var g4 = el("g", { opacity: 0 }, svg);
  var noiseAngles = [-52, -38, -24, -14, -6, 4, 12, 22, 34, 48, 128, 158, -70, 172];
  var noise = noiseAngles.map(function (a, i) {
    var len = 55 + ((i * 29) % 40);
    return vec(g4, a, len, C.faint, 1.1);
  });
  var avg = vec(g4, 0, 235, C.pos, 3);
  var avgLabel = txt(g4, 270, 133, "the average: a readable signal", { class: "ia-chip", fill: C.pos, "text-anchor": "middle" });

  var CAPTIONS = [
    "A benchmark question asks: what would make the model better at this?",
    "A document whose training nudge aligns with that direction scored positive,",
    "and one whose nudge opposes it scored negative.",
    "One document is noisy. Averaged over thousands, a readable signal emerges."
  ];

  function setCaption(tl, i, at) {
    tl.to(caption, { opacity: 0, duration: 0.25 }, at)
      .add(function () { caption.textContent = CAPTIONS[i]; }, at + 0.25)
      .to(caption, { opacity: 1, duration: 0.4 }, at + 0.3);
  }

  var tl = gsap.timeline({ paused: true, repeat: -1, repeatDelay: 1.4 });

  // beat 1
  tl.set(g1, { opacity: 1 }, 0);
  gsap.set([qLabel, pChip, nChip, avgLabel], { opacity: 0 });
  gsap.set([pProj, nProj], { opacity: 0 });
  gsap.set([pDrop, nDrop], { opacity: 0 });
  gsap.set([qv.head, pv.head, nv.head, avg.head], { opacity: 0 });
  setCaption(tl, 0, 0);
  drawIn(gsap, tl, qv.line, 0.9, 0.2);
  tl.to(qv.head, { opacity: 1, duration: 0.2 }, 1.0)
    .to(qLabel, { opacity: 1, duration: 0.4 }, 1.1);

  // beat 2
  tl.set(g2, { opacity: 1 }, 1.8);
  setCaption(tl, 1, 1.8);
  tl.fromTo(posDoc, { opacity: 0 }, { opacity: 1, duration: 0.4 }, 2.0);
  drawIn(gsap, tl, pv.line, 0.7, 2.2);
  tl.to(pv.head, { opacity: 1, duration: 0.2 }, 2.8)
    .to(pDrop, { opacity: 0.8, duration: 0.35 }, 3.1)
    .fromTo(pProj, { opacity: 0 }, { opacity: 1, duration: 0.45, ease: "power2.inOut" }, 3.4);
  tl.to([pv.line, pv.head], { stroke: C.pos, fill: C.pos, duration: 0.45 }, 3.4);
  tl.to(pChip, { opacity: 1, duration: 0.4 }, 3.7);

  // beat 3
  tl.set(g3, { opacity: 1 }, 4.6);
  setCaption(tl, 2, 4.6);
  tl.fromTo(negDoc, { opacity: 0 }, { opacity: 1, duration: 0.4 }, 4.8);
  drawIn(gsap, tl, nv.line, 0.7, 5.0);
  tl.to(nv.head, { opacity: 1, duration: 0.2 }, 5.6)
    .to(nDrop, { opacity: 0.8, duration: 0.35 }, 5.9)
    .fromTo(nProj, { opacity: 0 }, { opacity: 1, duration: 0.45, ease: "power2.inOut" }, 6.2);
  tl.to([nv.line, nv.head], { stroke: C.neg, fill: C.neg, duration: 0.45 }, 6.2);
  tl.to(nChip, { opacity: 1, duration: 0.4 }, 6.5);

  // beat 4
  tl.to([g2, g3, qLabel], { opacity: 0, duration: 0.5 }, 7.4);
  tl.to([qv.line, qv.head], { opacity: 0.18, duration: 0.5 }, 7.4);
  tl.set(g4, { opacity: 1 }, 7.4);
  setCaption(tl, 3, 7.4);
  noise.forEach(function (v, i) {
    tl.fromTo(v.g, { opacity: 0 }, { opacity: 0.4, duration: 0.3 }, 7.7 + i * 0.08);
  });
  drawIn(gsap, tl, avg.line, 0.9, 9.1);
  tl.to(avg.head, { opacity: 1, duration: 0.2 }, 9.9)
    .to(avgLabel, { opacity: 1, duration: 0.4 }, 10.0);
  // quiet hold on the final frame comes from repeatDelay

  // reset transient colors/visibility each loop
  tl.eventCallback("onRepeat", function () {
    gsap.set([pv.line, nv.line], { stroke: C.faint });
    gsap.set([pv.head, nv.head], { fill: C.faint, opacity: 0 });
    gsap.set([qv.head, avg.head], { opacity: 0 });
    gsap.set([g2, g3, g4], { opacity: 0 });
    gsap.set([qv.line, qLabel], { opacity: 1 });
    gsap.set([qLabel, pChip, nChip, avgLabel, pProj, nProj, pDrop, nDrop], { opacity: 0 });
    noise.forEach(function (v) { gsap.set(v.g, { opacity: 0 }); });
  });

  // play only while on screen
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) tl.play();
        else tl.pause();
      });
    }, { threshold: 0.35 });
    io.observe(container);
  } else {
    tl.play();
  }

  var hovered = false;
  container.addEventListener("mouseenter", function () { hovered = true; tl.pause(); });
  container.addEventListener("mouseleave", function () { if (hovered) { hovered = false; tl.play(); } });
}

function init() {
  var node = document.querySelector('.influence-viz[data-concept="influence"]');
  if (node) build(node);
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
else init();
