/* Animated Figure 1 — Capability Provenance Pipeline.
 Self-contained SVG + GSAP timeline. Falls back to a static schematic when
 GSAP is unavailable or the user prefers reduced motion.

 Stages:
   1) Corpus Construction — Dolma3 -> dedup -> WebOrganizer 24×24 taxonomy
   2) Training Data Attribution — 2×2 benchmark design -> Bergson/TrackStar
   3) Influence Map — signed z per tracked bin (real values from the paper)
   4) Unlearning Check — influence-targeted vs random forgetting (real Δ)

 Scenes 3–4 carry the actual OLMo3-7B/Dolma3 numbers from the paper's
 Figure 1 and §4, always labeled with the model, since Comma-2T results
 differ (maintainer decision, 2026-08-02, superseding the earlier
 illustrative-values-only rule). Six bins B1–B6 are tracked through every
 stage as the figure's through-line. */

var NS = "http://www.w3.org/2000/svg";
var VB_W = 880, VB_H = 360;

// Shared 8x8 grid geometry (scene 1) + its center (fan convergence point).
var GRID = { x: 512, y: 72, cs: 30, gap: 2, n: 8 };
var GRID_W = GRID.n * (GRID.cs + GRID.gap) - GRID.gap; // 254
var GCX = GRID.x + GRID_W / 2;                          // grid center X
var GCY = GRID.y + GRID_W / 2;                          // grid center Y

// ---- palette ----
var C = {
  socReason: "#762A83", socKnow: "#8C6D1F",
  stemReason: "#A14F00", stemKnow: "#1B7837",
  slate: "#4B5563", grid: "#c9b8cf", gridFill: "#efe6f2",
  ink: "#171717", faint: "#8a8790", hairline: "#ded9ce"
};

// ---- the 2×2 benchmark design: domain (social/STEM) × capability ----
var BENCH = [
  { name: "SocialIQA",  sub: "social reasoning",  color: C.socReason },
  { name: "MMLU SS",    sub: "social knowledge",  color: C.socKnow },
  { name: "ARC-Chal.",  sub: "STEM reasoning",    color: C.stemReason },
  { name: "MMLU STEM",  sub: "STEM knowledge",    color: C.stemKnow }
];

// ---- six bins tracked through every stage (paper Fig. 1) ----
var BINS = [
  { id: "B1", short: "Lit × Support",     grid: [1, 1] },
  { id: "B2", short: "Social × Q&A",      grid: [6, 4] },
  { id: "B3", short: "Travel × Q&A",      grid: [7, 6] },
  { id: "B4", short: "Sci&T × Academic",  grid: [5, 1] },
  { id: "B5", short: "Industrial × Docs", grid: [2, 5] },
  { id: "B6", short: "Politics × Docs",   grid: [4, 6] }
];

// Scene 3 data — real signed z-scores from the paper's Figure 1 (OLMo3-7B).
// Rows are benchmarks, columns the six tracked bins B1–B6.
var INFLUENCE = {
  cols: BINS.map(function (b) { return b.short; }),
  rows: [
    { label: "SocialIQA",   color: C.socReason,  v: [16.0, 1.9, 0.5, 0.1, 6.5, 0.4], sig: 0 },
    { label: "MMLU SS",     color: C.socKnow,    v: [-7.3, -0.1, 1.1, 2.2, 7.0, 6.9] },
    { label: "ARC-Chal.",   color: C.stemReason, v: [-0.5, -0.8, -3.4, 2.7, 8.6, 1.6] },
    { label: "MMLU STEM",   color: C.stemKnow,   v: [-5.7, -0.5, -0.6, 3.1, 8.4, 5.6] }
  ]
};

// Scene 4 data — real paired unlearning result for SocialIQA (paper §4.3 /
// Fig. 1): Δ = influence-targeted − random accuracy damage, in pp, per topic.
var UNLEARN = [
  { label: "Literature",       v: 17.53 },
  { label: "Education & Jobs", v: 15.03 },
  { label: "Sports & Fitness", v: 8.40 },
  { label: "Fashion & Beauty", v: 6.93 },
  { label: "Home & Hobbies",   v: 6.87 }
];

var STAGES = [
  { icon: "1", title: "Corpus Construction" },
  { icon: "2", title: "Attribution" },
  { icon: "3", title: "Influence Map" },
  { icon: "4", title: "Unlearning" }
];
var CAPTIONS = [
  "<strong>1) Build a labeled corpus.</strong> The Dolma3 mix is de-duplicated to ~1.26B unique documents and labeled with WebOrganizer's 24 topics &times; 24 formats: <strong>576 bins</strong>. A stratified working set samples 10,000 documents per bin (5.68M documents). Six bins, B1&ndash;B6, are tracked through every stage.",
  "<strong>2) Score every region.</strong> Four benchmarks form a 2&times;2 design: domain (social vs. STEM) crossed with capability (reasoning vs. knowledge). TrackStar, via Bergson, compares document gradients from OLMo3-7B Base with query gradients from OLMo3-7B Instruct, yielding a signed score for every bin &times; benchmark.",
  "<strong>3) Read the map.</strong> Bin-level averages, z-scored within each benchmark <em>(OLMo3-7B)</em>. Literature &times; Customer Support is extreme for SocialIQA (+16.0) yet negative for the three comparison benchmarks, whose support concentrates in documentation-heavy bins.",
  "<strong>4) Check it causally.</strong> Per topic, forget the top-200 documents by influence versus 1,000 random same-topic documents (NGDiff, rank-8 LoRA, OLMo3-7B Base). Influence-targeted forgetting damages SocialIQA far more than random controls (paired Wilcoxon p &asymp; 10<sup>&minus;5</sup>)."
];

// ---- helpers ----
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
function lerp(a, b, t) { return a + (b - a) * t; }
function hex(c) { return [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)]; }
function mix(c1, c2, t) {
  var a = hex(c1), b = hex(c2);
  return "rgb(" + Math.round(lerp(a[0], b[0], t)) + "," + Math.round(lerp(a[1], b[1], t)) + "," + Math.round(lerp(a[2], b[2], t)) + ")";
}
// influence: blue (+) / orange (-), diverging from near-white; capped at |z| = 3
// like the paper's color scale so the ±16 outlier saturates rather than washing
// out every other cell.
function inflColor(z) {
  var t = Math.min(Math.abs(z) / 3, 1);
  return z >= 0 ? mix("#f7f7f7", "#2F6FA8", t) : mix("#f7f7f7", "#B35806", t);
}
function inflInk(z) { return Math.min(Math.abs(z) / 3, 1) > 0.55 ? "#fff" : "#222"; }
function fmt(v, plus, decimals) {
  return (v >= 0 && plus ? "+" : "") + v.toFixed(decimals == null ? 2 : decimals);
}

// ================= scene builders =================
// Each returns a <g> with an _enter(gsap) -> timeline method.

// soft multi-hue palette used to suggest distinct strata/bins
var STRATA = ["#762A83", "#8C6D1F", "#A14F00", "#1B7837", "#2F6FA8", "#5589B9", "#87B7D5", "#4B5563"];

// a "document" page glyph: rounded page with a folded corner + text lines
function docGlyph(parent, x, y, w, h) {
  var fold = Math.min(8, w * 0.22), gg = el("g", {}, parent);
  el("path", {
    d: "M" + x + " " + (y + 3) + " Q" + x + " " + y + " " + (x + 3) + " " + y +
       " H" + (x + w - fold) + " L" + (x + w) + " " + (y + fold) +
       " V" + (y + h - 3) + " Q" + (x + w) + " " + (y + h) + " " + (x + w - 3) + " " + (y + h) +
       " H" + (x + 3) + " Q" + x + " " + (y + h) + " " + x + " " + (y + h - 3) + " Z",
    fill: "#ffffff", stroke: "#c4bbaf", "stroke-width": 1.1, "stroke-linejoin": "round"
  }, gg);
  el("path", { d: "M" + (x + w - fold) + " " + y + " V" + (y + fold) + " H" + (x + w) + " Z", fill: "#efe9dd" }, gg);
  var ly = y + 10, li = 0;
  while (ly < y + h - 6) {
    el("rect", { x: x + 5, y: ly, width: w - 10 - (li % 3 === 2 ? 7 : 0), height: 2.2, rx: 1.1, fill: "#d9d3c6" }, gg);
    ly += 6; li++;
  }
  return gg;
}

function sceneCorpus(root) {
  var g = el("g", { opacity: 0 }, root);
  // Dolma3 = a block of document pages filling the SAME area as the bin grid
  var blockW = GRID_W;
  var docCols = 5, docRows = 4;
  var pitchX = blockW / docCols, pitchY = blockW / docRows;
  var pageW = pitchX * 0.72, pageH = pitchY * 0.78;
  var docX = 40, docY = GCY - blockW / 2;
  txt(g, docX + blockW / 2, docY - 24, "Dolma3 web corpus", { class: "f1-title", "font-size": 15, "text-anchor": "middle" });
  txt(g, docX + blockW / 2, docY - 8, "de-duplicated, ~1.26B unique documents", { class: "f1-label", "font-size": 11, "text-anchor": "middle" });
  var sampleLbl = txt(g, docX + blockW / 2, docY + blockW + 18, "stratified working set: 10,000 docs per bin, 5.68M docs",
    { class: "f1-label", "font-size": 11, "text-anchor": "middle", fill: C.slate, opacity: 0 });
  var docs = [];
  for (var i = 0; i < docCols * docRows; i++) {
    var col = i % docCols, rrow = Math.floor(i / docCols);
    var px = docX + col * pitchX + (pitchX - pageW) / 2;
    var py = docY + rrow * pitchY + (pitchY - pageH) / 2;
    var gd = docGlyph(g, px, py, pageW, pageH);
    gd.setAttribute("opacity", "0");
    docs.push(gd);
  }
  // lines all emanate from the middle of the corpus block
  var ox = docX + blockW - 4, oy = GCY;

  // WebOrganizer 24×24 grid (right), title centered over the grid
  txt(g, GCX, GRID.y - 24, "WebOrganizer taxonomy: 576 bins", { class: "f1-title", "font-size": 15, "text-anchor": "middle" });
  txt(g, GCX, GRID.y - 8, "24 topics × 24 formats", { class: "f1-label", "font-size": 11, "text-anchor": "middle" });
  var cells = [];
  for (var r = 0; r < GRID.n; r++) for (var c = 0; c < GRID.n; c++) {
    cells.push(el("rect", {
      x: GRID.x + c * (GRID.cs + GRID.gap), y: GRID.y + r * (GRID.cs + GRID.gap),
      width: GRID.cs, height: GRID.cs, rx: 4,
      fill: C.gridFill, stroke: C.grid, "stroke-width": 1, opacity: 0
    }, g));
  }
  // stratification: many rounded lines from the corpus fanning into spread bins
  var fan = [];
  for (var rr = 0; rr < GRID.n; rr++) {
    var picks = [1 + (rr % 4), 4 + (rr % 3)];
    picks.forEach(function (cc, j) {
      var cx = GRID.x + cc * (GRID.cs + GRID.gap) + GRID.cs / 2;
      var cy = GRID.y + rr * (GRID.cs + GRID.gap) + GRID.cs / 2;
      var c1x = ox + (cx - ox) * 0.42, c2x = ox + (cx - ox) * 0.6;
      var p = el("path", {
        d: "M " + ox + " " + oy + " C " + c1x + " " + oy + " " + c2x + " " + cy + " " + cx + " " + cy,
        fill: "none", stroke: STRATA[(rr + j) % STRATA.length], "stroke-width": 1.3,
        "stroke-linecap": "round", opacity: 0
      }, g);
      fan.push(p);
    });
  }

  // the six tracked bins, marked in place on the grid (after the fan so the
  // marks sit above the flow lines)
  var marks = BINS.map(function (b) {
    var r = b.grid[0], c = b.grid[1];
    var x = GRID.x + c * (GRID.cs + GRID.gap), y = GRID.y + r * (GRID.cs + GRID.gap);
    var mg = el("g", { opacity: 0 }, g);
    el("rect", { x: x, y: y, width: GRID.cs, height: GRID.cs, rx: 4, fill: "#ffffff", stroke: C.ink, "stroke-width": 1.3 }, mg);
    txt(mg, x + GRID.cs / 2, y + GRID.cs / 2 + 3.5, b.id, { fill: C.ink, "font-size": 10, "font-weight": 700, "text-anchor": "middle" });
    return mg;
  });

  g._enter = function (gsap) {
    var tl = gsap.timeline();
    gsap.set(docs, { opacity: 0 });
    gsap.set(cells, { opacity: 0 });
    gsap.set(marks, { opacity: 0 });
    gsap.set(sampleLbl, { opacity: 0 });
    tl.to(docs, { opacity: 1, duration: 0.4, stagger: 0.04 }, 0)
      // fade ~a third of the documents: only a stratified sample is scored
      .to(docs.filter(function (_, i) { return i % 3 === 1; }), { opacity: 0.18, duration: 0.5 }, 0.7)
      .to(sampleLbl, { opacity: 1, duration: 0.4 }, 0.9);
    // draw the stratification lines from the middle of the corpus
    fan.forEach(function (p, i) {
      var len = p.getTotalLength();
      p.setAttribute("stroke-dasharray", len);
      p.setAttribute("stroke-dashoffset", len);
      gsap.set(p, { opacity: 0.5 });
      tl.to(p, { attr: { "stroke-dashoffset": 0 }, duration: 0.55, ease: "power2.out" }, 0.9 + i * 0.035);
    });
    tl.to(cells, { opacity: 1, duration: 0.5, stagger: { each: 0.012, from: "random" } }, 1.1);
    tl.to(marks, { opacity: 1, duration: 0.35, stagger: 0.09 }, 1.9);
    return tl;
  };
  return g;
}

function sceneAttribution(root) {
  var g = el("g", { opacity: 0 }, root);

  // ---- left: the 2×2 benchmark design, stepped so every chip has a clear
  // horizontal lane for its beam ----
  var chipW = 132, chipH = 44, pitch = 54, y0 = 80;
  var colX = [92, 240];
  txt(g, colX[0] + chipW / 2, 66, "reasoning", { class: "f1-kicker", "text-anchor": "middle" });
  txt(g, colX[1] + chipW / 2, 66, "knowledge", { class: "f1-kicker", "text-anchor": "middle" });
  var rowMid = [y0 + pitch / 2 + chipH / 2 - 5, y0 + 2 * pitch + pitch / 2 + chipH / 2 - 5];
  txt(g, 74, rowMid[0], "social", { class: "f1-kicker", "text-anchor": "middle", transform: "rotate(-90 74 " + rowMid[0] + ")" });
  txt(g, 74, rowMid[1], "STEM", { class: "f1-kicker", "text-anchor": "middle", transform: "rotate(-90 74 " + rowMid[1] + ")" });

  // geometry shared by chips, box, bins, and beams
  var lanes = BENCH.map(function (_, i) { return y0 + i * pitch + chipH / 2; });
  var boxX = 420, boxW = 160, boxY = 128, boxH = 96;
  var binX = 620, binW = 240, binH = 30, binPitch = 38;
  var binY0 = (VB_H - (BINS.length * binPitch - (binPitch - binH))) / 2;
  var binMid = BINS.map(function (_, i) { return binY0 + i * binPitch + binH / 2; });

  // beams first, so every card sits above them: queries flow into the scorer…
  var inBeams = BENCH.map(function (b, i) {
    var sx = colX[i % 2] + chipW, sy = lanes[i];
    var ty = boxY + 20 + i * 19;
    var cx1 = sx + (boxX - sx) * 0.45, cx2 = sx + (boxX - sx) * 0.7;
    return el("path", {
      d: "M " + sx + " " + sy + " C " + cx1 + " " + sy + " " + cx2 + " " + ty + " " + boxX + " " + ty,
      fill: "none", stroke: b.color, "stroke-width": 1.8, "stroke-linecap": "round", opacity: 0
    }, g);
  });
  // …and scores flow out to every tracked bin
  var outBeams = BINS.map(function (_, i) {
    var sx = boxX + boxW, sy = boxY + 14 + i * 14;
    var cx1 = sx + (binX - sx) * 0.45, cx2 = sx + (binX - sx) * 0.7;
    return el("path", {
      d: "M " + sx + " " + sy + " C " + cx1 + " " + sy + " " + cx2 + " " + binMid[i] + " " + binX + " " + binMid[i],
      fill: "none", stroke: C.slate, "stroke-width": 1.2, "stroke-linecap": "round", opacity: 0
    }, g);
  });

  var chips = BENCH.map(function (b, i) {
    var x = colX[i % 2], y = y0 + i * pitch;
    var cg = el("g", { opacity: 0 }, g);
    el("rect", { x: x, y: y, width: chipW, height: chipH, rx: 4, fill: "#ffffff", stroke: C.hairline, "stroke-width": 1.1 }, cg);
    el("circle", { cx: x + 14, cy: y + 16, r: 4, fill: b.color }, cg);
    txt(cg, x + 24, y + 20, b.name, { fill: C.ink, "font-size": 12.5, "font-weight": 600 });
    txt(cg, x + 24, y + 34, b.sub, { fill: C.faint, "font-size": 10 });
    return { g: cg };
  });

  // ---- center: the scoring engine ----
  var box = el("g", { opacity: 0 }, g);
  el("rect", { x: boxX, y: boxY, width: boxW, height: boxH, rx: 6, fill: "#ffffff", stroke: C.hairline, "stroke-width": 1.2 }, box);
  txt(box, boxX + boxW / 2, boxY + 22, "gradient attribution", { class: "f1-kicker", "text-anchor": "middle" });
  txt(box, boxX + boxW / 2, boxY + 46, "Bergson / TrackStar", { class: "f1-title", "font-size": 15, "text-anchor": "middle" });
  txt(box, boxX + boxW / 2, boxY + 68, "signed score per", { class: "f1-label", "font-size": 11, "text-anchor": "middle" });
  txt(box, boxX + boxW / 2, boxY + 83, "bin × benchmark", { class: "f1-label", "font-size": 11, "text-anchor": "middle" });
  var gradNote = txt(g, boxX + boxW / 2, boxY + boxH + 22, "doc gradients: OLMo3-7B Base", { class: "f1-label", "font-size": 10, fill: C.faint, "text-anchor": "middle", opacity: 0 });
  var gradNote2 = txt(g, boxX + boxW / 2, boxY + boxH + 36, "query gradients: OLMo3-7B Instruct", { class: "f1-label", "font-size": 10, fill: C.faint, "text-anchor": "middle", opacity: 0 });

  // ---- right: the six tracked bins ----
  txt(g, binX + binW / 2, binY0 - 14, "six tracked bins of 576", { class: "f1-kicker", "text-anchor": "middle" });
  var binChips = BINS.map(function (b, i) {
    var y = binY0 + i * binPitch;
    var bg = el("g", { opacity: 0 }, g);
    el("rect", { x: binX, y: y, width: binW, height: binH, rx: 3, fill: "#ffffff", stroke: C.hairline, "stroke-width": 1.1 }, bg);
    txt(bg, binX + 12, y + binH / 2 + 3.5, b.id, { fill: C.ink, "font-size": 10.5, "font-weight": 700 });
    txt(bg, binX + 36, y + binH / 2 + 3.5, b.short, { fill: C.slate, "font-size": 10.5 });
    return { g: bg };
  });

  g._enter = function (gsap) {
    var tl = gsap.timeline();
    gsap.set(chips.map(function (c) { return c.g; }), { opacity: 0 });
    gsap.set(box, { opacity: 0 });
    gsap.set([gradNote, gradNote2], { opacity: 0 });
    gsap.set(binChips.map(function (b) { return b.g; }), { opacity: 0 });
    tl.to(chips.map(function (c) { return c.g; }), { opacity: 1, duration: 0.35, stagger: 0.1 }, 0)
      .to(box, { opacity: 1, duration: 0.4 }, 0.5);
    inBeams.forEach(function (bm, i) {
      var len = bm.getTotalLength();
      bm.setAttribute("stroke-dasharray", len);
      bm.setAttribute("stroke-dashoffset", len);
      bm.setAttribute("opacity", 0.85);
      tl.to(bm, { attr: { "stroke-dashoffset": 0 }, duration: 0.5, ease: "power2.out" }, 0.8 + i * 0.14);
    });
    tl.to([gradNote, gradNote2], { opacity: 1, duration: 0.4, stagger: 0.1 }, 1.4);
    outBeams.forEach(function (bm, i) {
      var len = bm.getTotalLength();
      bm.setAttribute("stroke-dasharray", len);
      bm.setAttribute("stroke-dashoffset", len);
      bm.setAttribute("opacity", 0.6);
      tl.to(bm, { attr: { "stroke-dashoffset": 0 }, duration: 0.45, ease: "power2.out" }, 1.7 + i * 0.09);
    });
    tl.to(binChips.map(function (b) { return b.g; }), { opacity: 1, duration: 0.3, stagger: 0.08 }, 1.9);
    return tl;
  };
  return g;
}

function heatScene(root) {
  var g = el("g", { opacity: 0 }, root);
  var data = INFLUENCE;
  var nCol = data.cols.length, nRow = data.rows.length;
  var cw = 96, ch = 40, gap = 4;
  var labelGutter = 168;
  var gridW = nCol * (cw + gap) - gap;
  var x0 = (VB_W - (labelGutter + gridW)) / 2 + labelGutter;
  var gridH = nRow * (ch + gap) - gap;
  var y0 = (VB_H + 52 - gridH) / 2;
  txt(g, VB_W / 2, 30, "Influence map: signed z per bin", { class: "f1-title", "font-size": 15, "text-anchor": "middle" });
  txt(g, VB_W / 2, 48, "OLMo3-7B", { class: "f1-kicker", "text-anchor": "middle" });

  data.cols.forEach(function (c, ci) {
    txt(g, x0 + ci * (cw + gap) + cw / 2, y0 - 10, c, { class: "f1-label", "font-size": 10, "text-anchor": "middle" });
  });
  var cellObjs = [];
  data.rows.forEach(function (row, ri) {
    var y = y0 + ri * (ch + gap);
    el("circle", { cx: x0 - labelGutter + 12, cy: y + ch / 2, r: 4, fill: row.color }, g);
    txt(g, x0 - labelGutter + 22, y + ch / 2 + 4, row.label, { class: "f1-label", "font-size": 12, "font-weight": 600 });
    row.v.forEach(function (val, ci) {
      var x = x0 + ci * (cw + gap);
      var rect = el("rect", { x: x, y: y, width: cw, height: ch, rx: 4, fill: "#f4f4f4", opacity: 0 }, g);
      if (row.sig === ci) { rect.setAttribute("stroke", C.socReason); rect.setAttribute("stroke-width", 0); }
      var label = txt(g, x + cw / 2, y + ch / 2 + 4, "", { class: "f1-cellval", "text-anchor": "middle", opacity: 0 });
      cellObjs.push({
        rect: rect, label: label, val: val,
        color: inflColor(val), ink: inflInk(val),
        sig: row.sig === ci
      });
    });
  });
  txt(g, VB_W / 2, y0 + gridH + 28, "blue = supportive, orange = suppressive, color capped at |z| = 3",
    { class: "f1-label", "font-size": 10, fill: C.faint, "text-anchor": "middle" });

  g._enter = function (gsap) {
    var tl = gsap.timeline();
    cellObjs.forEach(function (o, i) {
      var delay = 0.03 * i;
      tl.to(o.rect, { opacity: 1, fill: o.color, duration: 0.35 }, delay);
      tl.to(o.label, { opacity: 1, duration: 0.25 }, delay + 0.1);
      var proxy = { n: 0 };
      tl.to(proxy, {
        n: o.val, duration: 0.5,
        onUpdate: function () { o.label.textContent = fmt(proxy.n, true, 1); o.label.setAttribute("fill", o.ink); }
      }, delay + 0.1);
      if (o.sig) tl.to(o.rect, { attr: { "stroke-width": 3 }, duration: 0.3 }, delay + 0.5);
    });
    return tl;
  };
  return g;
}

function barScene(root) {
  var g = el("g", { opacity: 0 }, root);
  var x0 = 250, maxW = 430, h = 28, pitch = 44;
  var maxV = UNLEARN[0].v;
  var y0 = 92;
  txt(g, VB_W / 2, 30, "Unlearning check: SocialIQA", { class: "f1-title", "font-size": 15, "text-anchor": "middle" });
  txt(g, VB_W / 2, 48, "Δ accuracy damage: influence-targeted minus random, in pp (OLMo3-7B)", { class: "f1-kicker", "text-anchor": "middle" });

  var bars = UNLEARN.map(function (row, i) {
    var y = y0 + i * pitch;
    txt(g, x0 - 14, y + h / 2 + 4, row.label, { class: "f1-label", "font-size": 12, "font-weight": 600, "text-anchor": "end" });
    var w = row.v / maxV * maxW;
    var rect = el("rect", { x: x0, y: y, width: 0, height: h, rx: 3, fill: mix("#eff3ff", "#2F6FA8", 0.8) }, g);
    var label = txt(g, x0 + 10, y + h / 2 + 4, "", { class: "f1-cellval", "font-size": 11.5, fill: C.ink, opacity: 0 });
    return { rect: rect, label: label, w: w, v: row.v, y: y };
  });
  // baseline drawn after the bars so it squares off their left edge
  el("line", { x1: x0, y1: y0 - 8, x2: x0, y2: y0 + UNLEARN.length * pitch - (pitch - h) + 8, stroke: C.slate, "stroke-width": 1.2 }, g);
  txt(g, VB_W / 2, y0 + UNLEARN.length * pitch + 24, "top topics shown (paired Wilcoxon, BH-adjusted, p ≈ 10⁻⁵)",
    { class: "f1-label", "font-size": 10, fill: C.faint, "text-anchor": "middle" });

  g._enter = function (gsap) {
    var tl = gsap.timeline();
    bars.forEach(function (b, i) {
      var at = 0.15 * i;
      tl.to(b.rect, { attr: { width: b.w }, duration: 0.6, ease: "power2.out" }, at);
      var proxy = { n: 0 };
      tl.to(b.label, { opacity: 1, duration: 0.2 }, at + 0.25);
      tl.to(proxy, {
        n: b.v, duration: 0.55,
        onUpdate: function () {
          b.label.textContent = fmt(proxy.n, true, 2) + " pp";
          b.label.setAttribute("x", x0 + Math.max(proxy.n / b.v, 0.01) * b.w + 10);
        }
      }, at + 0.1);
    });
    return tl;
  };
  return g;
}

// ================= controller =================
function build(container) {
  var fallback = container.querySelector(".fig1-fallback");
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var gsap = window.gsap;
  if (reduce || !gsap) { if (fallback) fallback.style.display = "block"; return; }

  // scaffold
  var interactive = document.createElement("div");
  interactive.className = "fig1-interactive";

  // stage buttons
  var stagesBar = document.createElement("div");
  stagesBar.className = "fig1-stages";
  var stageBtns = STAGES.map(function (s, i) {
    var b = document.createElement("button");
    b.className = "fig1-stage-btn";
    b.innerHTML = '<span class="num">' + s.icon + '</span><span>' + s.title + '</span>';
    b.addEventListener("click", function () { goTo(i, true); });
    stagesBar.appendChild(b);
    return b;
  });
  interactive.appendChild(stagesBar);

  // svg
  var svg = el("svg", { class: "fig1-svg", viewBox: "0 0 " + VB_W + " " + VB_H, role: "img", "aria-label": "Animated capability provenance pipeline" });
  var stageWrap = document.createElement("div");
  stageWrap.className = "fig1-stage-wrap";
  stageWrap.appendChild(svg);
  interactive.appendChild(stageWrap);

  var caption = document.createElement("p");
  caption.className = "fig1-caption";
  interactive.appendChild(caption);

  // controls
  var controls = document.createElement("div");
  controls.className = "fig1-controls";
  var prevBtn = ctlBtn("❮", "Previous stage");
  var playBtn = ctlBtn("❚❚", "Play or pause");
  var nextBtn = ctlBtn("❯", "Next stage");
  var prog = document.createElement("div"); prog.className = "fig1-progress";
  var progBar = document.createElement("div"); progBar.className = "fig1-progress-bar"; prog.appendChild(progBar);
  controls.appendChild(prevBtn); controls.appendChild(playBtn); controls.appendChild(prog); controls.appendChild(nextBtn);
  interactive.appendChild(controls);

  container.insertBefore(interactive, fallback);
  if (fallback) fallback.style.display = "none";

  // Reserve the height of the tallest caption so stage changes never reflow
  // the page (the caption block otherwise grows and shrinks per stage).
  function lockCaptionHeight() {
    var cur = caption.innerHTML;
    var max = 0;
    CAPTIONS.forEach(function (c) {
      caption.innerHTML = c;
      max = Math.max(max, caption.offsetHeight);
    });
    caption.innerHTML = cur;
    caption.style.minHeight = max + "px";
  }
  lockCaptionHeight();
  window.addEventListener("resize", lockCaptionHeight);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(lockCaptionHeight);

  // build scenes
  var scenes = [
    sceneCorpus(svg),
    sceneAttribution(svg),
    heatScene(svg),
    barScene(svg)
  ];

  var cur = -1, playing = true, tl = null, advance = null, progTween = null;
  var started = false, autoPaused = false;
  var DWELL = 0.9;

  function ctlBtn(label, aria) {
    var b = document.createElement("button");
    b.className = "fig1-btn"; b.innerHTML = label; b.setAttribute("aria-label", aria);
    return b;
  }

  function goTo(i, manual) {
    started = true;
    if (i < 0) i = scenes.length - 1;
    if (i >= scenes.length) i = 0;
    if (tl) tl.kill();
    if (advance) advance.kill();
    if (progTween) progTween.kill();
    scenes.forEach(function (s, idx) { if (idx !== i) gsap.set(s, { opacity: 0 }); });
    stageBtns.forEach(function (b, idx) { b.classList.toggle("is-active", idx === i); });
    caption.innerHTML = CAPTIONS[i];
    cur = i;
    gsap.set(scenes[i], { opacity: 1 });
    tl = scenes[i]._enter(gsap);
    // progress across full cycle
    var start = i / scenes.length * 100, end = (i + 1) / scenes.length * 100;
    gsap.set(progBar, { width: start + "%" });
    progTween = gsap.to(progBar, { width: end + "%", duration: (tl.duration() + DWELL), ease: "none" });
    if (manual) setPlaying(false);
    else scheduleAdvance();
  }

  function scheduleAdvance() {
    if (advance) advance.kill();
    advance = gsap.delayedCall(tl.duration() + DWELL, function () { goTo(cur + 1, false); });
  }

  function setPlaying(p) {
    playing = p;
    playBtn.innerHTML = p ? "❚❚" : "▶";
    if (p) { if (tl) tl.play(); if (progTween) progTween.play(); scheduleAdvance(); }
    else { if (advance) advance.kill(); if (progTween) progTween.pause(); }
  }

  prevBtn.addEventListener("click", function () { goTo(cur - 1, true); });
  nextBtn.addEventListener("click", function () { goTo(cur + 1, true); });

  // Scrub: click/drag the progress bar to seek to the matching stage
  function seekFromEvent(e) {
    var rect = prog.getBoundingClientRect();
    var px = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    var f = Math.max(0, Math.min(0.9999, px / rect.width));
    var stage = Math.floor(f * scenes.length);
    if (stage !== cur) goTo(stage, true);
  }
  var scrubbing = false;
  prog.addEventListener("mousedown", function (e) { scrubbing = true; seekFromEvent(e); e.preventDefault(); });
  window.addEventListener("mousemove", function (e) { if (scrubbing) seekFromEvent(e); });
  window.addEventListener("mouseup", function () { scrubbing = false; });
  prog.addEventListener("click", seekFromEvent);
  prog.setAttribute("role", "slider");
  prog.setAttribute("aria-label", "Animation stage");
  playBtn.addEventListener("click", function () {
    if (playing) setPlaying(false);
    else { setPlaying(true); if (advance) advance.kill(); scheduleAdvance(); }
  });

  // pause on hover for readability
  stageWrap.addEventListener("mouseenter", function () { if (playing && advance) advance.pause(); });
  stageWrap.addEventListener("mouseleave", function () { if (playing && advance) advance.resume(); });

  var params = new URLSearchParams(window.location.search);
  var startStage = parseInt(params.get("fig1stage"), 10);
  if (startStage >= 1 && startStage <= scenes.length) {
    goTo(startStage - 1, true);
    if (params.get("fig1end") && tl) tl.progress(1); // jump to final frame (screenshot aid)
  } else if ("IntersectionObserver" in window) {
    // Start only once scrolled into view, so the viewer sees stage 1 from the
    // top. Pause while off screen, resume where it left off when back.
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          if (!started) goTo(0, false);
          else if (autoPaused) { autoPaused = false; setPlaying(true); }
        } else if (started && playing) {
          autoPaused = true;
          setPlaying(false);
        }
      });
    }, { threshold: 0.25 });
    io.observe(container);
  } else goTo(0, false);
}

function init() {
  var nodes = document.querySelectorAll("[data-fig1]");
  nodes.forEach(function (n) { build(n); });
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
else init();
