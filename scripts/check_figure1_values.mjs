#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const publicRoot = path.resolve(scriptDir, "..");
const privateRoot = path.resolve(publicRoot, "..", "capabilibara");
const canonicalRoot = path.resolve(
  process.env.FIGURE1_CANONICAL_REPO || publicRoot,
);

const benchmarkFiles = [
  ["SocialIQA", "zscored_socialiqa.csv"],
  ["MMLU SS", "zscored_mmlu_social_science.csv"],
  ["ARC-Chal.", "zscored_arc_challenge.csv"],
  ["MMLU STEM", "zscored_mmlu_stem.csv"],
];
const bins = [
  ["literature", "customer_support"],
  ["social_life", "q_a_forum"],
  ["travel_and_tourism", "q_a_forum"],
  ["science_math_and_technology", "academic_writing"],
  ["industrial", "documentation"],
  ["politics", "documentation"],
];
const benchmarkQueries = ["SocialIQA", "MMLU Social Sciences", "ARC-Challenge", "MMLU STEM"];

function fail(message) {
  throw new Error(message);
}

function readCanonicalScores(fileName) {
  const filePath = path.join(canonicalRoot, "artifacts", "zscored_bin_scores", "aggregated", fileName);
  const lines = fs.readFileSync(filePath, "utf8").trim().split(/\r?\n/);
  const header = lines.shift().split(",");
  const topicIndex = header.indexOf("topic_label");
  const formatIndex = header.indexOf("format_label");
  const zscoreIndex = header.indexOf("zscore");
  if (topicIndex < 0 || formatIndex < 0 || zscoreIndex < 0) fail(`${fileName}: missing canonical columns`);
  if (lines.length !== 576) fail(`${fileName}: expected 576 rows, found ${lines.length}`);

  const values = new Map();
  for (const line of lines) {
    const columns = line.split(",");
    const key = `${columns[topicIndex]}:${columns[formatIndex]}`;
    values.set(key, Number(columns[zscoreIndex]));
  }
  return values;
}

function readSceneData(filePath) {
  const source = fs.readFileSync(filePath, "utf8");
  const block = source.match(/var INFLUENCE = \{([\s\S]*?)\n\s*\};/);
  if (!block) fail(`${filePath}: could not locate INFLUENCE`);
  const rows = [...block[1].matchAll(/\{\s*label:\s*"([^"]+)"[\s\S]*?v:\s*\[([^\]]+)\]/g)]
    .map((match) => ({ label: match[1], values: match[2].split(",").map(Number) }));
  if (rows.length !== benchmarkFiles.length) fail(`${filePath}: expected 4 influence rows, found ${rows.length}`);
  const caption = source.match(/var CAPTIONS = \[([\s\S]*?)\n\s*\];/);
  if (!caption) fail(`${filePath}: could not locate CAPTIONS`);
  return { rows, caption: caption[1] };
}

function checkScene(filePath) {
  const { rows, caption } = readSceneData(filePath);
  const expectedLabels = benchmarkFiles.map(([label]) => label);
  if (JSON.stringify(rows.map((row) => row.label)) !== JSON.stringify(expectedLabels)) {
    fail(`${filePath}: influence rows must label ${expectedLabels.join(", ")}`);
  }
  for (const query of benchmarkQueries) {
    if (!caption.includes(query)) fail(`${filePath}: stage-3 caption does not name ${query}`);
  }
  if (!caption.includes("aggregates all queries from one benchmark")
      || !caption.includes("including correct and incorrect answers")) {
    fail(`${filePath}: stage-3 caption must state that each row uses the full query cohort`);
  }

  for (let rowIndex = 0; rowIndex < benchmarkFiles.length; rowIndex += 1) {
    const [label, fileName] = benchmarkFiles[rowIndex];
    const canonical = readCanonicalScores(fileName);
    if (rows[rowIndex].values.length !== bins.length) {
      fail(`${filePath}: ${label} must contain 6 tracked-bin values`);
    }
    for (let binIndex = 0; binIndex < bins.length; binIndex += 1) {
      const key = bins[binIndex].join(":");
      const expected = canonical.get(key);
      const actual = rows[rowIndex].values[binIndex];
      if (!Number.isFinite(expected) || !Number.isFinite(actual) || Math.abs(actual - expected) > 1e-12) {
        fail(`${filePath}: ${label} ${key} is ${actual}, expected canonical ${expected}`);
      }
    }
  }
  console.log(`${filePath}: 24 canonical z-scores and four query labels match`);
}

const scenePaths = process.argv.slice(2);
if (scenePaths.length === 0) {
  scenePaths.push(path.join(publicRoot, "public", "static", "js", "figure1.js"));
  const privateMirror = path.join(privateRoot, "website", "static", "js", "figure1.js");
  if (fs.existsSync(privateMirror)) scenePaths.push(privateMirror);
}
for (const scenePath of scenePaths) checkScene(path.resolve(scenePath));
