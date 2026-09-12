"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(root, "web_experiment", "index.html"), "utf8");
const script = html.match(/<script>([\s\S]*)<\/script>/)[1];
new Function(script);

assert.ok(!html.includes("const TOPICS ="), "materials must not be hard-coded in the HTML");
assert.ok(html.includes("hydrateProtocol(info.protocol_config)"), "browser must consume the recorder protocol config");
assert.ok(html.includes("courseAttentionRating") && html.includes("mentalEffort"));
assert.ok(html.includes("videoInterest") && html.includes("videoDifficulty"));
assert.ok(html.includes("subtractionCompliance") && html.includes("reportedFinalNumber"));
assert.ok(html.includes('recordEvent("rating_item_response"'));
assert.ok(html.includes('recordEvent("quiz_item_response"'));
assert.ok(html.includes('recordEvent("self_caught"'));
assert.ok(html.includes('k==="k"'));
assert.ok(html.includes('pause_reason:"eeg_dropout"'));
assert.ok(html.includes("确认同步并继续"));
assert.ok(html.includes("videoMap.size===6"));
assert.ok(html.includes("planned_blocks[blockNum-1]"));
assert.ok(html.includes("durationSec+0.25<Number(restPlannedDuration)"));
assert.ok(html.includes("视频时长不足，无法容纳"));
assert.ok(!html.includes("line_50hz_ratio_pct"));
assert.ok(!html.includes("task_a"));
assert.ok(html.includes('recordEvent("browser_recovery_blocked"'));
assert.ok(html.includes("recoveryBlocked=true;lockSetup(true)"));
assert.ok(html.includes("if(selectedVideoFiles.length)matchSelectedVideos()"));
assert.ok(html.includes('normalized.match(/^视频([1-6])'));
assert.ok(html.includes('class="videoMatchGrid"'));
assert.ok(html.includes('status.textContent=item?"已匹配"'));
console.log("22 browser protocol checks passed");
