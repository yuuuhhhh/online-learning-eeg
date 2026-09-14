"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { randomUUID } = require("node:crypto");

const html = fs.readFileSync(path.join(__dirname, "../web_experiment/index.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const bootstrap = "setupQuestionnaireControls();updateGroupUi();updateConditionUi();updateTopicUi();updateProbeUi();renderVideoMatchPanel();discoverEegRecorder();";
assert.ok(script.includes(bootstrap), "shipped bootstrap line changed; update the recovery harness");
const body = script.replace(bootstrap, "");

const plannedBlocks = Array.from({ length: 6 }, (_, index) => ({
  block_id: index + 1,
  condition_label: index % 2 ? "B" : "A",
  video_id: `V${index + 1}`,
}));
const topics = Object.fromEntries(plannedBlocks.map(block => [block.video_id, {
  name: block.video_id,
  questions: Array.from({ length: 3 }, (_, index) => ({ question_id: `${block.video_id}Q${index + 1}` })),
}]));

function storage() {
  const values = new Map();
  return {
    getItem: key => values.get(key) || null,
    setItem: (key, value) => values.set(key, value),
    removeItem: key => values.delete(key),
  };
}

function harness() {
  const elements = new Map();
  const initial = { subjectId: "sub-test", sessionId: "ses-001", blockId: "Block 1", condition: "A", topicSelect: "V1" };
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      value: initial[id] || "", textContent: "", innerHTML: "", disabled: false, style: {}, dataset: {}, children: [],
      classList: { add() {}, remove() {}, toggle() {} }, addEventListener() {}, appendChild() {}, append() {},
      removeAttribute() {}, load() {}, closest() { return { style: {} } }, paused: true, ended: false,
      currentTime: 0, duration: 300, pause() { this.paused = true }, async play() { this.paused = false },
    });
    return elements.get(id);
  };
  const context = vm.createContext({
    console,
    crypto: { randomUUID },
    performance: { timeOrigin: 1_000_000, now: () => 100 },
    localStorage: storage(), sessionStorage: storage(), AbortController, URL,
    setTimeout: () => 1, clearTimeout() {}, setInterval: () => 1, clearInterval() {},
    alert() {}, confirm: () => true,
    document: {
      getElementById: element, querySelectorAll: () => [...elements.values()], querySelector: () => null,
      addEventListener() {}, createElement: () => element(randomUUID()),
    },
    window: { addEventListener() {}, scrollTo() {} },
    fetch: async () => ({ ok: true, json: async () => ({ ok: true }) }),
  });
  vm.runInContext(body, context);
  const run = code => vm.runInContext(code, context);
  run(`TOPICS=${JSON.stringify(topics)}`);
  return {
    run,
    apply: info => run(`applyEegSession(${JSON.stringify(info)})`),
    addEvents: records => run(`events.push(...${JSON.stringify(records)})`),
    recoveryEvents: () => JSON.parse(run('JSON.stringify(events.filter(event=>event.event_type==="browser_recovery_blocked"))')),
  };
}

function session(runId, { currentBlock = 0, completedBlocks = [1], phase = "idle" } = {}) {
  return {
    run_id: runId, run_directory: runId, subject_id: "sub-test", session_id: "ses-001",
    counterbalance_group: "G01", study_phase: "pilot", planned_blocks: plannedBlocks,
    planned_sequence: plannedBlocks.map(block => block.condition_label), current_block: currentBlock,
    completed_blocks: completedBlocks, phase, qc_ready_for_experiment: true,
    qc: { status: "good", signal_alive: true, messages: [], channels: [] },
  };
}

function completedEvents(runId, blockId = "Block 1") {
  return [
    { recorder_run_id: runId, event_type: "rating_end", block_id: blockId },
    ...topics.V1.questions.map(question => ({
      recorder_run_id: runId, event_type: "quiz_item_response", block_id: blockId, question_id: question.question_id,
    })),
    { recorder_run_id: runId, event_type: "quiz_end", block_id: blockId, total: topics.V1.questions.length },
  ];
}

let passed = 0;
function test(name, callback) {
  callback();
  passed++;
  console.log("PASS", name);
}

test("blockNumber normalizes every supported block id representation", () => {
  const h = harness();
  for (const value of [1, "1", "Block 1", "block-1"]) {
    assert.equal(h.run(`blockNumber(${JSON.stringify(value)})`), 1);
  }
});

test("complete rating and dynamically-sized quiz do not block recovery", () => {
  const h = harness();
  h.addEvents(completedEvents("run-complete"));
  h.apply(session("run-complete"));
  assert.equal(h.run("recoveryBlocked"), false);
  assert.equal(h.recoveryEvents().length, 0);
});

test("missing rating, a unique quiz item, or quiz_end blocks recovery", () => {
  const cases = [
    records => records.filter(event => event.event_type !== "rating_end"),
    records => records.filter(event => event.question_id !== "V1Q3"),
    records => records.filter(event => event.event_type !== "quiz_end"),
  ];
  for (const [index, alter] of cases.entries()) {
    const runId = `run-incomplete-${index}`;
    const h = harness();
    h.addEvents(alter(completedEvents(runId)));
    h.apply(session(runId));
    assert.equal(h.run("recoveryBlocked"), true);
    assert.equal(h.recoveryEvents().length, 1);
  }
});

test("duplicate question ids do not satisfy quiz completeness", () => {
  const h = harness();
  const records = completedEvents("run-duplicate");
  records.find(event => event.question_id === "V1Q3").question_id = "V1Q2";
  h.addEvents(records);
  h.apply(session("run-duplicate"));
  assert.equal(h.run("recoveryBlocked"), true);
});

test("normal rest_end transition and repeated polling never invoke recovery again", () => {
  const h = harness();
  const runId = "run-normal-transition";
  h.apply(session(runId, { completedBlocks: [] }));
  h.addEvents(completedEvents(runId));
  h.run(`blockStarted=true;blockFinished=true;conditionAtStart="A";topicAtStart="V1";
    configAtStart={recorder_run_id:"${runId}",subject_id:"sub-test",session_id:"ses-001",block_id:"Block 1",counterbalance_group:"G01",condition:"A",condition_type:"focused",topic_key:"V1"};
    restingAfterBlock=1;restPlannedDuration=30;restStartedPerf=-30000;finishBlockRest()`);
  const next = session(runId);
  h.apply(next); h.apply(next); h.apply(next);
  assert.equal(h.run("recoveryBlocked"), false);
  assert.equal(h.recoveryEvents().length, 0);
  assert.equal(h.run('$("blockId").value'), "Block 2");
});

test("same run is checked once while a new run is checked again", () => {
  const h = harness();
  h.addEvents(completedEvents("run-one"));
  h.apply(session("run-one"));
  h.run('events.splice(0,events.length)');
  h.apply(session("run-one"));
  assert.equal(h.run("recoveryBlocked"), false);
  h.apply(session("run-two"));
  assert.equal(h.run("recoveryBlocked"), true);
  assert.equal(h.recoveryEvents().length, 1);
});

test("a fresh JavaScript lifecycle still blocks reload during an active block", () => {
  const reloaded = harness();
  reloaded.apply(session("run-active", { currentBlock: 1, completedBlocks: [], phase: "quiz" }));
  assert.equal(reloaded.run("recoveryBlocked"), true);
  assert.equal(reloaded.recoveryEvents().length, 1);
});

test("blocked recovery preserves current UI context and records affected_block_id separately", () => {
  const h = harness();
  h.apply(session("run-context"));
  const event = h.recoveryEvents()[0];
  assert.equal(event.block_id, "Block 2");
  assert.equal(event.block_order, "Block 2");
  assert.equal(event.condition, "B");
  assert.equal(event.topic_key, "V2");
  assert.equal(event.affected_block_id, 1);
});

console.log(`${passed} browser recovery behavior tests passed`);
