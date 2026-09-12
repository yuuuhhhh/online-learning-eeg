// Run with Node.js; exercises the production script with in-memory DOM/network.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { randomUUID } = require("node:crypto");

const html = fs.readFileSync(path.join(__dirname, "../web_experiment/index.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script); // Check the entire shipped script, including bootstrap.
const bootstrap = "populateTopics();buildRatingButtons();buildProbeButtons();renderVideoMatchPanel();updateGroupUi();updateConditionUi();updateTopicUi();updateProbeUi();discoverEegRecorder();";
assert.ok(script.includes(bootstrap), "shipped bootstrap line changed; update the harness");
const body = script.replace(bootstrap, "");

function harness(network = async () => ({ ok: true, json: async () => ({ ok: true }) })) {
  const elements = new Map();
  const initial = { subjectId: "sub-test", sessionId: "ses-001", blockId: "1", condition: "A", topicSelect: "a" };
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      value: initial[id] || "", textContent: "", disabled: false, style: {}, dataset: {},
      classList: { add() {}, remove() {}, toggle() {} }, addEventListener() {},
      appendChild() {}, removeAttribute() {}, load() {}, paused: true, ended: false,
      currentTime: 0, duration: 300, pause() { this.paused = true },
      async play() { this.paused = false },
    });
    return elements.get(id);
  };
  const storage = () => {
    const values = new Map();
    return { getItem: k => values.get(k) || null, setItem: (k, v) => values.set(k, v), removeItem: k => values.delete(k) };
  };
  const requests = [];
  const context = vm.createContext({
    console, crypto: { randomUUID }, performance: { timeOrigin: 1_000_000, now: () => 100 },
    localStorage: storage(), sessionStorage: storage(), AbortController, URL,
    setTimeout: () => 1, clearTimeout() {}, setInterval: () => 1, clearInterval() {},
    alert() {}, confirm: () => true,
    document: { getElementById: element, querySelectorAll: () => [...elements.values()], addEventListener() {}, createElement: () => element(randomUUID()) },
    window: { addEventListener() {}, scrollTo() {} },
    fetch: async (url, options = {}) => {
      const payload = options.body ? JSON.parse(options.body) : null;
      requests.push({ url, payload });
      return network(url, payload);
    },
  });
  vm.runInContext(body, context);
  const run = code => vm.runInContext(code, context);
  run('activeEegSession={run_id:"run-current",run_directory:"run-legacy",subject_id:"sub-test",session_id:"ses-001",planned_sequence:["A","B","A","B","A","B"],order:1,qc_ready_for_experiment:true,qc:{status:"good"}}; eegEndpoint="http://recorder"; eegClockOffset=-0.05; eegClockRoundTripMs=4;');
  return { run, context, requests };
}
const settle = async () => { for (let i = 0; i < 10; i++) await new Promise(setImmediate) };
let passed = 0;
async function test(name, callback) { await callback(); passed++; console.log("PASS", name) }

(async () => {
  await test("event timestamp and immutable block run identity", async () => {
    const h = harness();
    h.run('configAtStart={recorder_run_id:"run-original",subject_id:"sub-original",session_id:"ses-original",block_id:"2",group_id:1};recordEvent("manual_sync_mark")');
    await settle();
    const event = h.requests[0].payload;
    assert.equal(event.recorder_run_id, "run-original");
    assert.equal(event.session_id, "ses-original");
    assert.equal(event.block_id, "2");
    assert.ok(Math.abs(event.event_timestamp - 1000.05) < 1e-9);
    assert.equal(event.clock_round_trip_ms, 4);
    assert.ok(event.client_id);
  });

  await test("semantic rejection retains payload and does not block subsequent events", async () => {
    let attempt = 0;
    const h = harness(async () => ++attempt === 1 ? { ok: false, status: 400, json: async () => ({ error: "wrong run" }) } : { ok: true });
    h.run('recordEvent("first");recordEvent("second")');
    await settle();
    assert.equal(h.run("eegEventQueue.length"), 0);
    assert.equal(h.run("rejectedEegEvents.length"), 1);
    assert.equal(h.run("rejectedEegEvents[0].event.event_type"), "first");
    assert.equal(h.run("JSON.parse(localStorage.getItem(BACKUP_KEY)).rejectedEegEvents.length"), 1);
    assert.equal(h.requests.length, 2);
  });

  await test("shutdown acknowledgement follows all queued writes", async () => {
    const h = harness();
    h.run('recordEvent("last_answer");prepareBrowserShutdown("close-1")');
    await settle();
    assert.deepEqual(h.requests.map(r => r.payload.event_type || "ack"), ["last_answer", "browser_shutdown_ready", "ack"]);
    assert.equal(h.requests[2].payload.last_event_id, h.requests[1].payload.client_event_id);
    assert.equal(h.run("shutdownAcknowledged"), true);
    assert.equal(h.run('recordEvent("too_late")'), null);
    assert.equal(h.run("eegEventQueue.length"), 0);
  });

  await test("network failure preserves queue until recovery before acknowledgement", async () => {
    let offline = true;
    const h = harness(async () => { if (offline) throw Error("offline"); return { ok: true } });
    h.run('recordEvent("last_answer");prepareBrowserShutdown("close-1")');
    await settle();
    assert.equal(h.run("eegEventQueue.length"), 2);
    assert.equal(h.run("shutdownAcknowledged"), false);
    assert.equal(h.requests.some(r => r.url.endsWith("/shutdown/ready")), false);
    offline = false;
    h.run('eegEndpoint="http://recorder";flushEegQueue()');
    await settle();
    assert.equal(h.run("eegEventQueue.length"), 0);
    assert.equal(h.run("shutdownAcknowledged"), true);
  });

  await test("new shutdown request works even if cancellation heartbeat was missed", async () => {
    const h = harness();
    h.run('prepareBrowserShutdown("close-1")');
    await settle();
    h.run('prepareBrowserShutdown("close-2")');
    await settle();
    assert.equal(h.run("shutdownRequestId"), "close-2");
    assert.equal(h.run("shutdownAcknowledged"), true);
    assert.equal(h.requests.filter(r => r.url.endsWith("/shutdown/ready")).length, 2);
  });

  await test("rejected current-run events accompany save acknowledgement for recovery", async () => {
    const h = harness();
    h.run('rejectedEegEvents.push({event:{recorder_run_id:"run-current"},reason:"invalid event"});prepareBrowserShutdown("close-1")');
    await settle();
    assert.equal(h.run("shutdownAcknowledged"), true);
    assert.equal(h.requests.find(r => r.url.endsWith("/shutdown/ready")).payload.unresolved_events.length, 1);
  });

  await test("old-run pending events remain recoverable instead of being discarded", async () => {
    const h = harness();
    h.run('eegEventQueue.push({recorder_run_id:"old-run",event_type:"last_answer"});applyEegSession(activeEegSession)');
    assert.equal(h.run("eegEventQueue.length"), 0);
    assert.equal(h.run("rejectedEegEvents[0].event.event_type"), "last_answer");
  });

  await test("local storage failure does not prevent event delivery", async () => {
    const h = harness();
    h.run('localStorage.setItem=()=>{throw Error("quota exceeded")};recordEvent("answer")');
    await settle();
    assert.equal(h.run("eegEventQueue.length"), 0);
    assert.equal(h.requests.length, 1);
  });

  await test("an in-flight response cannot remove a different queued event", async () => {
    let release;
    let calls = 0;
    const h = harness(async () => {
      calls++;
      if (calls === 1) await new Promise(resolve => release = resolve);
      return { ok: true };
    });
    h.run('recordEvent("old_answer");eegEventQueue[0].recorder_run_id="old-run";recordEvent("new_answer");applyEegSession(activeEegSession)');
    release();
    await settle();
    assert.equal(h.requests.length, 2);
    assert.equal(h.requests[1].payload.event_type, "new_answer");
    assert.equal(h.run("eegEventQueue.length"), 0);
  });

  await test("new browser flow contains no screen arithmetic scheduler or F/J response path", async () => {
    const h = harness();
    assert.equal(h.run('typeof generateArithmetic'), "undefined");
    assert.equal(h.run('typeof scheduleArithmetic'), "undefined");
    assert.equal(h.run('typeof finishArithmetic'), "undefined");
    assert.equal(script.includes("arithmetic_onset"), false);
    assert.equal(script.includes("arithmetic_response"), false);
    assert.equal(html.includes("questionSec"), false);
    assert.equal(html.includes("gapRange"), false);
  });

  await test("practice records start end and confirmation without block context", async () => {
    const h = harness();
    h.run(`applyEegSession({
      run_id:"run-practice",run_directory:"run-practice",subject_id:"sub-test",session_id:"ses-001",
      planned_sequence:["A","B","A","B","A","B"],order:1,qc_ready_for_experiment:true,qc:{status:"good"},
      subtraction_practice:{practice_number:100,started:false,confirmed:false},
      conditions:{A:{condition_type:"focused"},B:{condition_type:"bbbd_subtraction"}},
      b_start_numbers:{2:{start_number:809,generated_timestamp:900}},
      rest_durations_after_block_sec:{1:30,2:30,3:180,4:30,5:30}
    });confirmSubtractionPractice()`);
    await settle();
    const practice = h.requests.map(r => r.payload).filter(Boolean).filter(r => r.event_type?.includes("practice") || r.event_type === "practice_confirmed");
    assert.deepEqual(practice.map(r => r.event_type), ["subtraction_practice_start", "subtraction_practice_end", "practice_confirmed"]);
    assert.ok(practice.every(r => r.block_id === "" && r.condition === "" && r.is_formal_experiment === 0));
    assert.equal(h.run("practiceConfirmed"), true);
  });

  await test("B start number is displayed before playback and hidden when playback starts", async () => {
    const h = harness();
    h.run(`conditionAtStart="B";topicAtStart="a";blockStarted=true;blockFinished=false;
      configAtStart={recorder_run_id:"run-current",subject_id:"sub-test",session_id:"ses-001",block_id:"2",group_id:1,
        condition:"B",condition_type:"bbbd_subtraction",topic_key:"a",start_number:809,generated_timestamp:900};
      showConditionOverlay()`);
    assert.equal(h.run('$("startNumberDisplay").textContent'), "809");
    assert.equal(h.run('$("startNumberDisplay").style.display'), "block");
    await h.run("playVideoAfterPrompt()");
    await settle();
    assert.equal(h.run('$("startNumberDisplay").style.display'), "none");
    const sent = h.requests.map(r => r.payload).filter(Boolean);
    assert.deepEqual(sent.map(r => r.event_type), [
      "subtraction_start_number_displayed", "video_play", "subtraction_start_number_hidden"
    ]);
    assert.equal(sent[0].start_number, 809);
    assert.equal(sent[2].start_number, 809);
    assert.ok(Number.isFinite(sent[0].displayed_timestamp));
    assert.ok(Number.isFinite(sent[2].hidden_timestamp));
  });

  await test("rest schedule is 30 30 180 30 30 and debug skip records actual duration", async () => {
    const h = harness();
    h.run('activeEegSession.rest_durations_after_block_sec={1:30,2:30,3:180,4:30,5:30};configAtStart={recorder_run_id:"run-current",subject_id:"sub-test",session_id:"ses-001",block_id:"3",group_id:2,condition:"A",condition_type:"focused",topic_key:"c"}');
    assert.deepEqual(Array.from(h.run('[1,2,3,4,5].map(plannedRestAfterBlock)')), [30, 30, 180, 30, 30]);
    h.run("beginBlockRest(3);finishBlockRest()");
    await settle();
    const rests = h.requests.map(r => r.payload).filter(r => r?.event_type?.startsWith("rest_"));
    assert.deepEqual(rests.map(r => r.event_type), ["rest_start", "rest_end"]);
    assert.ok(rests.every(r => r.after_block_id === 3 && r.planned_duration === 180));
    assert.equal(typeof rests[1].actual_duration, "number");
  });
  // ---- thought probe (stage 3) ----
  const probeHarness = (condition = "A") => {
    const h = harness();
    h.run(`conditionAtStart="${condition}";topicAtStart="a";blockStarted=true;blockFinished=false;
      configAtStart={recorder_run_id:"run-current",subject_id:"sub-test",session_id:"ses-001",block_id:"3",group_id:2,
        condition:"${condition}",condition_type:"${condition === "A" ? "focused" : "bbbd_subtraction"}",topic_key:"a"};
      prepareProbeSchedule()`);
    return h;
  };
  const sentTypes = h => h.requests.map(r => r.payload).filter(Boolean).map(r => r.event_type);
  const reachProbe = h => h.run('$("video").paused=false;$("video").currentTime=probeSchedule.times[probeIndex];maybeTriggerProbe()');

  await test("each block schedules four probes inside the allowed video window", async () => {
    const h = probeHarness();
    const times = h.run("probeSchedule.times.slice()");
    const duration = h.run("probeSchedule.video_duration_sec");
    assert.equal(times.length, 4);
    assert.ok(times[0] >= 45, `first probe at ${times[0]}s`);
    assert.ok(times[times.length - 1] <= duration - 30, `last probe at ${times[times.length - 1]}s`);
    for (let i = 1; i < times.length; i++) assert.ok(times[i] - times[i - 1] >= 45, `gap ${times[i] - times[i - 1]}s`);
    assert.equal(h.run("probeSchedule.debug_fast"), false);
  });

  await test("probe times are redrawn per block instead of reusing fixed offsets", async () => {
    const schedules = Array.from({ length: 6 }, () => probeHarness().run("probeSchedule.times.join(',')"));
    assert.ok(new Set(schedules).size > 1, `all schedules identical: ${schedules[0]}`);
  });

  await test("debug fast configuration keeps the constraints on a short test clip", async () => {
    const h = harness();
    h.run(`$("probeFastMode").checked=true;$("video").duration=40;
      configAtStart={recorder_run_id:"run-current",subject_id:"sub-test",session_id:"ses-001",block_id:"3",group_id:2,topic_key:"a"};
      prepareProbeSchedule()`);
    const times = h.run("probeSchedule.times.slice()");
    assert.equal(h.run("probeSchedule.debug_fast"), true);
    assert.equal(times.length, 4);
    assert.ok(times[0] >= 5 && times[times.length - 1] <= 37);
    for (let i = 1; i < times.length; i++) assert.ok(times[i] - times[i - 1] >= 5);
  });

  await test("a paused video cannot advance to the next probe", async () => {
    const h = probeHarness();
    h.run('$("video").paused=true;$("video").currentTime=probeSchedule.times[3]+10;maybeTriggerProbe()');
    assert.equal(h.run("activeProbe===null"), true);
    assert.equal(h.run("probeIndex"), 0);
    assert.equal(sentTypes(h).includes("probe_onset"), false);
  });

  await test("reaching a probe time pauses the video and records probe_onset", async () => {
    const h = probeHarness();
    reachProbe(h);
    await settle();
    assert.equal(h.run('$("video").paused'), true);
    assert.match(h.run("activeProbe.probe_id"), /^3_1_[a-z0-9]{8}$/i);
    const onset = h.requests.map(r => r.payload).find(r => r?.event_type === "probe_onset");
    assert.equal(onset.pause_reason, "thought_probe");
    assert.equal(onset.probe_index, 1);
    assert.equal(onset.block_id, "3");
    assert.equal(onset.video_id, "a");
    assert.ok(onset.probe_schedule_id);
    assert.ok(Number.isFinite(onset.event_timestamp));
  });

  await test("confidence cannot be submitted before the attention question is answered", async () => {
    const h = probeHarness();
    reachProbe(h);
    assert.equal(h.run("submitProbeConfidence(3)"), null);
    assert.equal(h.run("activeProbe.response===null"), true);
    await settle();
    assert.equal(sentTypes(h).includes("confidence_response"), false);
    assert.equal(h.run('$("video").paused'), true);
  });

  await test("confidence only accepts the integers 1 to 4", async () => {
    const h = probeHarness();
    reachProbe(h);
    h.run("answerProbeAttention(1)");
    for (const value of [0, 5, 2.5, "x"]) assert.equal(h.run(`submitProbeConfidence(${JSON.stringify(value)})`), null);
    assert.ok(h.run("submitProbeConfidence(4)"));
    await settle();
    const confidence = h.requests.map(r => r.payload).filter(r => r?.event_type === "confidence_response");
    assert.equal(confidence.length, 1);
    assert.equal(confidence[0].confidence, 4);
  });

  await test("submitting confidence resumes the video from the same position", async () => {
    const h = probeHarness();
    reachProbe(h);
    const resumeAt = h.run('$("video").currentTime');
    h.run("answerProbeAttention(3);submitProbeConfidence(2)");
    await settle();
    assert.equal(h.run('$("video").paused'), false);
    assert.equal(h.run('$("video").currentTime'), resumeAt);
    assert.equal(h.run("activeProbe===null"), true);
    assert.equal(h.run("probeIndex"), 1);
    const resume = h.requests.map(r => r.payload).find(r => r?.event_type === "video_resume");
    assert.equal(resume.pause_reason, "thought_probe");
    assert.match(resume.probe_id, /^3_1_/);
  });

  await test("repeating a block during debugging does not reuse probe identifiers", async () => {
    const h = probeHarness();
    reachProbe(h);
    const first = h.run("activeProbe.probe_id");
    h.run("answerProbeAttention(1);submitProbeConfidence(1)");
    await settle();
    h.run("prepareProbeSchedule()");
    reachProbe(h);
    const second = h.run("activeProbe.probe_id");
    assert.ok(first.startsWith("3_1_") && second.startsWith("3_1_"));
    assert.notEqual(first, second);
  });

  await test("the four options map to ON OFF OFF AMBIGUOUS without condition correction", async () => {
    const expected = ["ON", "OFF", "OFF", "AMBIGUOUS"];
    for (const condition of ["A", "B"]) {
      for (let response = 1; response <= 4; response++) {
        const h = probeHarness(condition);
        reachProbe(h);
        h.run(`answerProbeAttention(${response})`);
        await settle();
        const answer = h.requests.map(r => r.payload).find(r => r?.event_type === "attention_response");
        assert.equal(answer.probe_attention, expected[response - 1], `${condition} option ${response}`);
        assert.equal(answer.response, response);
        assert.equal(answer.condition, condition);
        assert.ok(Number.isFinite(answer.response_time_ms));
      }
    }
  });

  await test("every probe produces the full onset answer confidence resume chain", async () => {
    const h = probeHarness("B");
    for (let index = 0; index < 4; index++) {
      reachProbe(h);
      h.run("answerProbeAttention(2);submitProbeConfidence(3)");
      await settle();
    }
    const probeEvents = h.requests.map(r => r.payload)
      .filter(r => ["probe_onset", "attention_response", "confidence_response", "video_resume"].includes(r?.event_type));
    assert.equal(probeEvents.length, 16);
    for (let index = 0; index < 4; index++) {
      const chain = probeEvents.slice(index * 4, index * 4 + 4);
      assert.deepEqual(chain.map(r => r.event_type),
        ["probe_onset", "attention_response", "confidence_response", "video_resume"]);
      assert.ok(chain.every(r => r.probe_id.startsWith(`3_${index + 1}_`) && r.block_id === "3" &&
        r.subject_id === "sub-test" && r.session_id === "ses-001" &&
        r.condition === "B" && r.condition_type === "bbbd_subtraction" &&
        Number.isFinite(r.client_timestamp) && Number.isFinite(r.event_timestamp)));
    }
    assert.equal(h.run("probeIndex"), 4);
    h.run('$("video").currentTime=999;maybeTriggerProbe()');
    assert.equal(h.run("activeProbe===null"), true);
  });

  await test("probe schedule is reported to the recorder and cleared between blocks", async () => {
    const h = probeHarness();
    h.run("recordProbeSchedule()");
    await settle();
    const schedule = h.requests.map(r => r.payload).find(r => r?.event_type === "probe_schedule");
    assert.equal(schedule.probe_times.length, 4);
    assert.equal(schedule.video_duration_sec, 300);
    assert.deepEqual(schedule.probe_config, { probes_per_block: 4, min_first_onset_sec: 45, min_end_margin_sec: 30, min_gap_sec: 45 });
    h.run("clearProbeState()");
    assert.equal(h.run("probeSchedule===null"), true);
    assert.equal(h.run("probeIndex"), 0);
  });

  await test("high 50 Hz is shown as a warning that keeps the data usable", async () => {
    const h = harness();
    h.run('applyQc({status:"warning",status_label:"注意",needs_notch:true,warning_50hz:true,packet_loss_rate_pct:0,data_age_sec:0,messages:["通道1的50 Hz工频干扰超过50%，正在确认是否持续。"],channels:[{line_50hz_ratio_pct:88.2},{line_50hz_ratio_pct:12.0}]})');
    assert.equal(h.run('$("qcState").className'), "qcState warning");
    assert.ok(h.run('$("qcState").textContent').includes("50 Hz"));
    assert.ok(h.run('$("qcShared").textContent').includes("分析阶段再做notch"));
    assert.equal(h.run('$("qcLine0").textContent'), "88.20%");
  });

  await test("a clean window shows no notch hint", async () => {
    const h = harness();
    h.run('applyQc({status:"good",status_label:"正常",needs_notch:false,packet_loss_rate_pct:0,data_age_sec:0,messages:["当前QC指标正常。"],channels:[{line_50hz_ratio_pct:3.1},{line_50hz_ratio_pct:2.4}]})');
    assert.equal(h.run('$("qcState").className'), "qcState good");
    assert.equal(h.run('$("qcShared").textContent').includes("notch"), false);
  });

  console.log(`${passed} browser event tests passed.`);
})().catch(error => { console.error(error); process.exitCode = 1 });
