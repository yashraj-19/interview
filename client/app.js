// SViam browser client — full-duplex WebRTC to the Pipecat server.
// getUserMedia uses echoCancellation/noiseSuppression/autoGainControl so the
// candidate can interrupt the bot cleanly (browser AEC removes the bot's voice
// from the mic).

const STUN = [{ urls: "stun:stun.l.google.com:19302" }];

// Pull ICE servers (STUN + TURN relay) from the server, which reads TURN creds
// from env. Falls back to STUN-only so LOCAL dev (browser<->localhost) is unchanged.
// No iceTransportPolicy='relay' — default ICE tries direct first, TURN is fallback.
async function getIceServers() {
  try {
    const r = await fetch("/api/ice");
    if (r.ok) {
      const j = await r.json();
      if (j && Array.isArray(j.iceServers) && j.iceServers.length) return j.iceServers;
    }
  } catch (e) { /* fall through to STUN */ }
  return STUN;
}

const connectBtn = document.getElementById("connectBtn");
const endBtn = document.getElementById("endBtn");
const remoteAudio = document.getElementById("remoteAudio");
const transcriptEl = document.getElementById("transcript");
const dot = document.getElementById("dot");
const statusText = document.getElementById("statusText");

let pc = null;
let localStream = null;
let ws = null;
let pcId = null;
let started = false;

const STATUS_LABEL = { listening: "Listening", speaking: "Speaking", thinking: "Thinking" };

function setStatus(s) {
  dot.className = "dot " + (s in STATUS_LABEL ? s : "idle");
  statusText.textContent = STATUS_LABEL[s] || s;
}

function addTurn(role, text) {
  if (transcriptEl.querySelector(".empty")) transcriptEl.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "turn " + role;
  const who = document.createElement("div");
  who.className = "who";
  who.textContent = role === "bot" ? "SViam" : "You";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.appendChild(who);
  wrap.appendChild(bubble);
  transcriptEl.appendChild(wrap);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function openEventSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (e) => {
    let ev;
    try { ev = JSON.parse(e.data); } catch { return; }
    if (ev.type === "status") setStatus(ev.status);
    else if (ev.type === "transcript") addTurn(ev.role, ev.text);
  };
}

// Force Opus to send CONTINUOUS audio (no DTX). Without this, the browser sends
// nothing during silence, which starves the server's turn detector and makes
// turn-end fire tens of seconds late.
function disableOpusDtx(sdp) {
  const m = sdp.match(/a=rtpmap:(\d+) opus\/48000/i);
  if (!m) return sdp;
  const pt = m[1];
  const fmtp = new RegExp(`a=fmtp:${pt} ([^\\r\\n]*)`);
  if (fmtp.test(sdp)) {
    return sdp.replace(fmtp, (line, params) =>
      /usedtx=/.test(params)
        ? `a=fmtp:${pt} ${params.replace(/usedtx=\d/, "usedtx=0")}`
        : `a=fmtp:${pt} ${params};usedtx=0`
    );
  }
  return sdp.replace(
    new RegExp(`(a=rtpmap:${pt} opus/48000/2\\r?\\n)`),
    `$1a=fmtp:${pt} usedtx=0\r\n`
  );
}

// Wait for ICE gathering to finish (non-trickle: send one complete offer).
function waitForIce(pc) {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    const check = () => {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", check);
        resolve();
      }
    };
    pc.addEventListener("icegatheringstatechange", check);
    setTimeout(resolve, 2500); // fallback so we never hang
  });
}

async function connect() {
  connectBtn.disabled = true;
  try {
    localStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      video: false,
    });
  } catch (err) {
    setStatus("idle");
    statusText.textContent = "Microphone blocked";
    connectBtn.disabled = false;
    return;
  }

  const iceServers = await getIceServers();
  pc = new RTCPeerConnection({ iceServers });
  pc.ontrack = (e) => { remoteAudio.srcObject = e.streams[0]; };
  pc.onconnectionstatechange = () => {
    if (["failed", "disconnected", "closed"].includes(pc.connectionState) && started) end();
  };
  localStream.getTracks().forEach((t) => pc.addTrack(t, localStream));

  openEventSocket();

  // pipecat's SmallWebRTC connection negotiates a data channel. The client is the
  // offerer, so it must originate it — without this the peer connection stays
  // half-established ("data channel not established"), which destabilizes the audio.
  pc.createDataChannel("pipecat");

  const offer = await pc.createOffer();
  offer.sdp = disableOpusDtx(offer.sdp);
  await pc.setLocalDescription(offer);
  await waitForIce(pc);

  const res = await fetch("/api/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type, pc_id: pcId }),
  });
  const answer = await res.json();
  pcId = answer.pc_id;
  await pc.setRemoteDescription(answer);

  started = true;
  endBtn.disabled = false;
  setStatus("listening");
}

function end() {
  started = false;
  if (pc) { try { pc.close(); } catch {} pc = null; }
  if (localStream) { localStream.getTracks().forEach((t) => t.stop()); localStream = null; }
  if (ws) { try { ws.close(); } catch {} ws = null; }
  remoteAudio.srcObject = null;
  pcId = null;
  dot.className = "dot idle";
  statusText.textContent = "Not connected";
  connectBtn.disabled = false;
  endBtn.disabled = true;
}

connectBtn.addEventListener("click", connect);
endBtn.addEventListener("click", end);
