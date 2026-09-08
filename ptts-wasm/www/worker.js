import init, { Model, cpu_features } from './ptts_wasm.js';

// The Phonon checkpoint is not published, so it is served from the same origin
// as the page: point a static server at the model repo (scratch/wasm-lan does
// this with --models). Swap in a URL once it is hosted.
// The Phonon checkpoint is not published, so it is served from the same origin
// as the page: point a static server at the model repo (scratch/wasm-lan does
// this with --models). Swap in a URL once it is hosted.
const MODEL_BASE = '/models';

const CHECKPOINT = {
  label: 'Phonon',
  quant: 'q8',
  model: `${MODEL_BASE}/model/model.q8.gguf`,
  tokenizer: `${MODEL_BASE}/model/tokenizer.model`,
  // This checkpoint's architecture differs from the one built into the wasm
  // module, so its config travels with it and is parsed at load.
  config: `${MODEL_BASE}/model/config.json`,
  // Precomputed conditioning (scratch/prep-voices), not raw embeddings. An
  // embedding has to be run through `prompt_audio` before it can be used --
  // a 125-frame prefill through the whole backbone, per voice. Doing eight of
  // those on a phone at page load burns the thermal budget before the user has
  // asked for any speech, so the browser loads a finished KV cache instead and
  // does no arithmetic at all to register a voice.
  voiceDir: `${MODEL_BASE}/voices-prepared`,
  voices: ['Archie', 'Damon', 'Elodie-Rose', 'Freddie', 'Freya', 'Garrett', 'Marlowe', 'Zoey'],
};

function post(type, data = {}, transferables = []) {
  self.postMessage({ type, ...data }, transferables);
}

// ---- Fetch with progress (posts to main thread) ----
async function fetchWithProgress(url, label) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);
  const total = parseInt(resp.headers.get('content-length') || '0', 10);
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total > 0) {
      const pct = Math.round(received / total * 100);
      post('progress', {
        label,
        pct,
        detail: `${(received / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`,
      });
    } else {
      post('progress', { label, pct: -1, detail: `${(received / 1e6).toFixed(1)} MB` });
    }
  }
  post('progress_done');
  const buf = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    buf.set(chunk, offset);
    offset += chunk.length;
  }
  return buf;
}

// ---- Minimal protobuf decoder for sentencepiece .model files ----
function decodeSentencepieceModel(buffer) {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  let pos = 0;

  function readVarint() {
    let result = 0, shift = 0;
    while (pos < buffer.length) {
      const b = buffer[pos++];
      result |= (b & 0x7f) << shift;
      shift += 7;
      if ((b & 0x80) === 0) return result;
    }
    return result;
  }

  function readBytes(n) {
    const data = buffer.slice(pos, pos + n);
    pos += n;
    return data;
  }

  function decodePiece(data) {
    let pPos = 0, piece = '', score = 0, type = 1;
    const pView = new DataView(data.buffer, data.byteOffset, data.byteLength);
    while (pPos < data.length) {
      const key = readVarIntFrom(data, pPos);
      pPos = key.pos;
      const fieldNum = key.val >>> 3;
      const wireType = key.val & 0x7;
      if (fieldNum === 1 && wireType === 2) {
        const len = readVarIntFrom(data, pPos);
        pPos = len.pos;
        piece = new TextDecoder().decode(data.slice(pPos, pPos + len.val));
        pPos += len.val;
      } else if (fieldNum === 2 && wireType === 5) {
        score = pView.getFloat32(pPos, true);
        pPos += 4;
      } else if (fieldNum === 3 && wireType === 0) {
        const v = readVarIntFrom(data, pPos);
        type = v.val;
        pPos = v.pos;
      } else {
        if (wireType === 0) { const v = readVarIntFrom(data, pPos); pPos = v.pos; }
        else if (wireType === 1) { pPos += 8; }
        else if (wireType === 2) { const len = readVarIntFrom(data, pPos); pPos = len.pos + len.val; }
        else if (wireType === 5) { pPos += 4; }
        else break;
      }
    }
    return { piece, score, type };
  }

  function readVarIntFrom(buf, p) {
    let result = 0, shift = 0;
    while (p < buf.length) {
      const b = buf[p++];
      result |= (b & 0x7f) << shift;
      shift += 7;
      if ((b & 0x80) === 0) return { val: result, pos: p };
    }
    return { val: result, pos: p };
  }

  const pieces = [];
  while (pos < buffer.length) {
    const key = readVarint();
    const fieldNum = key >>> 3;
    const wireType = key & 0x7;
    if (fieldNum === 1 && wireType === 2) {
      const len = readVarint();
      const data = readBytes(len);
      const p = decodePiece(data);
      pieces.push(p);
    } else {
      if (wireType === 0) { readVarint(); }
      else if (wireType === 1) { pos += 8; }
      else if (wireType === 2) { const len = readVarint(); pos += len; }
      else if (wireType === 5) { pos += 4; }
      else break;
    }
  }
  return pieces;
}

// ---- Unigram tokenizer (Viterbi) ----
class UnigramTokenizer {
  constructor(pieces) {
    this.pieces = pieces;
    this.vocab = new Map();
    this.unkId = 0;
    for (let i = 0; i < pieces.length; i++) {
      const p = pieces[i];
      if (p.type === 2) this.unkId = i;
      if (p.type === 1 || p.type === 4) {
        this.vocab.set(p.piece, { id: i, score: p.score });
      }
      if (p.type === 6) {
        this.vocab.set(p.piece, { id: i, score: p.score });
      }
    }
  }

  encode(text) {
    const normalized = '\u2581' + text.replace(/ /g, '\u2581');
    return this._viterbi(normalized);
  }

  _viterbi(text) {
    const n = text.length;
    const best = new Array(n + 1);
    best[0] = { score: 0, len: 0, id: -1 };
    for (let i = 1; i <= n; i++) {
      best[i] = { score: -Infinity, len: 0, id: -1 };
    }

    for (let i = 0; i < n; i++) {
      if (best[i].score === -Infinity) continue;
      for (let len = 1; len <= n - i && len <= 64; len++) {
        const sub = text.substring(i, i + len);
        const entry = this.vocab.get(sub);
        if (entry) {
          const newScore = best[i].score + entry.score;
          if (newScore > best[i + len].score) {
            best[i + len] = { score: newScore, len: len, id: entry.id };
          }
        }
      }
      if (best[i + 1].score === -Infinity) {
        const ch = text.charCodeAt(i);
        const byteStr = `<0x${ch.toString(16).toUpperCase().padStart(2, '0')}>`;
        const byteEntry = this.vocab.get(byteStr);
        const fallbackId = byteEntry ? byteEntry.id : this.unkId;
        const fallbackScore = byteEntry ? byteEntry.score : -100;
        best[i + 1] = { score: best[i].score + fallbackScore, len: 1, id: fallbackId };
      }
    }

    const ids = [];
    let p = n;
    while (p > 0) {
      ids.push(best[p].id);
      p -= best[p].len;
    }
    ids.reverse();
    return new Uint32Array(ids);
  }
}

// ---- Worker state ----
// Start WASM compilation immediately so the optimizing compiler (TurboFan)
// finishes well before the first generation runs.
const wasmModulePromise = WebAssembly.compileStreaming(fetch('ptts_wasm_bg.wasm'));

let model = null;
let tokenizer = null;
let voiceIndexMap = {};

async function handleLoad() {
  const ckpt = CHECKPOINT;
  const wasmModule = await wasmModulePromise;
  await init(wasmModule);
  post('status', { message: 'WASM initialized. Downloading tokenizer and model...' });

  const tokData = await fetchWithProgress(ckpt.tokenizer, 'Tokenizer');
  const pieces = decodeSentencepieceModel(tokData);
  tokenizer = new UnigramTokenizer(pieces);
  post('status', { message: `Tokenizer loaded (${pieces.length} pieces)` });

  // Fetched before the weights: a checkpoint that needs a config and cannot get
  // one would otherwise download hundreds of megabytes and then fail to load.
  let configJson = null;
  if (ckpt.config) {
    const resp = await fetch(ckpt.config);
    if (!resp.ok) throw new Error(`Failed to fetch ${ckpt.config}: ${resp.status}`);
    configJson = await resp.text();
  }

  const modelWeights = await fetchWithProgress(ckpt.model, 'Model weights');

  post('status', { message: `Initializing ${ckpt.label}...` });
  model = new Model(modelWeights, ckpt.quant, configJson);

  // Voices are fetched on first use, not here: one is ~9 MB and the user only
  // ever hears one at a time, so loading eight would cost bandwidth and delay
  // the first generation for nothing.
  voiceIndexMap = {};

  const sampleRate = model.sample_rate();
  const features = cpu_features();
  post('loaded', { sampleRate, features, voices: ckpt.voices });
}

async function ensureVoice(name) {
  if (voiceIndexMap[name] !== undefined) return voiceIndexMap[name];
  const url = `${CHECKPOINT.voiceDir}/${name}.safetensors`;
  const data = await fetchWithProgress(url, `Voice: ${name}`);
  // Registering a precomputed voice is a tensor copy, not a prefill.
  voiceIndexMap[name] = model.add_voice(data);
  return voiceIndexMap[name];
}

async function handleGenerate(text, voiceName, temperature) {
  if (!model) throw new Error('model is not loaded yet');
  if (!voiceName) throw new Error('no voice selected');
  const voiceIndex = await ensureVoice(voiceName);

  const [processedText, framesAfterEos] = model.prepare_text(text);
  const tokenIds = tokenizer.encode(processedText);

  post('gen_start', { numTokens: tokenIds.length });

  // Time the prompt step: `start_generation` runs `prompt_text` on the
  // transformer state, which is the bulk of the prefill cost.
  const promptT0 = performance.now();
  model.start_generation(voiceIndex, tokenIds, framesAfterEos, temperature);
  const promptMs = performance.now() - promptT0;

  let step = 0;
  let stepMsTotal = 0;
  let stepMsMin = Infinity;
  let stepMsMax = 0;
  while (true) {
    const t0 = performance.now();
    const chunk = model.generation_step();
    const dt = performance.now() - t0;
    if (!chunk) break;
    // A step that only sampled -- the vocoder decodes a batch of frames at a
    // time -- yields no audio yet, so there is nothing to hand over.
    if (chunk.length === 0) { stepMsTotal += dt; continue; }
    stepMsTotal += dt;
    if (dt < stepMsMin) stepMsMin = dt;
    if (dt > stepMsMax) stepMsMax = dt;
    post('chunk', { data: chunk, step }, [chunk.buffer]);
    step++;
  }

  const stepMsAvg = step > 0 ? stepMsTotal / step : 0;
  if (step === 0) stepMsMin = 0;
  post('done', {
    promptMs,
    numSteps: step,
    stepMsAvg,
    stepMsMin,
    stepMsMax,
  });
}

self.onmessage = async (e) => {
  const { type, ...data } = e.data;
  try {
    if (type === 'load') {
      await handleLoad();
    } else if (type === 'generate') {
      await handleGenerate(data.text, data.voiceName, data.temperature);
    }
  } catch (err) {
    post('error', { message: err.message });
    console.error(err);
  }
};
