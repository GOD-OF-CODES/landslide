// AUDIO workstream: module worker that renders procedural sound banks off the main thread.
// message in:  {id, gen, sr, seed, arg}     message out: {id, sr, channels: Float32Array[]} | {id, error}
import { GENERATORS } from './dsp.js';

self.onmessage = (e) => {
  const { id, gen, sr, seed, arg } = e.data;
  try {
    const fn = GENERATORS[gen];
    if (!fn) throw new Error('unknown generator ' + gen);
    const channels = fn(sr, seed, arg || {});
    self.postMessage({ id, sr, channels }, channels.map((c) => c.buffer));
  } catch (err) {
    self.postMessage({ id, error: String(err?.message || err) });
  }
};
