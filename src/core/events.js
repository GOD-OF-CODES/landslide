// Tiny synchronous event bus. See DESIGN.md "Events" for the list of event names and payloads.
export class Events {
  constructor() { this.map = new Map(); }
  on(name, fn) {
    if (!this.map.has(name)) this.map.set(name, new Set());
    this.map.get(name).add(fn);
    return () => this.off(name, fn);
  }
  once(name, fn) {
    const off = this.on(name, (p) => { off(); fn(p); });
    return off;
  }
  off(name, fn) { this.map.get(name)?.delete(fn); }
  emit(name, payload) {
    const set = this.map.get(name);
    if (!set) return;
    for (const fn of [...set]) {
      try { fn(payload); } catch (e) { console.error(`[events] handler for "${name}" threw`, e); }
    }
  }
}
