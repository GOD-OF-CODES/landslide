// GAME workstream: the player's inventory (DESIGN.md "game": `inventory: Set<'jerrycan'|'hatchet'|'planks'>`).
//
// It IS a Set (so `ctx.game.inventory.has('hatchet')` works for everyone), and every change is pushed to the HUD
// (`hud.setInventory(ids, details)`: ids = ['jerrycan', ...], details = [{id, label, short, icon}]) and to the bus as `inventory`.

export const ITEM_INFO = {
  jerrycan: { label: 'Jerrycan (5 L)', short: 'Fuel', icon: 'jerrycan' },
  hatchet: { label: 'Hatchet', short: 'Hatchet', icon: 'hatchet' },
  planks: { label: 'Scaffold planks', short: 'Planks', icon: 'planks' },
};

export default class Inventory extends Set {
  constructor(ctx, items = []) {
    super();
    this.ctx = ctx;
    this._silent = true;
    for (const i of items) super.add(i);
    this._silent = false;
  }

  add(id) {
    if (!id || super.has(id)) return this;
    super.add(id);
    this._changed();
    return this;
  }

  delete(id) {
    const r = super.delete(id);
    if (r) this._changed();
    return r;
  }

  clear() {
    if (!this.size) return;
    super.clear();
    this._changed();
  }

  /** Replace the contents (used by checkpoints). */
  set(ids) {
    super.clear();
    for (const i of ids || []) super.add(i);
    this._changed();
  }

  list() { return [...this]; }

  /** Rich list for the HUD (it may use either form). */
  describe() { return this.list().map((id) => ({ id, ...(ITEM_INFO[id] || { label: id, short: id, icon: id }) })); }

  _changed() {
    if (this._silent || !this.ctx) return;
    const ids = this.list();
    try { this.ctx.hud?.setInventory?.(ids, this.describe()); } catch (e) { console.warn('[inventory] hud.setInventory', e); }
    this.ctx.events?.emit?.('inventory', { items: ids });
  }
}
