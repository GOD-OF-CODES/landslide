// Inline SVG icons for the HUD (drawn by hand, no external assets). All use currentColor.
// 48x48 viewBox, 1.6 px strokes, so they stay crisp at 28-40 px.

const svg = (body, vb = '0 0 48 48') =>
  `<svg viewBox="${vb}" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;

export const ICONS = {
  // 20 L NATO jerrycan: three-bar handle, angled spout corner, embossed X
  jerrycan: svg(`
    <path d="M13 13.5h19.5l3.5 3.5v24.5a2 2 0 0 1-2 2H15a2 2 0 0 1-2-2z"/>
    <path d="M17 13.5V7.5h13v6"/>
    <path d="M21.3 7.5v6M25.7 7.5v6"/>
    <path d="M13 18.5l-4.2-4.2 3.2-3.2 4.2 4.2"/>
    <path d="M17 19.5h15v20H17z" stroke-opacity=".55"/>
    <path d="M17.6 20.2 31.4 38.8M31.4 20.2 17.6 38.8" stroke-opacity=".8"/>`),
  // Hatchet: curved hickory handle, forged head with bearded blade
  hatchet: svg(`
    <path d="M13.5 43c3.4-7.8 8.2-18 13.8-29" stroke-width="3.2"/>
    <path d="M13.5 43c3.4-7.8 8.2-18 13.8-29" stroke-width="1" stroke="#000" stroke-opacity=".35"/>
    <path d="M22.5 6.4 31.4 5c2.6-.4 5.1-1.7 8.2-3.4 2.4 5.8 2.9 12.6 1.1 19.4-2.6-2.3-5.8-4.3-8.7-5.1l-8.7-1.6z" fill="currentColor" fill-opacity=".14"/>
    <path d="M39.6 1.6c2.4 5.8 2.9 12.6 1.1 19.4" stroke-width="2.4"/>
    <path d="M26.4 6.1l1.6 8.4" stroke-opacity=".6"/>`),
  // Three scaffold boards, stacked, with steel band ends
  planks: svg(`
    <path d="M5 15.5 33 9.5l10 3.5-28 6z"/>
    <path d="M5 15.5v3.2l10 3.5 28-6v-3.2"/><path d="M15 19v3.2"/>
    <path d="M5 24.5 33 18.5l10 3.5-28 6z"/>
    <path d="M5 24.5v3.2l10 3.5 28-6v-3.2"/><path d="M15 28v3.2"/>
    <path d="M5 33.5 33 27.5l10 3.5-28 6z"/>
    <path d="M5 33.5v3.2l10 3.5 28-6v-3.2"/><path d="M15 37v3.2"/>
    <path d="M9 14.6l10 3.5M9 23.6l10 3.5M9 32.6l10 3.5" stroke-opacity=".5"/>
    <path d="M18 14.8l17-3.6M18 23.8l17-3.6M18 32.8l17-3.6" stroke-opacity=".28"/>`),
  // Fuel pump pictogram used by the low-fuel lamp (filled)
  pump: `<path d="M3 21V4.5A1.5 1.5 0 0 1 4.5 3h8A1.5 1.5 0 0 1 14 4.5V21zM5.2 5.2v5.6h6.6V5.2zM14 9h1.6a1.6 1.6 0 0 1 1.6 1.6v6.2a1 1 0 0 0 2 0V8.4l-2.6-2.6 1-1 3.2 3.2v8.8a2.6 2.6 0 0 1-5.2 0v-5.6H14zM1.8 21h13.4v1.6H1.8z" fill="currentColor"/>`,
};

export const ITEM_NAMES = { jerrycan: 'Jerrycan', hatchet: 'Hatchet', planks: 'Planks' };
