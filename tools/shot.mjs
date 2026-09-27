// Headless-Chrome screenshot + console capture for self-verification.
// Usage: node tools/shot.mjs <url-or-path> <out.png> [--wait=ms] [--w=1280] [--h=720] [--eval="js"]
// - Path like "/?debug&cam=..." is resolved against http://localhost:5173
// - Waits until window.__READY === true (or timeout), then optional extra wait, then screenshots.
// - Prints console errors/warnings and window.__STATS (if any) as JSON to stdout.
import { chromium } from 'playwright-core';
const args = process.argv.slice(2);
const pos = args.filter((a) => !a.startsWith('--'));
const opt = Object.fromEntries(args.filter((a) => a.startsWith('--')).map((a) => { const [k, ...v] = a.slice(2).split('='); return [k, v.join('=') || true]; }));
let url = pos[0] || '/';
if (url.startsWith('/')) url = 'http://localhost:5173' + url;
const out = pos[1] || 'shot.png';
const W = +(opt.w || 1280), H = +(opt.h || 720);
const browser = await chromium.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true,
  args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-webgl', '--autoplay-policy=no-user-gesture-required'],
});
const page = await browser.newPage({ viewport: { width: W, height: H } });
const logs = [];
page.on('console', (m) => { if (['error', 'warning'].includes(m.type()) || opt.verbose) logs.push(`[${m.type()}] ${m.text()}`); });
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}\n${e.stack || ''}`));
page.on('requestfailed', (r) => logs.push(`[requestfailed] ${r.url()} ${r.failure()?.errorText}`));
const t0 = Date.now();
await page.goto(url, { waitUntil: 'load', timeout: 120000 });
try { await page.waitForFunction(() => window.__READY === true, null, { timeout: +(opt.timeout || 120000) }); }
catch { logs.push('[shot] timed out waiting for window.__READY'); }
if (opt.eval) { try { await page.evaluate(opt.eval); } catch (e) { logs.push('[eval error] ' + e.message); } }
await page.waitForTimeout(+(opt.wait || 1500));
await page.screenshot({ path: out });
const stats = await page.evaluate(() => window.__STATS || null).catch(() => null);
const gl = await page.evaluate(() => { const c = document.createElement('canvas').getContext('webgl2'); const d = c && c.getExtension('WEBGL_debug_renderer_info'); return d ? c.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'n/a'; }).catch(() => 'n/a');
console.log(JSON.stringify({ out, ms: Date.now() - t0, gl, stats, logs }, null, 2));
await browser.close();
