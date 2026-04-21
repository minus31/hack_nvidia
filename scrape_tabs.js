const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const BASE_URL = 'https://nemotron-dev-materials-q9notf2ox.brevlab.com/';

const TABS = [
  { hash: 'common',      label: 'Common',              file: '01_common' },
  { hash: 'nim',         label: 'NIM & Nemotron',      file: '02_nim_nemotron' },
  { hash: 'nat',         label: 'NeMo Agent Toolkit',  file: '03_nemo_agent_toolkit' },
  { hash: 'nemoclaw',    label: 'NemoClaw',            file: '04_nemoclaw' },
  { hash: 'megatron',    label: 'Megatron-Bridge',     file: '05_megatron_bridge' },
  { hash: 'nemorl',      label: 'NeMo RL',             file: '06_nemo_rl' },
  { hash: 'datadesigner',label: 'NeMo Data Designer',  file: '07_nemo_data_designer' },
  { hash: 'curator',     label: 'NeMo Curator',        file: '08_nemo_curator' },
];

const OUT_DIR = path.join(__dirname, 'docs', 'material', 'raw');
fs.mkdirSync(OUT_DIR, { recursive: true });

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  // Load page once to understand structure
  await page.goto(BASE_URL, { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(2000);

  for (const tab of TABS) {
    console.log(`\nScraping: ${tab.label}`);

    // Navigate with hash
    await page.goto(`${BASE_URL}#tab=${tab.hash}`, { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(1000);

    // Try clicking matching tab button
    const allButtons = page.locator('button, a, [role="tab"]');
    const count = await allButtons.count();
    for (let i = 0; i < count; i++) {
      const btn = allButtons.nth(i);
      const text = (await btn.textContent()).trim();
      const cleanLabel = tab.label.replace(/[📦🤖🦞🔧🎯📊🧹]/g, '').trim();
      if (text.includes(cleanLabel) || text.toLowerCase().includes(cleanLabel.toLowerCase())) {
        try {
          await btn.click({ timeout: 3000 });
          await page.waitForTimeout(1500);
          break;
        } catch {}
      }
    }

    // Extract content: find active/visible section
    const content = await page.evaluate(() => {
      // Remove nav/header/sidebar clutter if possible
      const selectors = [
        '[role="tabpanel"]:not([hidden])',
        '.tab-content.active',
        '.tab-pane.show.active',
        'main',
        'article',
      ];
      for (const sel of selectors) {
        const el = document.querySelector(sel);
        if (el && el.innerText && el.innerText.length > 200) {
          return el.innerText;
        }
      }
      return document.body.innerText;
    });

    // Save raw text
    const rawPath = path.join(OUT_DIR, `${tab.file}.txt`);
    fs.writeFileSync(rawPath, content);
    console.log(`  Saved ${content.length} chars -> ${rawPath}`);
  }

  await browser.close();
  console.log('\nDone!');
})();
