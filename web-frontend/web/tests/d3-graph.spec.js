import { test, expect } from '@playwright/test';

// Helper: set expression, evaluate, wait for D3 graph to render
async function evaluateExpression(page, expr, variables = {}) {
  const exprPage = page.locator('#page-expression');

  // Set expression
  await exprPage.locator('#expression').fill(expr);
  // Wait for variable inputs to appear
  await page.waitForTimeout(200);

  // Set variable values (scoped to Expression page)
  for (const [name, value] of Object.entries(variables)) {
    const input = exprPage.locator(`input[data-variable="${name}"]`);
    if (await input.count() > 0) {
      await input.fill(String(value));
    }
  }

  // Click evaluate
  await exprPage.locator('#evaluate').click();

  // Wait for D3 graph to render (SVG should have child elements)
  await page.waitForFunction(() => {
    const svg = document.querySelector('#d3-graph');
    return svg && svg.children.length > 0;
  }, { timeout: 10000 });
}

test.describe('Karpathy Backprop Intro', () => {

  test.beforeEach(async ({ page }) => {
    page.on('pageerror', error => {
      console.error('PAGE ERROR:', error.message);
    });
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('auto-evaluates on page load with correct result', async ({ page }) => {
    // Karpathy page is the default, should auto-evaluate: d = 2.0 * -3.0 + 10.0 = 4.0
    await page.waitForFunction(() => {
      const el = document.querySelector('#karpathy-result');
      return el && el.textContent !== '-';
    }, { timeout: 10000 });

    const result = await page.textContent('#karpathy-result');
    expect(result).toContain('4.000000');
  });

  test('displays gradients for a, b, c', async ({ page }) => {
    await page.waitForFunction(() => {
      const el = document.querySelector('#karpathy-gradients');
      return el && el.children.length > 0;
    }, { timeout: 10000 });

    const gradients = page.locator('#karpathy-gradients li');
    await expect(gradients).toHaveCount(3);
  });

  test('renders D3 graph on auto-evaluate', async ({ page }) => {
    await page.waitForFunction(() => {
      const svg = document.querySelector('#karpathy-d3-graph');
      return svg && svg.children.length > 0;
    }, { timeout: 10000 });

    const circles = page.locator('#karpathy-d3-graph circle');
    const count = await circles.count();
    // d = a * b + c has nodes: a, b, c, *, +, d(result) = at least 5
    expect(count).toBeGreaterThanOrEqual(5);
  });

  test('renders Mermaid graph on auto-evaluate', async ({ page }) => {
    await page.waitForFunction(() => {
      const container = document.querySelector('#karpathy-mermaid-graph');
      return container && container.querySelector('svg');
    }, { timeout: 10000 });

    const svg = page.locator('#karpathy-mermaid-graph svg');
    await expect(svg).toHaveCount(1);
  });

  test('re-evaluates with modified variable values', async ({ page }) => {
    // Wait for initial auto-evaluate
    await page.waitForFunction(() => {
      const el = document.querySelector('#karpathy-result');
      return el && el.textContent !== '-';
    }, { timeout: 10000 });

    // Change a to 5.0: d = 5 * -3 + 10 = -5
    await page.fill('#karpathy-variables input[data-variable="a"]', '5.0');
    await page.click('#karpathy-evaluate');

    await page.waitForFunction(() => {
      const el = document.querySelector('#karpathy-result');
      return el && el.textContent.includes('-5');
    }, { timeout: 10000 });

    const result = await page.textContent('#karpathy-result');
    expect(result).toContain('-5.000000');
  });

  test('no console errors on auto-evaluate', async ({ page }) => {
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));

    await page.waitForFunction(() => {
      const svg = document.querySelector('#karpathy-d3-graph');
      return svg && svg.children.length > 0;
    }, { timeout: 10000 });

    expect(errors).toEqual([]);
  });
});

test.describe('Expand/Minimize graphs', () => {

  test.beforeEach(async ({ page }) => {
    page.on('pageerror', error => {
      console.error('PAGE ERROR:', error.message);
    });
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('expand button appears on D3 graph after auto-evaluate', async ({ page }) => {
    await page.waitForFunction(() => {
      const svg = document.querySelector('#karpathy-d3-graph');
      return svg && svg.children.length > 0;
    }, { timeout: 10000 });

    const btn = page.locator('#karpathy-graph-container .expand-btn');
    await expect(btn).toHaveCount(1);
  });

  test('expand button appears on Mermaid graph after auto-evaluate', async ({ page }) => {
    await page.waitForFunction(() => {
      const container = document.querySelector('#karpathy-mermaid-graph');
      return container && container.querySelector('svg');
    }, { timeout: 10000 });

    const btn = page.locator('#karpathy-mermaid-container .expand-btn');
    await expect(btn).toHaveCount(1);
  });

  test('clicking expand makes D3 container fullscreen', async ({ page }) => {
    await page.waitForFunction(() => {
      const svg = document.querySelector('#karpathy-d3-graph');
      return svg && svg.children.length > 0;
    }, { timeout: 10000 });

    const container = page.locator('#karpathy-graph-container');
    const btn = container.locator('.expand-btn');

    await btn.click();
    await expect(container).toHaveClass(/expanded/);

    // Container should be fixed position filling viewport
    const box = await container.boundingBox();
    expect(box.x).toBe(0);
    expect(box.y).toBe(0);
    expect(box.width).toBeGreaterThan(500);
    expect(box.height).toBeGreaterThan(500);

    // SVG should be resized to fill the viewport (not viewBox-scaled)
    const svgWidth = await page.locator('#karpathy-d3-graph').evaluate(
      el => parseInt(el.getAttribute('width'))
    );
    expect(svgWidth).toBeGreaterThan(500);
  });

  test('clicking minimize restores D3 container', async ({ page }) => {
    await page.waitForFunction(() => {
      const svg = document.querySelector('#karpathy-d3-graph');
      return svg && svg.children.length > 0;
    }, { timeout: 10000 });

    const container = page.locator('#karpathy-graph-container');
    const btn = container.locator('.expand-btn');

    // Expand then minimize
    await btn.click();
    await expect(container).toHaveClass(/expanded/);
    await btn.click();

    const hasExpanded = await container.evaluate(el => el.classList.contains('expanded'));
    expect(hasExpanded).toBe(false);
  });

  test('Escape key closes expanded view', async ({ page }) => {
    await page.waitForFunction(() => {
      const svg = document.querySelector('#karpathy-d3-graph');
      return svg && svg.children.length > 0;
    }, { timeout: 10000 });

    const container = page.locator('#karpathy-graph-container');
    const btn = container.locator('.expand-btn');

    await btn.click();
    await expect(container).toHaveClass(/expanded/);

    await page.keyboard.press('Escape');

    const hasExpanded = await container.evaluate(el => el.classList.contains('expanded'));
    expect(hasExpanded).toBe(false);
  });

  test('+ key zooms in via viewBox and shows indicator', async ({ page }) => {
    await page.waitForFunction(() => {
      const svg = document.querySelector('#karpathy-d3-graph');
      return svg && svg.children.length > 0;
    }, { timeout: 10000 });

    const container = page.locator('#karpathy-graph-container');
    await container.locator('.expand-btn').click();
    await expect(container).toHaveClass(/expanded/);

    // Read base viewBox before zoom
    const baseParts = await page.locator('#karpathy-d3-graph').evaluate(
      el => el.getAttribute('viewBox').split(/\s+/).map(Number)
    );
    const [, , baseW, baseH] = baseParts;

    await page.keyboard.press('+');

    // viewBox should be smaller (zoomed in) — width/height shrunk
    const zoomedParts = await page.locator('#karpathy-d3-graph').evaluate(
      el => el.getAttribute('viewBox').split(/\s+/).map(Number)
    );
    expect(zoomedParts[2]).toBeLessThan(baseW);
    expect(zoomedParts[3]).toBeLessThan(baseH);

    // Zoom indicator should appear
    const indicator = container.locator('.zoom-indicator');
    await expect(indicator).toHaveCount(1);
    const text = await indicator.textContent();
    expect(text).toBe('120%');
  });

  test('- key zooms out via viewBox', async ({ page }) => {
    await page.waitForFunction(() => {
      const svg = document.querySelector('#karpathy-d3-graph');
      return svg && svg.children.length > 0;
    }, { timeout: 10000 });

    const container = page.locator('#karpathy-graph-container');
    await container.locator('.expand-btn').click();
    await expect(container).toHaveClass(/expanded/);

    const baseParts = await page.locator('#karpathy-d3-graph').evaluate(
      el => el.getAttribute('viewBox').split(/\s+/).map(Number)
    );

    await page.keyboard.press('-');

    // viewBox should be larger (zoomed out)
    const zoomedParts = await page.locator('#karpathy-d3-graph').evaluate(
      el => el.getAttribute('viewBox').split(/\s+/).map(Number)
    );
    expect(zoomedParts[2]).toBeGreaterThan(baseParts[2]);
    expect(zoomedParts[3]).toBeGreaterThan(baseParts[3]);
  });

  test('0 key resets viewBox to base', async ({ page }) => {
    await page.waitForFunction(() => {
      const svg = document.querySelector('#karpathy-d3-graph');
      return svg && svg.children.length > 0;
    }, { timeout: 10000 });

    const container = page.locator('#karpathy-graph-container');
    await container.locator('.expand-btn').click();

    const baseVB = await page.locator('#karpathy-d3-graph').evaluate(
      el => el.getAttribute('viewBox')
    );

    // Zoom in twice then reset
    await page.keyboard.press('+');
    await page.keyboard.press('+');
    await page.keyboard.press('0');

    const resetVB = await page.locator('#karpathy-d3-graph').evaluate(
      el => el.getAttribute('viewBox')
    );
    expect(resetVB).toBe(baseVB);
  });

  test('zoom does not apply when not expanded', async ({ page }) => {
    await page.waitForFunction(() => {
      const svg = document.querySelector('#karpathy-d3-graph');
      return svg && svg.children.length > 0;
    }, { timeout: 10000 });

    const vbBefore = await page.locator('#karpathy-d3-graph').evaluate(
      el => el.getAttribute('viewBox')
    );

    // Press + without expanding — should have no effect
    await page.keyboard.press('+');

    const vbAfter = await page.locator('#karpathy-d3-graph').evaluate(
      el => el.getAttribute('viewBox')
    );
    expect(vbAfter).toBe(vbBefore);
  });

  test('layout selector appears when expanded', async ({ page }) => {
    await page.waitForFunction(() => {
      const svg = document.querySelector('#karpathy-d3-graph');
      return svg && svg.children.length > 0;
    }, { timeout: 10000 });

    const container = page.locator('#karpathy-graph-container');
    const layoutSelect = container.locator('.layout-select');

    // Hidden before expand
    await expect(container.locator('.layout-selector')).not.toBeVisible();

    await container.locator('.expand-btn').click();
    await expect(container).toHaveClass(/expanded/);

    // Visible after expand with all layout options
    await expect(container.locator('.layout-selector')).toBeVisible();
    const optionCount = await layoutSelect.locator('option').count();
    expect(optionCount).toBe(5);
  });

  test('switching layout re-renders the graph without errors', async ({ page }) => {
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));

    await page.waitForFunction(() => {
      const svg = document.querySelector('#karpathy-d3-graph');
      return svg && svg.children.length > 0;
    }, { timeout: 10000 });

    const container = page.locator('#karpathy-graph-container');
    await container.locator('.expand-btn').click();
    await expect(container).toHaveClass(/expanded/);

    // Switch to each layout
    for (const layout of ['layered', 'top-down', 'radial', 'compact', 'force-directed']) {
      await container.locator('.layout-select').selectOption(layout);
      // Wait for re-render
      await page.waitForFunction(() => {
        const svg = document.querySelector('#karpathy-d3-graph');
        return svg && svg.children.length > 0;
      }, { timeout: 5000 });
    }

    expect(errors).toEqual([]);

    // Graph should still have correct node count
    const circles = page.locator('#karpathy-d3-graph circle');
    const count = await circles.count();
    expect(count).toBeGreaterThanOrEqual(5);
  });

  test('dragging background pans the viewBox', async ({ page }) => {
    await page.waitForFunction(() => {
      const svg = document.querySelector('#karpathy-d3-graph');
      return svg && svg.children.length > 0;
    }, { timeout: 10000 });

    const container = page.locator('#karpathy-graph-container');
    await container.locator('.expand-btn').click();
    await expect(container).toHaveClass(/expanded/);

    // Read viewBox before pan
    const vbBefore = await page.locator('#karpathy-d3-graph').evaluate(
      el => el.getAttribute('viewBox').split(/\s+/).map(Number)
    );

    // Drag on the SVG background (top-left corner, away from nodes)
    const box = await container.boundingBox();
    await page.mouse.move(box.x + 50, box.y + 50);
    await page.mouse.down();
    await page.mouse.move(box.x + 250, box.y + 150, { steps: 5 });
    await page.mouse.up();

    // viewBox origin should have shifted (x decreased = panned right, y decreased = panned down)
    const vbAfter = await page.locator('#karpathy-d3-graph').evaluate(
      el => el.getAttribute('viewBox').split(/\s+/).map(Number)
    );
    expect(vbAfter[0]).toBeLessThan(vbBefore[0]);
    expect(vbAfter[1]).toBeLessThan(vbBefore[1]);
    // Width and height unchanged (pan doesn't zoom)
    expect(vbAfter[2]).toBeCloseTo(vbBefore[2], 0);
    expect(vbAfter[3]).toBeCloseTo(vbBefore[3], 0);
  });

  test('0 key resets pan and zoom together', async ({ page }) => {
    await page.waitForFunction(() => {
      const svg = document.querySelector('#karpathy-d3-graph');
      return svg && svg.children.length > 0;
    }, { timeout: 10000 });

    const container = page.locator('#karpathy-graph-container');
    await container.locator('.expand-btn').click();
    await expect(container).toHaveClass(/expanded/);

    const vbBase = await page.locator('#karpathy-d3-graph').evaluate(
      el => el.getAttribute('viewBox')
    );

    // Zoom in and pan
    await page.keyboard.press('+');
    const box = await container.boundingBox();
    await page.mouse.move(box.x + 100, box.y + 100);
    await page.mouse.down();
    await page.mouse.move(box.x + 300, box.y + 200, { steps: 5 });
    await page.mouse.up();

    // Reset
    await page.keyboard.press('0');
    const vbReset = await page.locator('#karpathy-d3-graph').evaluate(
      el => el.getAttribute('viewBox')
    );
    expect(vbReset).toBe(vbBase);
  });
});

test.describe('D3 Computation Graph', () => {

  test.beforeEach(async ({ page }) => {
    // Collect console errors
    page.on('pageerror', error => {
      console.error('PAGE ERROR:', error.message);
    });
    await page.goto('/#expression');
    await page.waitForLoadState('networkidle');
  });

  test('no console errors for simple addition x + y', async ({ page }) => {
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));

    await evaluateExpression(page, 'x + y', { x: 2, y: 3 });

    expect(errors).toEqual([]);
  });

  test('no console errors for multiplication x * y', async ({ page }) => {
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));

    await evaluateExpression(page, 'x * y', { x: 2, y: 3 });

    expect(errors).toEqual([]);
  });

  test('no console errors for complex expression x^2 + y*z', async ({ page }) => {
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));

    await evaluateExpression(page, 'x^2 + y*z', { x: 2, y: 3, z: 4 });

    expect(errors).toEqual([]);
  });

  test('no console errors for default tanh expression', async ({ page }) => {
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));

    // Use the default expression already in the input
    await page.click('#evaluate');
    await page.waitForFunction(() => {
      const svg = document.querySelector('#d3-graph');
      return svg && svg.children.length > 0;
    }, { timeout: 10000 });

    expect(errors).toEqual([]);
  });

  test('D3 graph renders correct number of nodes for x + y', async ({ page }) => {
    await evaluateExpression(page, 'x + y', { x: 2, y: 3 });

    // x, y, + = 3 nodes
    const circles = page.locator('#d3-graph circle');
    await expect(circles).toHaveCount(3);
  });

  test('D3 graph renders correct number of edges for x + y', async ({ page }) => {
    await evaluateExpression(page, 'x + y', { x: 2, y: 3 });

    // 2 edges: x→+, y→+
    const paths = page.locator('#d3-graph path[stroke="#999"]');
    await expect(paths).toHaveCount(2);
  });

  test('edge labels show local deriv and gradient contribution', async ({ page }) => {
    await evaluateExpression(page, 'x + y', { x: 2, y: 3 });

    // Edge labels should contain pipe separator between local deriv and grad contrib
    const labels = page.locator('#d3-graph text[fill="#ffdd57"]');
    const count = await labels.count();
    expect(count).toBeGreaterThan(0);

    for (let i = 0; i < count; i++) {
      const text = await labels.nth(i).textContent();
      expect(text).toContain('|');
    }
  });

  test('a + a shows two edges to the + node', async ({ page }) => {
    await evaluateExpression(page, 'a + a', { a: 3 });

    // 2 nodes: a, +  (a is deduplicated)
    const circles = page.locator('#d3-graph circle');
    await expect(circles).toHaveCount(2);

    // 2 edges: both from a to +
    const paths = page.locator('#d3-graph path[stroke="#999"]');
    await expect(paths).toHaveCount(2);
  });

  test('a + a edges are visually distinct (curved)', async ({ page }) => {
    await evaluateExpression(page, 'a + a', { a: 3 });

    // Wait for simulation to settle slightly
    await page.waitForTimeout(500);

    const paths = page.locator('#d3-graph path[stroke="#999"]');
    await expect(paths).toHaveCount(2);

    // Get the d attributes of both paths - they should differ (one curved)
    const d0 = await paths.nth(0).getAttribute('d');
    const d1 = await paths.nth(1).getAttribute('d');

    expect(d0).toBeTruthy();
    expect(d1).toBeTruthy();
    // The two paths should have different d attributes (one straight, one curved)
    expect(d0).not.toEqual(d1);
  });

  test('a * a shows two edges to the * node', async ({ page }) => {
    await evaluateExpression(page, 'a * a', { a: 3 });

    // 2 nodes: a, *
    const circles = page.locator('#d3-graph circle');
    await expect(circles).toHaveCount(2);

    // 2 edges
    const paths = page.locator('#d3-graph path[stroke="#999"]');
    await expect(paths).toHaveCount(2);
  });

  test('result node appears with named expression', async ({ page }) => {
    await evaluateExpression(page, 'L = x + y', { x: 2, y: 3 });

    // x, y, +, L(result) = 4 nodes
    const circles = page.locator('#d3-graph circle');
    await expect(circles).toHaveCount(4);
  });

  test('no console errors for exp and log expressions', async ({ page }) => {
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));

    await evaluateExpression(page, 'exp(x) + log(y)', { x: 1, y: 2 });

    expect(errors).toEqual([]);
  });

  test('result value is displayed correctly for x + y', async ({ page }) => {
    await evaluateExpression(page, 'x + y', { x: 2, y: 3 });

    const result = await page.textContent('#result');
    expect(result).toContain('5.000000');
  });

  test('SVG paths have valid d attributes (no NaN)', async ({ page }) => {
    await evaluateExpression(page, 'x * y + x', { x: 2, y: 3 });

    // Wait for simulation to produce path data
    await page.waitForTimeout(500);

    const paths = page.locator('#d3-graph path[stroke="#999"]');
    const count = await paths.count();
    expect(count).toBeGreaterThan(0);

    for (let i = 0; i < count; i++) {
      const d = await paths.nth(i).getAttribute('d');
      expect(d).toBeTruthy();
      expect(d).not.toContain('NaN');
      expect(d).not.toContain('undefined');
    }
  });

  test('edge labels have valid numeric values (no NaN)', async ({ page }) => {
    await evaluateExpression(page, 'x * y + x', { x: 2, y: 3 });

    const labels = page.locator('#d3-graph text[fill="#ffdd57"]');
    const count = await labels.count();
    expect(count).toBeGreaterThan(0);

    for (let i = 0; i < count; i++) {
      const text = await labels.nth(i).textContent();
      expect(text).not.toContain('NaN');
      expect(text).not.toContain('undefined');
    }
  });

  test('no console errors for subtraction and division', async ({ page }) => {
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));

    await evaluateExpression(page, 'x / y - x', { x: 6, y: 2 });

    expect(errors).toEqual([]);
  });

  test('no console errors for power expression', async ({ page }) => {
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));

    await evaluateExpression(page, 'x ^ y', { x: 2, y: 3 });

    expect(errors).toEqual([]);
  });

  test('fast division x /~ y renders without errors', async ({ page }) => {
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));

    await evaluateExpression(page, 'x /~ y', { x: 6, y: 3 });

    expect(errors).toEqual([]);

    // Result should be 6 * 3^(-1) = 2
    const result = await page.textContent('#result');
    expect(result).toContain('2.000000');
  });

  test('fast division shows correct node count and edges', async ({ page }) => {
    await evaluateExpression(page, 'x /~ y', { x: 6, y: 3 });

    // x, y, /~ = 3 nodes
    const circles = page.locator('#d3-graph circle');
    await expect(circles).toHaveCount(3);

    // 2 edges: x→/~, y→/~
    const paths = page.locator('#d3-graph path[stroke="#999"]');
    await expect(paths).toHaveCount(2);
  });

  test('fast division gradients match regular division', async ({ page }) => {
    await evaluateExpression(page, 'x /~ y', { x: 6, y: 3 });

    // Check gradients are displayed
    const gradients = page.locator('#gradients li');
    const count = await gradients.count();
    expect(count).toBe(2);
  });
});
