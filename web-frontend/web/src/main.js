import * as d3 from "d3";
import mermaid from "mermaid";
import katex from "katex";
import "katex/dist/katex.min.css";

// Initialize mermaid with KaTeX math support
mermaid.initialize({
  startOnLoad: false,
  theme: "dark",
  securityLevel: "loose",
  htmlLabels: true,
});

// Register KaTeX with mermaid for $$...$$ math rendering
window.katex = katex;

// Import Scala.js module
let PaladiumFrontend = null;

async function loadScalaJS() {
  try {
    // Try to load from Mill's output directory
    const module = await import("scalajs/main.js");
    PaladiumFrontend = module.PaladiumFrontend || window.PaladiumFrontend;
    console.log("Scala.js module loaded successfully");
  } catch (e) {
    console.warn(
      "Could not load Scala.js module directly, falling back to API:",
      e,
    );
  }
}

// ============================================
// Page routing
// ============================================

function hashToPage(hash) {
  switch (hash) {
    case "#expression": return "page-expression";
    case "#mlp": return "page-mlp";
    case "#neural-network": return "page-nn";
    case "#block-dsl": return "page-block";
    default: return "page-karpathy";
  }
}

function initRouter() {
  const navLinks = document.querySelectorAll(".nav-link");
  const pages = document.querySelectorAll(".page");

  function showPage(pageId) {
    pages.forEach((p) => p.classList.remove("active"));
    navLinks.forEach((a) => a.classList.remove("active"));

    const page = document.getElementById(pageId);
    const link = document.querySelector(`[data-page="${pageId}"]`);
    if (page) page.classList.add("active");
    if (link) link.classList.add("active");
  }

  navLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const pageId = link.dataset.page;
      showPage(pageId);
      history.pushState(null, "", link.getAttribute("href"));
    });
  });

  // Handle browser back/forward
  window.addEventListener("popstate", () => {
    const hash = location.hash || "#expression";
    const pageId = hashToPage(hash);
    showPage(pageId);
  });

  // Initial route from URL hash
  const hash = location.hash;
  const initialPage = hashToPage(hash);
  if (initialPage !== "page-karpathy") {
    showPage(initialPage);
  }
}

// ============================================
// Expression page (existing functionality)
// ============================================

// Split "d = expr" into { name, body } or { name: null, body: expr }
function splitAssignment(expr) {
  const eqIdx = expr.indexOf("=");
  if (eqIdx > 0) {
    const lhs = expr.substring(0, eqIdx).trim();
    if (/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(lhs)) {
      return { name: lhs, body: expr.substring(eqIdx + 1).trim() };
    }
  }
  return { name: null, body: expr };
}

// Extract variable names from expression (excludes result name on LHS)
function extractVariables(expr) {
  const { body } = splitAssignment(expr);
  const varPattern = /[a-zA-Z_][a-zA-Z0-9_]*/g;
  const reserved = new Set(["log", "sin", "cos", "tan", "exp", "tanh", "sqrt", "sigmoid", "relu", "abs"]);
  const matches = body.match(varPattern) || [];
  return [...new Set(matches.filter((v) => !reserved.has(v)))];
}

// Update variable inputs based on expression
function updateVariableInputs() {
  const expr = document.getElementById("expression").value;
  const variables = extractVariables(expr);
  const container = document.getElementById("variables");

  // Get existing values
  const existingValues = {};
  container.querySelectorAll(".variable-input").forEach((div) => {
    const name = div.querySelector("label").textContent.replace(":", "");
    const value = div.querySelector("input").value;
    existingValues[name] = value;
  });

  // Get existing nudge values
  const existingNudges = {};
  container.querySelectorAll(".nudge-input").forEach((input) => {
    existingNudges[input.dataset.nudgeFor] = input.value;
  });

  // Rebuild inputs
  container.innerHTML = "";
  variables.forEach((varName) => {
    const div = document.createElement("div");
    div.className = "variable-input";

    const varRow = document.createElement("div");
    varRow.className = "var-row";
    const label = document.createElement("label");
    label.textContent = varName + ":";
    const input = document.createElement("input");
    input.type = "text";
    const defaults = {
      x1: "2.0",
      x2: "0.0",
      w1: "-3.0",
      w2: "1.0",
      b: "6.8813735870195432",
    };
    input.value = existingValues[varName] || defaults[varName] || "1";
    input.dataset.variable = varName;
    varRow.appendChild(label);
    varRow.appendChild(input);

    const nudgeRow = document.createElement("div");
    nudgeRow.className = "nudge-row";
    const nudgeLabel = document.createElement("label");
    nudgeLabel.textContent = "h:";
    const nudgeInput = document.createElement("input");
    nudgeInput.type = "text";
    nudgeInput.className = "nudge-input";
    nudgeInput.value = existingNudges[varName] ?? "0";
    nudgeInput.dataset.nudgeFor = varName;
    nudgeRow.appendChild(nudgeLabel);
    nudgeRow.appendChild(nudgeInput);

    div.appendChild(varRow);
    div.appendChild(nudgeRow);
    container.appendChild(div);
  });
}

// Get variable values from inputs
function getVariableValues() {
  const values = {};
  document
    .querySelectorAll(".variable-input input[data-variable]")
    .forEach((input) => {
      values[input.dataset.variable] = parseFloat(input.value) || 0;
    });
  return values;
}

// Get nudge (h) values from inputs
function getNudgeValues() {
  const nudges = {};
  document.querySelectorAll(".nudge-input").forEach((input) => {
    const val = parseFloat(input.value);
    nudges[input.dataset.nudgeFor] = isNaN(val) ? 0 : val;
  });
  return nudges;
}

// Evaluate using Scala.js directly
async function evaluateScalaJS(expr, variables) {
  if (!PaladiumFrontend) {
    throw new Error("Scala.js not loaded");
  }

  const jsVars = {};
  for (const [k, v] of Object.entries(variables)) {
    jsVars[k] = v;
  }

  const result = PaladiumFrontend.evaluateExpression(expr, jsVars);
  const gradients = PaladiumFrontend.getGradients(expr, jsVars);
  const symbolicGradients = PaladiumFrontend.getSymbolicGradients(expr);
  const mermaidGraph = PaladiumFrontend.toMermaidGraph(expr, jsVars);
  const d3Graph = JSON.parse(PaladiumFrontend.toD3Graph(expr, jsVars));

  return { result, gradients, symbolicGradients, mermaidGraph, d3Graph };
}

// Evaluate using API fallback (now uses /api/trace for graph data)
async function evaluateAPI(expr, variables) {
  const [evalRes, gradRes, symbolicRes, traceRes] = await Promise.all([
    fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ expression: expr, variables }),
    }).then((r) => r.json()),

    fetch("/api/gradient", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ expression: expr, variables }),
    }).then((r) => r.json()),

    fetch("/api/symbolic-gradient", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ expression: expr }),
    }).then((r) => r.json()),

    fetch("/api/trace", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ expression: expr, variables }),
    }).then((r) => r.json()),
  ]);

  return {
    result: evalRes.result,
    gradients: { value: gradRes.value, gradients: gradRes.gradients },
    symbolicGradients: symbolicRes.gradients,
    mermaidGraph: traceRes.mermaid,
    d3Graph: { nodes: traceRes.nodes, links: traceRes.links },
  };
}

// Render Mermaid graph
let mermaidCounter = 0;
async function renderMermaid(graphDef) {
  if (!graphDef) return;

  const container = document.getElementById("mermaid-graph");
  try {
    mermaidCounter++;
    const { svg } = await mermaid.render(
      `mermaid-svg-${mermaidCounter}`,
      graphDef,
    );
    container.innerHTML = svg;
  } catch (e) {
    container.textContent = "Error rendering graph: " + e.message;
  }
}

// Format number for display
function fmt(n) {
  if (Number.isInteger(n)) return n.toString();
  return Math.abs(n) < 0.001 && n !== 0 ? n.toExponential(2) : n.toFixed(4);
}

// High-precision format for finite difference display
function fmtPrecise(n) {
  if (Number.isInteger(n)) return n.toString();
  return Math.abs(n) < 1e-10 && n !== 0
    ? n.toExponential(6)
    : n.toPrecision(10);
}

// ============================================
// D3 graph layout presets
// ============================================

// Compute topological depth for each node (distance from leaves)
function computeNodeDepths(nodes, links) {
  const idToNode = {};
  nodes.forEach((n) => (idToNode[n.id] = n));
  const children = {};
  const parents = {};
  nodes.forEach((n) => {
    children[n.id] = [];
    parents[n.id] = [];
  });
  links.forEach((l) => {
    const src = typeof l.source === "object" ? l.source.id : l.source;
    const tgt = typeof l.target === "object" ? l.target.id : l.target;
    children[src].push(tgt);
    parents[tgt].push(src);
  });

  // BFS from roots (nodes with no parents) to assign depth
  const depth = {};
  const roots = nodes.filter((n) => parents[n.id].length === 0);
  const queue = roots.map((n) => ({ id: n.id, d: 0 }));
  roots.forEach((n) => (depth[n.id] = 0));
  while (queue.length > 0) {
    const { id, d } = queue.shift();
    for (const child of children[id]) {
      const nd = d + 1;
      if (depth[child] === undefined || nd > depth[child]) {
        depth[child] = nd;
        queue.push({ id: child, d: nd });
      }
    }
  }
  // Fallback for any disconnected nodes
  nodes.forEach((n) => {
    if (depth[n.id] === undefined) depth[n.id] = 0;
  });
  return depth;
}

// Pre-compute layered X positions: spread nodes at each depth evenly
function computeLayerPositions(nodes, depth, width, padding) {
  const layers = {};
  nodes.forEach((n) => {
    const d = depth[n.id];
    if (!layers[d]) layers[d] = [];
    layers[d].push(n.id);
  });
  const xPos = {};
  for (const [, ids] of Object.entries(layers)) {
    const count = ids.length;
    const spacing = (width - 2 * padding) / (count + 1);
    ids.forEach((id, i) => {
      xPos[id] = padding + spacing * (i + 1);
    });
  }
  return xPos;
}

const LAYOUT_PRESETS = {
  "force-directed": {
    label: "Force-Directed",
    create: (nodes, links, width, height) =>
      d3
        .forceSimulation(nodes)
        .force(
          "link",
          d3
            .forceLink(links)
            .id((d) => d.id)
            .distance(150),
        )
        .force("charge", d3.forceManyBody().strength(-500))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collide", d3.forceCollide(40)),
  },
  layered: {
    label: "Layered",
    create: (nodes, links, width, height) => {
      const depth = computeNodeDepths(nodes, links);
      const maxDepth = Math.max(...Object.values(depth), 1);
      const padY = 80;
      const padX = 80;
      const xPos = computeLayerPositions(nodes, depth, width, padX);
      return d3
        .forceSimulation(nodes)
        .force(
          "link",
          d3
            .forceLink(links)
            .id((d) => d.id)
            .distance(100)
            .strength(0.2),
        )
        .force("charge", d3.forceManyBody().strength(-400))
        .force(
          "x",
          d3.forceX((d) => xPos[d.id]).strength(0.7),
        )
        .force(
          "y",
          d3
            .forceY(
              (d) =>
                padY +
                (depth[d.id] / maxDepth) * (height - 2 * padY),
            )
            .strength(0.9),
        )
        .force("collide", d3.forceCollide(45));
    },
  },
  "top-down": {
    label: "Top-Down",
    create: (nodes, links, width, height) => {
      const depth = computeNodeDepths(nodes, links);
      const maxDepth = Math.max(...Object.values(depth), 1);
      const padding = 80;
      return d3
        .forceSimulation(nodes)
        .force(
          "link",
          d3
            .forceLink(links)
            .id((d) => d.id)
            .distance(100)
            .strength(0.3),
        )
        .force("charge", d3.forceManyBody().strength(-500))
        .force("x", d3.forceX(width / 2).strength(0.05))
        .force(
          "y",
          d3
            .forceY(
              (d) =>
                padding +
                (depth[d.id] / maxDepth) * (height - 2 * padding),
            )
            .strength(0.8),
        )
        .force("collide", d3.forceCollide(45));
    },
  },
  radial: {
    label: "Radial",
    create: (nodes, links, width, height) => {
      const depth = computeNodeDepths(nodes, links);
      const maxDepth = Math.max(...Object.values(depth), 1);
      const maxRadius = Math.min(width, height) / 2 - 80;
      return d3
        .forceSimulation(nodes)
        .force(
          "link",
          d3
            .forceLink(links)
            .id((d) => d.id)
            .distance(80)
            .strength(0.2),
        )
        .force("charge", d3.forceManyBody().strength(-300))
        .force(
          "radial",
          d3
            .forceRadial(
              (d) => (depth[d.id] / maxDepth) * maxRadius,
              width / 2,
              height / 2,
            )
            .strength(0.8),
        )
        .force("collide", d3.forceCollide(45));
    },
  },
  compact: {
    label: "Compact",
    create: (nodes, links, width, height) =>
      d3
        .forceSimulation(nodes)
        .force(
          "link",
          d3
            .forceLink(links)
            .id((d) => d.id)
            .distance(55)
            .strength(1),
        )
        .force("charge", d3.forceManyBody().strength(-120))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collide", d3.forceCollide(30)),
  },
};

let currentLayout = "force-directed";

// Render D3 force-directed graph with values and gradients
function renderD3Graph(
  graphData,
  svgSelector = "#d3-graph",
  containerSelector = "#graph-container",
  layout = currentLayout,
) {
  if (!graphData) return;

  const container = document.querySelector(containerSelector);
  const width = container.clientWidth;
  const height = 500;

  // Store graph data for re-rendering with different layout
  container._d3GraphData = graphData;
  container._d3SvgSelector = svgSelector;

  // Clear previous graph
  d3.select(svgSelector).selectAll("*").remove();

  const svg = d3
    .select(svgSelector)
    .attr("width", width)
    .attr("height", height);

  // Create a copy of nodes and links for D3
  const nodes = graphData.nodes.map((n) => ({ ...n }));
  const links = graphData.links.map((l) => ({ ...l }));

  // Detect multi-edges (same source->target) and assign curve offsets
  const linkGroups = {};
  links.forEach((l) => {
    const key = `${l.source}-${l.target}`;
    if (!linkGroups[key]) linkGroups[key] = 0;
    l.linkNum = linkGroups[key];
    linkGroups[key]++;
  });
  links.forEach((l) => {
    const key = `${l.source}-${l.target}`;
    l.totalLinks = linkGroups[key];
  });

  // Color scale for node types
  const color = d3
    .scaleOrdinal()
    .domain([
      "variable",
      "literal",
      "constant",
      "operation",
      "unary",
      "function",
      "result",
    ])
    .range([
      "#e94560",
      "#7b1fa2",
      "#c62828",
      "#ff9a3c",
      "#ff6b6b",
      "#4ecdc4",
      "#4caf50",
    ]);

  // Gradient magnitude scale for border glow
  const maxGrad = Math.max(
    ...nodes.map((n) => Math.abs(n.gradient || 0)),
    0.001,
  );
  const gradScale = d3.scaleLinear().domain([0, maxGrad]).range([0, 1]);

  // Create force simulation from layout preset
  const preset = LAYOUT_PRESETS[layout] || LAYOUT_PRESETS["force-directed"];
  const simulation = preset.create(nodes, links, width, height);

  // Add arrow marker
  svg
    .append("defs")
    .append("marker")
    .attr("id", "arrow")
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 32)
    .attr("refY", 0)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
    .append("path")
    .attr("fill", "#999")
    .attr("d", "M0,-5L10,0L0,5");

  // Create links as paths (supports curved multi-edges)
  const link = svg
    .append("g")
    .selectAll("path")
    .data(links)
    .join("path")
    .attr("stroke", "#999")
    .attr("stroke-opacity", 0.6)
    .attr("stroke-width", 2)
    .attr("fill", "none")
    .attr("marker-end", "url(#arrow)");

  // Add local derivative + gradient contribution labels on edges
  const linkLabel = svg
    .append("g")
    .selectAll("text")
    .data(links)
    .join("text")
    .text((d) =>
      d.localDeriv !== undefined
        ? `${fmt(d.localDeriv)} | ${fmt(d.gradContrib)}`
        : "",
    )
    .attr("text-anchor", "middle")
    .attr("fill", "#ffdd57")
    .attr("font-size", "10px")
    .attr("font-weight", "bold")
    .attr("dy", "-5");

  // Create node groups
  const node = svg
    .append("g")
    .selectAll("g")
    .data(nodes)
    .join("g")
    .call(
      d3
        .drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended),
    );

  // Add circles to nodes
  node
    .append("circle")
    .attr("r", 28)
    .attr("fill", (d) => color(d.type))
    .attr("stroke", (d) => {
      const intensity = gradScale(Math.abs(d.gradient || 0));
      return d3.interpolateRgb("#555", "#ffdd57")(intensity);
    })
    .attr("stroke-width", (d) => {
      const intensity = gradScale(Math.abs(d.gradient || 0));
      return 2 + intensity * 3;
    });

  // Add label (operator or variable name)
  node
    .append("text")
    .text((d) => d.label)
    .attr("text-anchor", "middle")
    .attr("dy", "-0.4em")
    .attr("fill", "#000")
    .attr("font-size", "13px")
    .attr("font-weight", "bold");

  // Add value below label
  node
    .append("text")
    .text((d) => (d.value !== undefined ? fmt(d.value) : ""))
    .attr("text-anchor", "middle")
    .attr("dy", "1em")
    .attr("fill", "#000")
    .attr("font-size", "10px");

  // Add gradient tooltip on hover
  node
    .append("title")
    .text((d) => `${d.label}\nvalue: ${d.value}\ngradient: ${d.gradient}`);

  // Compute curve control point for multi-edges
  function linkPath(d) {
    if (d.totalLinks <= 1) {
      return `M${d.source.x},${d.source.y}L${d.target.x},${d.target.y}`;
    }
    const dx = d.target.x - d.source.x;
    const dy = d.target.y - d.source.y;
    const mx = (d.source.x + d.target.x) / 2;
    const my = (d.source.y + d.target.y) / 2;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    const px = -dy / len;
    const py = dx / len;
    const offset = (d.linkNum - (d.totalLinks - 1) / 2) * 40;
    const cx = mx + px * offset;
    const cy = my + py * offset;
    return `M${d.source.x},${d.source.y}Q${cx},${cy} ${d.target.x},${d.target.y}`;
  }

  // Compute label position (midpoint of line or Bezier curve)
  function linkMid(d) {
    if (d.totalLinks <= 1) {
      return {
        x: (d.source.x + d.target.x) / 2,
        y: (d.source.y + d.target.y) / 2,
      };
    }
    const dx = d.target.x - d.source.x;
    const dy = d.target.y - d.source.y;
    const mx = (d.source.x + d.target.x) / 2;
    const my = (d.source.y + d.target.y) / 2;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    const px = -dy / len;
    const py = dx / len;
    const offset = (d.linkNum - (d.totalLinks - 1) / 2) * 40;
    const cx = mx + px * offset;
    const cy = my + py * offset;
    return {
      x: (d.source.x + 2 * cx + d.target.x) / 4,
      y: (d.source.y + 2 * cy + d.target.y) / 4,
    };
  }

  // Update positions on each tick
  simulation.on("tick", () => {
    link.attr("d", linkPath);

    linkLabel.attr("x", (d) => linkMid(d).x).attr("y", (d) => linkMid(d).y);

    node.attr("transform", (d) => `translate(${d.x},${d.y})`);
  });

  // Drag functions
  function dragstarted(event) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }

  function dragged(event) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }

  function dragended(event) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
  }

  // Store simulation on container for expand/collapse resizing
  container._d3Simulation = simulation;
  container._d3OrigWidth = width;
  container._d3OrigHeight = height;
}

// Compute finite difference for a single variable: (f(x+h) - f(x)) / h
async function evalAt(expr, variables) {
  if (PaladiumFrontend) {
    return PaladiumFrontend.evaluateExpression(expr, variables);
  }
  const res = await fetch("/api/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ expression: expr, variables }),
  }).then((r) => r.json());
  return res.result;
}

async function computeFiniteDifferences(expr, variables, nudges) {
  const f0 = await evalAt(expr, variables);
  const diffs = {};
  for (const varName of Object.keys(nudges)) {
    const h = nudges[varName];
    const nudged = { ...variables, [varName]: (variables[varName] || 0) + h };
    const f1 = await evalAt(expr, nudged);
    diffs[varName] = { approx: (f1 - f0) / h, f0, f1, h };
  }
  return diffs;
}

// Main evaluation function
async function evaluate() {
  const expr = document.getElementById("expression").value;
  const variables = getVariableValues();

  try {
    let data;
    if (PaladiumFrontend) {
      data = await evaluateScalaJS(expr, variables);
    } else {
      data = await evaluateAPI(expr, variables);
    }

    // Display result
    document.getElementById("result").textContent = data.result.toFixed(6);

    // Display numerical gradients
    const gradientsList = document.getElementById("gradients");
    gradientsList.innerHTML = "";
    const grads = data.gradients.gradients || data.gradients;
    for (const [varName, value] of Object.entries(grads)) {
      const li = document.createElement("li");
      li.textContent = `d/d${varName} = ${value.toFixed(6)}`;
      gradientsList.appendChild(li);
    }

    // Display symbolic gradients
    const symbolicList = document.getElementById("symbolic-gradients");
    symbolicList.innerHTML = "";
    for (const [varName, expr] of Object.entries(data.symbolicGradients)) {
      const li = document.createElement("li");
      li.textContent = `d/d${varName} = ${expr}`;
      symbolicList.appendChild(li);
    }

    // Compute and display finite differences (only for variables with h != 0)
    const nudges = getNudgeValues();
    const activeNudges = Object.fromEntries(
      Object.entries(nudges).filter(([, h]) => h !== 0),
    );
    const finiteDiffSection = document.getElementById("finite-diff-section");
    const finiteDiffList = document.getElementById("finite-diff");
    finiteDiffList.innerHTML = "";
    if (Object.keys(activeNudges).length > 0) {
      finiteDiffSection.style.display = "";

      // Show the seed gradient dL/dL = 1.0
      const { name: resultName } = splitAssignment(expr);
      const seedLabel = resultName || "f";
      const seedLi = document.createElement("li");
      seedLi.textContent = `\u2202${seedLabel}/\u2202${seedLabel} = 1.0  \u2190 backprop seed`;
      seedLi.style.opacity = "0.6";
      seedLi.style.fontStyle = "italic";
      finiteDiffList.appendChild(seedLi);

      const finiteDiffs = await computeFiniteDifferences(
        expr,
        variables,
        activeNudges,
      );
      for (const [varName, info] of Object.entries(finiteDiffs)) {
        const autodiffGrad = grads[varName];
        const li = document.createElement("li");
        let content = `\u2202f/\u2202${varName} \u2248 (${fmtPrecise(info.f1)} \u2212 ${fmtPrecise(info.f0)}) / ${info.h} = ${fmtPrecise(info.approx)}`;
        if (autodiffGrad !== undefined) {
          content += `  (autodiff: ${fmtPrecise(autodiffGrad)})`;
        }
        li.textContent = content;
        finiteDiffList.appendChild(li);
      }
    } else {
      finiteDiffSection.style.display = "none";
    }

    // Render graphs
    if (data.mermaidGraph) {
      await renderMermaid(data.mermaidGraph);
    }
    if (data.d3Graph) {
      renderD3Graph(data.d3Graph);
    }
  } catch (e) {
    console.error("Evaluation error:", e);
    document.getElementById("result").textContent = "Error: " + e.message;
  }
}

// ============================================
// Karpathy Backprop Intro page
// ============================================

let karpathyMermaidCounter = 0;

async function renderKarpathyMermaid(graphDef) {
  if (!graphDef) return;
  const container = document.getElementById("karpathy-mermaid-graph");
  try {
    karpathyMermaidCounter++;
    const { svg } = await mermaid.render(
      `karpathy-mermaid-svg-${karpathyMermaidCounter}`,
      graphDef,
    );
    container.innerHTML = svg;
  } catch (e) {
    container.textContent = "Error rendering graph: " + e.message;
  }
}

async function karpathyEvaluate() {
  const expr = document.getElementById("karpathy-expression").value;
  const variables = {};
  document
    .querySelectorAll("#karpathy-variables input[data-variable]")
    .forEach((input) => {
      variables[input.dataset.variable] = parseFloat(input.value) || 0;
    });

  try {
    let data;
    if (PaladiumFrontend) {
      data = await evaluateScalaJS(expr, variables);
    } else {
      data = await evaluateAPI(expr, variables);
    }

    // Display result
    document.getElementById("karpathy-result").textContent =
      data.result.toFixed(6);

    // Display numerical gradients
    const gradientsList = document.getElementById("karpathy-gradients");
    gradientsList.innerHTML = "";
    const grads = data.gradients.gradients || data.gradients;
    for (const [varName, value] of Object.entries(grads)) {
      const li = document.createElement("li");
      li.textContent = `\u2202d/\u2202${varName} = ${value.toFixed(6)}`;
      gradientsList.appendChild(li);
    }

    // Display symbolic gradients
    const symbolicList = document.getElementById("karpathy-symbolic");
    symbolicList.innerHTML = "";
    for (const [varName, gradExpr] of Object.entries(
      data.symbolicGradients,
    )) {
      const li = document.createElement("li");
      li.textContent = `\u2202d/\u2202${varName} = ${gradExpr}`;
      symbolicList.appendChild(li);
    }

    // Render graphs
    if (data.mermaidGraph) {
      await renderKarpathyMermaid(data.mermaidGraph);
    }
    if (data.d3Graph) {
      renderD3Graph(
        data.d3Graph,
        "#karpathy-d3-graph",
        "#karpathy-graph-container",
      );
    }
  } catch (e) {
    console.error("Karpathy evaluation error:", e);
    document.getElementById("karpathy-result").textContent =
      "Error: " + e.message;
  }
}

// ============================================
// Neural Network page
// ============================================

// Current MLP state
let nnState = {
  topology: [],
  params: {},
  initialized: false,
};

function parseTopology() {
  const text = document.getElementById("nn-topology").value;
  return text
    .split(",")
    .map((s) => parseInt(s.trim(), 10))
    .filter((n) => !isNaN(n) && n > 0);
}

function initNN() {
  if (!PaladiumFrontend) {
    alert("Scala.js module not loaded. Please wait and try again.");
    return;
  }

  const topology = parseTopology();
  if (topology.length < 2) {
    alert("Topology needs at least 2 layers (input + output).");
    return;
  }

  const seed = parseInt(document.getElementById("nn-seed").value, 10) || 42;
  const resultJson = PaladiumFrontend.initMLP(topology, seed);
  const result = JSON.parse(resultJson);

  nnState.topology = topology;
  nnState.params = result.params;
  nnState.initialized = true;

  document.getElementById("nn-param-count").textContent =
    `(${result.paramCount} parameters)`;
  renderParams();
  renderTrainingData();
  document.getElementById("nn-params-section").style.display = "";
}

function renderParams() {
  const container = document.getElementById("nn-params");
  container.innerHTML = "";

  const layers = nnState.topology;
  const layerPairs = [];
  for (let i = 0; i < layers.length - 1; i++) {
    layerPairs.push({ fanIn: layers[i], fanOut: layers[i + 1], idx: i });
  }

  layerPairs.forEach(({ fanIn, fanOut, idx }) => {
    const card = document.createElement("div");
    card.className = "layer-card";

    const h4 = document.createElement("h4");
    h4.textContent = `Layer ${idx}: ${fanIn} \u2192 ${fanOut}`;
    card.appendChild(h4);

    // Weights
    for (let i = 0; i < fanIn; i++) {
      for (let j = 0; j < fanOut; j++) {
        const name = `w${idx}_${i}_${j}`;
        card.appendChild(makeParamRow(name));
      }
    }

    // Biases
    for (let j = 0; j < fanOut; j++) {
      const name = `b${idx}_${j}`;
      card.appendChild(makeParamRow(name));
    }

    container.appendChild(card);
  });
}

function makeParamRow(name) {
  const row = document.createElement("div");
  row.className = "param-row";

  const label = document.createElement("label");
  label.textContent = name;

  const input = document.createElement("input");
  input.type = "text";
  input.value = nnState.params[name].toFixed(6);
  input.dataset.param = name;
  input.addEventListener("change", () => {
    const val = parseFloat(input.value);
    if (!isNaN(val)) {
      nnState.params[name] = val;
    }
  });

  row.appendChild(label);
  row.appendChild(input);
  return row;
}

function updateParamInputs() {
  document.querySelectorAll("#nn-params input[data-param]").forEach((input) => {
    const name = input.dataset.param;
    if (nnState.params[name] !== undefined) {
      input.value = nnState.params[name].toFixed(6);
    }
  });
}

function renderTrainingData() {
  const container = document.getElementById("nn-training-data");
  container.innerHTML = "";

  const inputSize = nnState.topology[0];
  const outputSize = nnState.topology[nnState.topology.length - 1];

  // Add a default sample if empty
  if (container.children.length === 0) {
    addTrainingSample(
      Array(inputSize)
        .fill(0)
        .map((_, i) => (i === 0 ? "1.0" : "0.0"))
        .join(", "),
      Array(outputSize).fill("1.0").join(", "),
    );
  }
}

function addTrainingSample(inputDefault = "", targetDefault = "") {
  const container = document.getElementById("nn-training-data");
  const row = document.createElement("div");
  row.className = "training-data-row";

  const inputLabel = document.createElement("label");
  inputLabel.textContent = "Input:";
  const inputField = document.createElement("input");
  inputField.type = "text";
  inputField.className = "sample-input";
  inputField.value = inputDefault;
  inputField.placeholder = "e.g., 1.0, 0.5";

  const sep = document.createElement("span");
  sep.textContent = "\u2192";
  sep.style.opacity = "0.5";

  const targetLabel = document.createElement("label");
  targetLabel.textContent = "Target:";
  const targetField = document.createElement("input");
  targetField.type = "text";
  targetField.className = "sample-target";
  targetField.value = targetDefault;
  targetField.placeholder = "e.g., 1.0";

  const removeBtn = document.createElement("button");
  removeBtn.textContent = "\u00d7";
  removeBtn.style.cssText =
    "padding: 0.3rem 0.6rem; font-size: 0.85rem; background: #555;";
  removeBtn.addEventListener("click", () => row.remove());

  row.appendChild(inputLabel);
  row.appendChild(inputField);
  row.appendChild(sep);
  row.appendChild(targetLabel);
  row.appendChild(targetField);
  row.appendChild(removeBtn);
  container.appendChild(row);
}

function getTrainingData() {
  const samples = [];
  document.querySelectorAll(".training-data-row").forEach((row) => {
    const inputs = row
      .querySelector(".sample-input")
      .value.split(",")
      .map((s) => parseFloat(s.trim()))
      .filter((n) => !isNaN(n));
    const targets = row
      .querySelector(".sample-target")
      .value.split(",")
      .map((s) => parseFloat(s.trim()))
      .filter((n) => !isNaN(n));
    if (inputs.length > 0 && targets.length > 0) {
      samples.push({ inputs, targets });
    }
  });
  return samples;
}

function nnForward() {
  if (!PaladiumFrontend || !nnState.initialized) return;

  const samples = getTrainingData();
  if (samples.length === 0) return;

  const sample = samples[0];
  const resultJson = PaladiumFrontend.mlpForward(
    nnState.topology,
    nnState.params,
    sample.inputs,
  );
  const outputs = JSON.parse(resultJson);

  document.getElementById("nn-stats").style.display = "";
  document.getElementById("nn-output").textContent = outputs
    .map((v) => v.toFixed(4))
    .join(", ");

  // Compute loss if targets available
  const gradResult = JSON.parse(
    PaladiumFrontend.mlpGradients(
      nnState.topology,
      nnState.params,
      sample.inputs,
      sample.targets,
    ),
  );
  document.getElementById("nn-loss").textContent = gradResult.loss.toFixed(6);
  document.getElementById("nn-epoch").textContent = "-";

  // Render graph
  try {
    const graphJson = PaladiumFrontend.mlpToD3Graph(
      nnState.topology,
      nnState.params,
      sample.inputs,
    );
    const graphData = JSON.parse(graphJson);
    renderD3Graph(graphData, "#nn-d3-graph", "#nn-graph-container");
  } catch (e) {
    console.error("Graph rendering error:", e);
  }
}

function nnTrain() {
  if (!PaladiumFrontend || !nnState.initialized) return;

  const samples = getTrainingData();
  if (samples.length === 0) {
    alert("Add at least one training sample.");
    return;
  }

  const lr = parseFloat(document.getElementById("nn-lr").value) || 0.05;
  const epochs =
    parseInt(document.getElementById("nn-epochs").value, 10) || 100;

  const logContainer = document.getElementById("nn-log");
  logContainer.style.display = "";
  logContainer.innerHTML = "";

  document.getElementById("nn-stats").style.display = "";
  document.getElementById("nn-train").disabled = true;

  let epoch = 0;

  function trainStep() {
    if (epoch >= epochs) {
      document.getElementById("nn-train").disabled = false;
      updateParamInputs();

      // Render final graph
      try {
        const graphJson = PaladiumFrontend.mlpToD3Graph(
          nnState.topology,
          nnState.params,
          samples[0].inputs,
        );
        const graphData = JSON.parse(graphJson);
        renderD3Graph(graphData, "#nn-d3-graph", "#nn-graph-container");
      } catch (e) {
        console.error("Graph rendering error:", e);
      }
      return;
    }

    let totalLoss = 0;

    // Train on each sample
    for (const sample of samples) {
      const resultJson = PaladiumFrontend.mlpTrainStep(
        nnState.topology,
        nnState.params,
        sample.inputs,
        sample.targets,
        lr,
      );
      const result = JSON.parse(resultJson);
      nnState.params = result.params;
      totalLoss += result.loss;
    }

    const avgLoss = totalLoss / samples.length;

    // Log every 10 epochs or first/last
    if (epoch % 10 === 0 || epoch === epochs - 1) {
      const entry = document.createElement("div");
      entry.className = "log-entry";
      entry.textContent = `Epoch ${epoch}: loss = ${avgLoss.toFixed(6)}`;
      logContainer.appendChild(entry);
      logContainer.scrollTop = logContainer.scrollHeight;
    }

    document.getElementById("nn-loss").textContent = avgLoss.toFixed(6);
    document.getElementById("nn-epoch").textContent = `${epoch + 1}/${epochs}`;

    // Forward pass on first sample for output display
    const fwdJson = PaladiumFrontend.mlpForward(
      nnState.topology,
      nnState.params,
      samples[0].inputs,
    );
    const outputs = JSON.parse(fwdJson);
    document.getElementById("nn-output").textContent = outputs
      .map((v) => v.toFixed(4))
      .join(", ");

    epoch++;
    // Use requestAnimationFrame for smooth UI updates
    requestAnimationFrame(trainStep);
  }

  requestAnimationFrame(trainStep);
}

// ============================================
// Render network Mermaid graph
let networkMermaidCounter = 0;
async function renderNetworkMermaid(graphDef) {
  if (!graphDef) return;

  const container = document.getElementById("network-mermaid-graph");
  try {
    networkMermaidCounter++;
    const { svg } = await mermaid.render(
      `network-mermaid-svg-${networkMermaidCounter}`,
      graphDef,
    );
    container.innerHTML = svg;
  } catch (e) {
    container.textContent = "Error rendering graph: " + e.message;
  }
}

// Populate network selector dropdown
function populateNetworkSelector() {
  if (!PaladiumFrontend) return;

  const select = document.getElementById("network-select");
  try {
    const networks = PaladiumFrontend.listNetworks();
    select.innerHTML = "";
    for (let i = 0; i < networks.length; i++) {
      const option = document.createElement("option");
      option.value = networks[i];
      option.textContent = networks[i];
      select.appendChild(option);
    }
  } catch (e) {
    console.warn("Could not list networks:", e);
  }
}

// Display source code in the preview panel
function renderSourceCode(code) {
  const el = document.getElementById("network-source-code");
  if (el) {
    el.textContent = code;
  }
}

// Visualize selected network
async function visualizeNetwork() {
  if (!PaladiumFrontend) {
    document.getElementById("network-mermaid-graph").textContent =
      "Scala.js not loaded";
    return;
  }

  const name = document.getElementById("network-select").value;
  const view = document.getElementById("network-view-toggle").value;

  try {
    let graphDef;
    if (view === "arch") {
      graphDef = PaladiumFrontend.getNetworkArchMermaid(name);
      document.getElementById("network-graph-title").textContent =
        "Architecture Diagram";
    } else {
      graphDef = PaladiumFrontend.getNetworkOpsMermaid(name);
      document.getElementById("network-graph-title").textContent =
        "Operations Graph";
    }
    await renderNetworkMermaid(graphDef);

    // Show source code for preset
    try {
      const source = PaladiumFrontend.getNetworkSourceCode(name);
      renderSourceCode(source);
    } catch (e) {
      console.warn("Could not get source code:", e);
    }
  } catch (e) {
    console.error("Network visualization error:", e);
    document.getElementById("network-mermaid-graph").textContent =
      "Error: " + e.message;
  }
}

// Build and visualize from topology string
async function buildFromTopology() {
  if (!PaladiumFrontend) {
    document.getElementById("network-mermaid-graph").textContent =
      "Scala.js not loaded";
    return;
  }

  const topology = document.getElementById("network-topology").value;
  const view = document.getElementById("network-view-toggle").value;

  try {
    const graphDef = PaladiumFrontend.buildFromTopology(topology, view);
    document.getElementById("network-graph-title").textContent =
      view === "arch" ? "Architecture Diagram" : "Operations Graph";
    await renderNetworkMermaid(graphDef);

    // Show generated source code
    try {
      const source = PaladiumFrontend.getTopologySourceCode(topology);
      renderSourceCode(source);
    } catch (e) {
      console.warn("Could not get topology source code:", e);
    }
  } catch (e) {
    console.error("Topology build error:", e);
    document.getElementById("network-mermaid-graph").textContent =
      "Error: " + e.message;
  }
}

// ============================================
// Block Composition DSL page
// ============================================

let blockMermaidCounter = 0;

async function renderBlockMermaid(graphDef) {
  if (!graphDef) return;
  const container = document.getElementById("block-mermaid-graph");
  try {
    blockMermaidCounter++;
    const { svg } = await mermaid.render(
      `block-mermaid-svg-${blockMermaidCounter}`,
      graphDef,
    );
    container.innerHTML = svg;
  } catch (e) {
    container.textContent = "Error rendering graph: " + e.message;
  }
}

async function buildBlockNetwork() {
  if (!PaladiumFrontend) {
    document.getElementById("block-mermaid-graph").textContent =
      "Scala.js not loaded";
    return;
  }

  const spec = document.getElementById("block-spec").value;
  const view = document.getElementById("block-view-toggle").value;

  try {
    const graphDef = PaladiumFrontend.buildBlockNetwork(spec, view);
    document.getElementById("block-graph-title").textContent =
      view === "arch" ? "Architecture Diagram" : "Operations Graph";
    await renderBlockMermaid(graphDef);

    try {
      const source = PaladiumFrontend.getBlockSourceCode(spec);
      document.getElementById("block-source-code").textContent = source;
    } catch (e) {
      console.warn("Could not get block source code:", e);
    }
  } catch (e) {
    console.error("Block build error:", e);
    document.getElementById("block-mermaid-graph").textContent =
      "Error: " + e.message;
  }
}

// ============================================
// Expandable graph containers
// ============================================

const EXPAND_ICON =
  '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>';
const MINIMIZE_ICON =
  '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M4 14h6v6m0-6L3 21M20 10h-6V4m0 6l7-7"/></svg>';

const ZOOM_STEP = 0.2;
const ZOOM_MIN = 0.2;
const ZOOM_MAX = 5.0;
let expandedZoom = 1.0;
let zoomIndicatorTimeout = null;

function setupExpandableGraphs() {
  const d3Selectors = [
    "#karpathy-graph-container",
    "#graph-container",
    "#nn-graph-container",
  ];
  const mermaidSelectors = [
    "#karpathy-mermaid-container",
    "#mermaid-container",
    "#network-mermaid-container",
    "#block-mermaid-container",
  ];

  [...d3Selectors, ...mermaidSelectors].forEach((selector) => {
    const container = document.querySelector(selector);
    if (!container) return;

    container.classList.add("graph-expandable");
    const isD3 = d3Selectors.includes(selector);

    // Add layout selector for D3 containers
    if (isD3) {
      const layoutDiv = document.createElement("div");
      layoutDiv.className = "layout-selector";
      const layoutLabel = document.createElement("label");
      layoutLabel.textContent = "Layout:";
      const layoutSelect = document.createElement("select");
      layoutSelect.className = "layout-select";
      for (const [key, preset] of Object.entries(LAYOUT_PRESETS)) {
        const opt = document.createElement("option");
        opt.value = key;
        opt.textContent = preset.label;
        if (key === currentLayout) opt.selected = true;
        layoutSelect.appendChild(opt);
      }
      layoutSelect.addEventListener("change", () => {
        const layout = layoutSelect.value;
        currentLayout = layout;
        if (container._d3GraphData && container._d3SvgSelector) {
          renderD3Graph(
            container._d3GraphData,
            container._d3SvgSelector,
            selector,
            layout,
          );
          // Restore expanded sizing after re-render
          const svg = container.querySelector("svg");
          if (svg && container.classList.contains("expanded")) {
            const newW = window.innerWidth - 32;
            const newH = window.innerHeight - 32;
            svg.setAttribute("width", newW);
            svg.setAttribute("height", newH);
            container._baseViewBox = { x: 0, y: 0, w: newW, h: newH };
            svg.setAttribute("viewBox", `0 0 ${newW} ${newH}`);
            expandedZoom = 1.0;
          }
        }
      });
      layoutDiv.appendChild(layoutLabel);
      layoutDiv.appendChild(layoutSelect);
      container.appendChild(layoutDiv);
    }

    const btn = document.createElement("button");
    btn.className = "expand-btn";
    btn.innerHTML = EXPAND_ICON;
    btn.title = "Expand";

    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const expanding = !container.classList.contains("expanded");
      container.classList.toggle("expanded");
      btn.innerHTML = expanding ? MINIMIZE_ICON : EXPAND_ICON;
      btn.title = expanding ? "Minimize" : "Expand";

      // Reset zoom and pan on every expand/collapse
      expandedZoom = 1.0;
      container._panOffset = { x: 0, y: 0 };

      if (isD3) {
        if (expanding && container._d3GraphData && container._d3SvgSelector) {
          // Re-render at full viewport size with current layout
          renderD3Graph(
            container._d3GraphData,
            container._d3SvgSelector,
            selector,
            currentLayout,
          );
          const svg = container.querySelector("svg");
          if (svg) {
            const newW = window.innerWidth - 32;
            const newH = window.innerHeight - 32;
            svg.setAttribute("width", newW);
            svg.setAttribute("height", newH);
            container._baseViewBox = { x: 0, y: 0, w: newW, h: newH };
            svg.setAttribute("viewBox", `0 0 ${newW} ${newH}`);
          }
        } else if (!expanding) {
          // Re-render at original size
          if (container._d3GraphData && container._d3SvgSelector) {
            renderD3Graph(
              container._d3GraphData,
              container._d3SvgSelector,
              selector,
              currentLayout,
            );
          }
          const svg = container.querySelector("svg");
          if (svg) {
            svg.removeAttribute("viewBox");
          }
          delete container._baseViewBox;
        }
      } else {
        // Mermaid: store/restore base viewBox for zoom
        const mermaidSvg = container.querySelector("pre.mermaid svg");
        if (mermaidSvg) {
          if (expanding) {
            const vb = mermaidSvg.getAttribute("viewBox");
            if (vb) {
              const parts = vb.split(/[\s,]+/).map(Number);
              container._baseViewBox = {
                x: parts[0],
                y: parts[1],
                w: parts[2],
                h: parts[3],
              };
              container._origViewBox = vb;
            }
          } else {
            if (container._origViewBox) {
              mermaidSvg.setAttribute("viewBox", container._origViewBox);
            }
            delete container._baseViewBox;
            delete container._origViewBox;
          }
        }
      }

      // Remove zoom indicator on collapse
      if (!expanding) {
        const indicator = container.querySelector(".zoom-indicator");
        if (indicator) indicator.remove();
      }
    });

    container.appendChild(btn);
  });

  // Keyboard controls for expanded view
  document.addEventListener("keydown", (e) => {
    const expanded = document.querySelector(".graph-expandable.expanded");
    if (!expanded) return;

    if (e.key === "Escape") {
      expanded.querySelector(".expand-btn").click();
      return;
    }

    // Zoom with + / - (also handle = for unshifted +, and numpad)
    if (e.key === "+" || e.key === "=" || e.key === "-") {
      e.preventDefault();
      if (e.key === "-") {
        expandedZoom = Math.max(ZOOM_MIN, expandedZoom - ZOOM_STEP);
      } else {
        expandedZoom = Math.min(ZOOM_MAX, expandedZoom + ZOOM_STEP);
      }
      applyExpandedZoom(expanded);
    }

    // Reset zoom and pan with 0
    if (e.key === "0") {
      e.preventDefault();
      expandedZoom = 1.0;
      expanded._panOffset = { x: 0, y: 0 };
      updateExpandedViewBox(expanded);
      showZoomIndicator(expanded);
    }
  });

  // Pan: mousedown on expanded container background starts panning
  let panState = null;

  document.addEventListener("mousedown", (e) => {
    const expanded = document.querySelector(".graph-expandable.expanded");
    if (!expanded) return;
    if (!expanded.contains(e.target)) return;

    // Don't pan when clicking on a D3 node (g group with a circle child)
    if (
      e.target.closest &&
      e.target.closest("g")?.querySelector(":scope > circle")
    )
      return;

    // Don't pan when clicking on buttons or select elements
    if (e.target.closest("button") || e.target.closest("select")) return;

    const svg =
      expanded.querySelector("pre.mermaid svg") ||
      expanded.querySelector("svg");
    if (!svg || !expanded._baseViewBox) return;

    const elW = svg.clientWidth || parseInt(svg.getAttribute("width"));
    const vbW = expanded._baseViewBox.w / expandedZoom;
    const scale = vbW / elW;

    panState = {
      container: expanded,
      startX: e.clientX,
      startY: e.clientY,
      startPan: { ...(expanded._panOffset || { x: 0, y: 0 }) },
      scale,
    };
    expanded.classList.add("panning");
    e.preventDefault();
  });

  document.addEventListener("mousemove", (e) => {
    if (!panState) return;
    const dx = (e.clientX - panState.startX) * panState.scale;
    const dy = (e.clientY - panState.startY) * panState.scale;
    // Drag right → viewport moves left → viewBox x decreases
    panState.container._panOffset = {
      x: panState.startPan.x - dx,
      y: panState.startPan.y - dy,
    };
    updateExpandedViewBox(panState.container);
  });

  document.addEventListener("mouseup", () => {
    if (panState) {
      panState.container.classList.remove("panning");
      panState = null;
    }
  });
}

function updateExpandedViewBox(container) {
  const svg =
    container.querySelector("pre.mermaid svg") ||
    container.querySelector("svg");
  if (!svg) return;

  const base = container._baseViewBox;
  if (!base) return;

  const pan = container._panOffset || { x: 0, y: 0 };

  // Zoom: smaller viewBox = zoomed in, centered on base
  const vbW = base.w / expandedZoom;
  const vbH = base.h / expandedZoom;
  const vbX = base.x + (base.w - vbW) / 2 + pan.x;
  const vbY = base.y + (base.h - vbH) / 2 + pan.y;
  svg.setAttribute("viewBox", `${vbX} ${vbY} ${vbW} ${vbH}`);
}

function applyExpandedZoom(container) {
  updateExpandedViewBox(container);
  showZoomIndicator(container);
}

function showZoomIndicator(container) {
  let indicator = container.querySelector(".zoom-indicator");
  if (!indicator) {
    indicator = document.createElement("div");
    indicator.className = "zoom-indicator";
    container.appendChild(indicator);
  }
  indicator.textContent = `${Math.round(expandedZoom * 100)}%`;
  indicator.style.opacity = "0.7";

  clearTimeout(zoomIndicatorTimeout);
  zoomIndicatorTimeout = setTimeout(() => {
    indicator.style.opacity = "0";
  }, 1500);
}

// Initialize
// ============================================

document.addEventListener("DOMContentLoaded", async () => {
  await loadScalaJS();

  // Router
  initRouter();

  // Karpathy Intro page
  document
    .getElementById("karpathy-evaluate")
    .addEventListener("click", karpathyEvaluate);

  // Expression page
  document
    .getElementById("expression")
    .addEventListener("input", updateVariableInputs);
  document.getElementById("evaluate").addEventListener("click", evaluate);
  updateVariableInputs();

  // Neural Network page
  document.getElementById("nn-init").addEventListener("click", initNN);
  document
    .getElementById("nn-add-sample")
    .addEventListener("click", () => addTrainingSample());
  document.getElementById("nn-train").addEventListener("click", nnTrain);
  document.getElementById("nn-forward").addEventListener("click", nnForward);
  document
    .getElementById("visualize-network")
    .addEventListener("click", visualizeNetwork);
  document
    .getElementById("network-view-toggle")
    .addEventListener("change", visualizeNetwork);
  document
    .getElementById("build-topology")
    .addEventListener("click", buildFromTopology);

  // Block DSL page
  document
    .getElementById("block-build")
    .addEventListener("click", buildBlockNetwork);
  document
    .getElementById("block-view-toggle")
    .addEventListener("change", buildBlockNetwork);
  document.querySelectorAll(".block-preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.getElementById("block-spec").value = btn.dataset.spec;
      buildBlockNetwork();
    });
  });

  // Initial setup
  updateVariableInputs();
  populateNetworkSelector();
  setupExpandableGraphs();

  // Auto-evaluate Karpathy page on load
  if (hashToPage(location.hash) === "page-karpathy") {
    karpathyEvaluate();
  }
});
