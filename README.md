# Prox Founding Engineer Challenge — Vulcan OmniPro 220 Agent

> **Manuals are not text. They are compressed machine behavior.** This project turns a 48-page welding manual into a multimodal reasoning agent that can draw wiring diagrams, visualize duty cycles, walk through troubleshooting trees, and simulate failure modes — not just quote paragraphs.

![Vulcan OmniPro 220](product.webp)

## Quick Start (2 minutes)

```bash
git clone https://github.com/SatyamDave/take-home-prox.git
cd take-home-prox/prox-challenge

# Set your API key
cp .env.example .env
# Edit .env and add your Anthropic API key: ANTHROPIC_API_KEY=sk-ant-...

# Install & run
./setup.sh
./run.sh
```

Then open **http://localhost:5173**

Or run the hosted version at: **https://huggingface.co/spaces/SatyamDave/vulcan-agent**

---

## What This Solves

Technical support for complex physical products breaks because product knowledge doesn't live in plain text. A welder manual is not just paragraphs — it's:

- Wiring topology and polarity rules
- Duty-cycle constraint matrices
- Process-dependent setup procedures
- Troubleshooting decision trees
- Weld diagnosis images
- Implicit technician knowledge

Traditional RAG flattens this into chunks and loses the structure. They can quote a page, but they can't explain how the machine behaves.

## The Core Idea

Instead of retrieving text from manuals, this system:

1. **Builds a structured model** of the machine (text nodes, tables, procedures, diagrams, relationships)
2. **Retrieves across node types** — pulling together the spec table, the setup guide, and the visual context for the same question
3. **Simulates behavior** from the current machine state (e.g., wrong polarity → cable state → current flow → heat distribution → weld outcome)
4. **Explains with visual artifacts** — not text walls. The system chooses the right presentation for the task.

## What The System Does

### 1. Structured Knowledge Extraction

The backend converts PDFs into typed knowledge nodes:

| Node Type | Purpose | Example |
|-----------|---------|---------|
| `text` | Explanative sections | Process descriptions |
| `table` | Constraint matrices | Duty cycle specs, voltage tables |
| `procedure` | Step-by-step ops | Polarity setup for TIG |
| `diagram` | Visual configuration | Wiring schematics, weld diagnosis |
| `relationships` | Cross-node connections | All nodes about "polarity" linked |

### 2. Multi-Hop Retrieval

Queries are expanded, retrieved across structured nodes, and enriched by following relationships. The system pulls together the table, the setup guidance, and the visual context for the same question.

### 3. Reasoning & Simulation

The agent constructs an internal machine state and runs a lightweight simulation loop.

Example: `wrong polarity → cable state → current flow → heat distribution → weld outcome`

This is especially visible in polarity questions, where the system distinguishes between expected polarity and actual polarity, then predicts the effect of the mismatch.

### 4. Multimodal Response Engine

The response is not locked to text. The system chooses an artifact based on the task:

| Artifact | When Used |
|----------|-----------|
| **Polarity Diagram** | Cable setup, torch/workpiece wiring |
| **Duty Cycle Visualizer** | Operating windows, thermal constraints |
| **Troubleshooting Tree** | Diagnostic workflows, weld defects |
| **Parameter Explorer** | Setup configuration by material/thickness |
| **Interactive Spec Table** | Reference data from the manual |

When something is too cognitively hard to explain in words, the agent draws it.

## Example: Polarity Question

**User asks:** *"What happens if polarity is reversed for TIG?"*

The system:
1. Constructs machine state (TIG → DCEN → torch on negative)
2. Simulates reversed current flow (torch on positive → DCEP)
3. Propagates effects to heat balance and weld quality
4. Renders a **visual polarity diagram** showing initial vs computed state
5. Provides a consequence chain: immediate → short-term → continued use

That's the difference between documentation retrieval and machine reasoning.

## Architecture

```
prox-challenge/
├── backend/
│   ├── main.py                 # FastAPI server + API endpoints
│   ├── advanced_agent.py       # Agent: retrieval, reasoning, simulation, artifact selection
│   ├── knowledge_extractor.py  # PDF → structured knowledge nodes
│   ├── vector_store.py         # Multi-type node indexing & retrieval
│   ├── simulation_engine.py    # Machine state simulation loop
│   ├── constraint_engine.py    # Duty cycle & spec reasoning
│   ├── reasoning_engine.py     # Technical reasoning & inference
│   ├── verification_engine.py  # Answer verification
│   ├── synthesis_engine.py     # Response synthesis
│   ├── query_planner.py        # Query expansion & planning
│   ├── domain_knowledge.py     # Hand-crafted domain rules
│   └── knowledge_base.json     # Pre-extracted knowledge base
├── frontend/
│   ├── src/App.tsx             # Chat UI with reasoning, evidence, artifacts
│   ├── src/components/ArtifactRenderer.tsx  # Visual artifact rendering
│   └── ...                     # React + TypeScript + TailwindCSS
├── files/                      # Source PDF manuals
└── product.webp                # Product image
```

### Backend Pipeline

1. **Knowledge Extractor** → Parses PDFs into typed nodes with relationships
2. **Vector Store** → Indexes all nodes for retrieval (ChromaDB + sentence-transformers)
3. **Advanced Agent** → Orchestrates retrieval, simulation, and response generation:
   - Parse user query
   - Expand query & retrieve across node types
   - Follow relationships for enrichment
   - Build machine state
   - Run simulation if applicable
   - Synthesize response
   - Select best artifact type
   - Format structured response

### Frontend

- React + TypeScript chat interface
- Reasoning summary with evidence and assumptions
- Interactive artifact rendering (SVG diagrams, charts, trees)
- API key management UI
- Weld defect image upload

## Best Demo Queries

- **"What happens if polarity is reversed for TIG?"** — Shows polarity diagram + simulation
- **"What's the duty cycle for MIG welding at 200A on 240V?"** — Duty cycle visualizer + spec retrieval
- **"I'm getting porosity in my flux-cored welds. What should I check?"** — Troubleshooting tree
- **"What are the recommended settings for welding 1/4 inch mild steel?"** — Parameter explorer

## Environment Variables

Create a `.env` file in the project root:

```bash
ANTHROPIC_API_KEY=sk-ant-api03-...
```

The system also supports OpenRouter as an alternative:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

If neither key is set, the system runs in **fallback mode** (deterministic reasoning without LLM).

## Deployed Version

Live on Hugging Face Spaces: **https://huggingface.co/spaces/SatyamDave/vulcan-agent**

## Design Decisions

### Why not just use RAG on text chunks?

Because the knowledge in a technical manual is not just text. The duty cycle is a matrix. The polarity setup is a wiring diagram. The weld diagnosis is a set of labeled photos. Chunking loses the structure that makes the knowledge useful.

### Why simulation over retrieval?

Retrieval can find the polarity page. Simulation can tell you what happens when you configure it wrong. The agent builds an internal model of the machine and propagates state changes — the same way a senior technician reasons through a problem.

### Why artifacts instead of long text answers?

Because a technician standing in front of a welder doesn't want to read a wall of text. They want to see which cable goes where, what the duty cycle window looks like, and what steps to follow. Visual artifacts match how people actually use technical information.

### Knowledge base pre-extraction

The `knowledge_base.json` is pre-built from the PDFs so the app starts instantly. The extraction pipeline is rerunnable if manuals change.
