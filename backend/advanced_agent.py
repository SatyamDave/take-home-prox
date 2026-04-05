"""
Claude-driven reasoning agent for the Vulcan OmniPro 220.

Architecture:
1. LLM (Claude via OpenRouter) — intent extraction, evidence synthesis, explanation
2. Constraint engine — hard safety decisions, outcome classification, violation detection
3. Simulation engine — state transitions for polarity, duty cycle, troubleshooting
4. Artifact builders — state-driven visual outputs

The LLM is NEVER allowed to decide safety or override constraints.
It only parses queries, synthesizes evidence, and generates explanation text.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import anthropic
from openai import OpenAI

from constraint_engine import ConstraintEngine
from domain_knowledge import WeldingDomainKnowledge
from query_planner import QueryPlanner
from simulation_engine import SimulationEngine
from vector_store import VectorStore

load_dotenv()

SYSTEM_PROMPT = """You are a technical reasoning engine for the Vulcan OmniPro 220 welding system.

You receive a user query and retrieved manual evidence.
Your job is to:
1. Classify the user intent
2. Extract context parameters from the query
3. Synthesize the retrieved evidence into structured claims
4. Identify any missing parameters

CRITICAL RULES:
- NEVER fabricate specifications not present in the evidence
- NEVER override physical constraints
- If parameters are missing, flag them in missing_params
- If the query is out of domain, set intent to "out_of_domain"
- Be precise and deterministic

Respond ONLY with valid JSON matching this schema:
{
  "intent": "polarity" | "duty_cycle" | "setup" | "troubleshooting" | "out_of_domain" | "general",
  "context": {
    "process": string | null,
    "material": string | null,
    "thickness": string | null,
    "voltage": string | null,
    "amperage": number | null
  },
  "evidence_claims": [
    {"claim": string, "page": number, "source": string}
  ],
  "missing_params": [string],
  "is_reversed_polarity_query": boolean,
  "is_continuous_operation_query": boolean
}
"""


class AdvancedVulcanAgent:
    def __init__(self, vector_store: VectorStore, knowledge_base: Dict[str, Any]):
        self.vector_store = vector_store
        self.knowledge_base = knowledge_base
        self.query_planner = QueryPlanner()  # Fallback for LLM failure
        self.domain_knowledge = WeldingDomainKnowledge()
        self.constraint_engine = ConstraintEngine(self.domain_knowledge)
        self.simulation_engine = SimulationEngine()

        # Primary: Anthropic SDK (direct Claude access)
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_client = None
        self.claude_model = "claude-sonnet-4-20250514"
        if anthropic_key and anthropic_key != "your-api-key-here":
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
            except Exception:
                pass

        # Fallback: OpenRouter (if reviewer provides OPENROUTER_API_KEY instead)
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_client = None
        self.model = "anthropic/claude-sonnet-4"
        if openrouter_key:
            try:
                self.openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
            except Exception:
                pass

        # If neither is available, all LLM calls fall back to regex
        self.llm_available = self.anthropic_client is not None or self.openrouter_client is not None

    # ──────────────────────────────────────────────────────────────
    #  Main entry point
    # ──────────────────────────────────────────────────────────────

    def chat(self, user_message: str, image_data: Optional[str] = None) -> Dict[str, Any]:
        # 1. LLM parses intent + context
        analysis = self._llm_parse_query(user_message)

        primary_intent = analysis["primary_intent"]
        context = analysis["context"]

        # 2. Out-of-domain guard
        if primary_intent == "out_of_domain":
            return self._build_out_of_domain_response(user_message)

        # 3. Missing-parameter guard (LLM-identified + deterministic check)
        missing = self._check_missing_params(primary_intent, context)
        if missing:
            return self._build_insufficient_state_response(primary_intent, missing)

        # 4. Retrieve evidence (multi-hop)
        retrieved = self._multi_hop_retrieval(user_message, analysis)
        evidence = self._build_evidence_bundle(retrieved)

        # 5. LLM synthesizes evidence into structured claims
        evidence_claims = analysis.get("evidence_claims", [])

        # 6. Build simulation state
        simulation = self._build_simulation(user_message, analysis, evidence)

        # 7. Constraint engine validates — this is the AUTHORITY on safety
        validation = self.constraint_engine.validate_state(analysis, simulation, user_message)

        # 8. Build artifact even for constraint failures (polarity, duty cycle, etc.)
        artifact = self._build_artifact(user_message, analysis, evidence, simulation, validation)

        # 9. If constraint violation → return decision with artifact
        if not validation["valid"]:
            explanation = self._llm_generate_explanation(
                user_message, analysis, [], simulation, validation, artifact
            )
            decision = self._build_decision_package(analysis, simulation, validation)
            technical_response = {
                "outcome": {
                    "level": decision["outcome"]["level"],
                    "headline": decision["outcome"]["headline"],
                    "valid": decision["valid"],
                    "reason": decision["outcome"]["reason"],
                },
                "instruction": decision["instruction"],
                "consequences": decision["consequences"],
                "constraint_trace": decision["constraint_trace"],
                "explanation": explanation,
                "state": simulation["state"],
                "simulation": simulation["steps"],
                "comparison": simulation.get("comparison"),
                "artifact": artifact,
                "confidence": {"label": "high", "score": 1.0},
                "assumptions": ["Invalid machine state detected by constraint engine."],
                "sources": [],
            }
            return {
                "text": self._format_response_text(technical_response),
                "artifacts": [artifact] if artifact else [],
                "images": [],
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "technical_response": technical_response,
                "metadata": {
                    "query_intent": analysis["primary_intent"],
                    "retrieved_nodes": len(evidence["nodes"]),
                    "artifact_generated": artifact is not None,
                },
            }

        # 10. LLM generates explanation from structured data
        explanation = self._llm_generate_explanation(
            user_message, analysis, evidence_claims, simulation, validation, artifact
        )

        # 11. Assemble response
        assumptions = self._collect_assumptions(analysis, evidence)
        sources = self._collect_sources(evidence)
        confidence = self._estimate_confidence(evidence, assumptions)
        decision = self._build_decision_package(analysis, simulation, validation)

        technical_response = {
            "outcome": {
                "level": decision["outcome"]["level"],
                "headline": decision["outcome"]["headline"],
                "valid": decision["valid"],
                "reason": decision["outcome"]["reason"],
            },
            "instruction": decision["instruction"],
            "consequences": decision["consequences"],
            "constraint_trace": decision["constraint_trace"],
            "explanation": explanation,
            "state": simulation["state"],
            "simulation": simulation["steps"],
            "comparison": simulation.get("comparison"),
            "artifact": artifact,
            "confidence": confidence,
            "assumptions": assumptions,
            "sources": sources,
        }

        artifacts = [artifact] if artifact else []

        return {
            "text": self._format_response_text(technical_response),
            "artifacts": artifacts,
            "images": self._select_supporting_images(evidence),
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "technical_response": technical_response,
            "metadata": {
                "query_intent": primary_intent,
                "retrieved_nodes": len(evidence["nodes"]),
                "artifact_generated": artifact is not None,
            },
        }

    # ──────────────────────────────────────────────────────────────
    #  LLM: Query parsing (intent extraction)
    # ──────────────────────────────────────────────────────────────

    def _llm_parse_query(self, user_message: str) -> Dict[str, Any]:
        if not self.openrouter_client and not self.anthropic_client:
            return self._fallback_parse_query(user_message)

        try:
            if self.openrouter_client:
                resp = self.openrouter_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0, max_tokens=600,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content or "{}"
            else:
                resp = self.anthropic_client.messages.create(
                    model=self.claude_model,
                    max_tokens=600,
                    temperature=0,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )
                raw = resp.content[0].text if resp.content else "{}"
            parsed = json.loads(raw)
            return self._normalize_parsed_analysis(parsed, user_message)
        except Exception:
            return self._fallback_parse_query(user_message)

    def _fallback_parse_query(self, user_message: str) -> Dict[str, Any]:
        return self.query_planner.analyze_query_intent(user_message)

    def _normalize_parsed_analysis(self, parsed: Dict[str, Any], user_message: str) -> Dict[str, Any]:
        intent = parsed.get("intent", "general")
        context = parsed.get("context", {})

        # Cross-validate with regex fallbacks
        ql = user_message.lower()
        if "tig" in ql and not context.get("process"):
            context["process"] = "TIG"
        elif "mig" in ql and not context.get("process"):
            context["process"] = "MIG"
        elif "stick" in ql and not context.get("process"):
            context["process"] = "STICK"
        elif "flux" in ql and not context.get("process"):
            context["process"] = "FLUX CORE"

        # Extract voltage
        if not context.get("voltage"):
            if "120" in ql or "110" in ql:
                context["voltage"] = "120V"
            elif "240" in ql or "220" in ql or "230" in ql:
                context["voltage"] = "240V"

        # Extract amperage
        if not context.get("amperage"):
            amp_match = re.search(r'(\d+)\s*(?:a|amp)s?', ql)
            if amp_match:
                val = int(amp_match.group(1))
                if 30 <= val <= 300:
                    context["amperage"] = float(val)

        # Extract thickness
        if not context.get("thickness"):
            frac_match = re.search(r'(\d+)/(\d+)', ql)
            if frac_match:
                context["thickness"] = f"{frac_match.group(1)}/{frac_match.group(2)}"

        # Extract material
        if not context.get("material"):
            if "aluminum" in ql or "aluminium" in ql:
                context["material"] = "aluminum"
            elif "stainless" in ql:
                context["material"] = "stainless steel"
            elif "steel" in ql or "mild" in ql:
                context["material"] = "mild steel"

        return {
            "primary_intent": intent,
            "context": context,
            "evidence_claims": parsed.get("evidence_claims", []),
            "missing_params": parsed.get("missing_params", []),
            "is_reversed_polarity_query": parsed.get("is_reversed_polarity_query", False),
            "is_continuous_operation_query": parsed.get("is_continuous_operation_query", False),
            "original_query": user_message,
            "complexity": len([v for v in context.values() if v]),
        }

    # ──────────────────────────────────────────────────────────────
    #  LLM: Evidence synthesis
    # ──────────────────────────────────────────────────────────────

    def _llm_generate_explanation(
        self,
        query: str,
        analysis: Dict[str, Any],
        evidence_claims: List[Dict[str, Any]],
        simulation: Dict[str, Any],
        validation: Dict[str, Any],
        artifact: Optional[Dict[str, Any]],
    ) -> str:
        if not self.openrouter_client and not self.anthropic_client:
            return self._rule_based_explanation(query, analysis, simulation, validation)

        state = simulation.get("state", {})
        instruction = validation.get("instruction", "")
        consequences = validation.get("consequences", [])

        prompt = f"""You are explaining a technical decision for the Vulcan OmniPro 220 welder.

User query: {query}
Machine state: {json.dumps(state, indent=2) if state else "No specific state."}
Outcome: {validation.get('outcome', 'SAFE')}
Instruction: {instruction}
Consequences: {json.dumps(consequences, indent=2)}
Evidence claims from manual: {json.dumps(evidence_claims, indent=2)}

RULES:
- Short, direct sentences. No hedging ("may", "might", "typically", "can").
- Deterministic causal language. Max 4 sentences.
- Explain WHY the instruction is correct based on the evidence.
- Sound like a senior technician, not a chatbot.

Write the explanation:"""

        try:
            if self.openrouter_client:
                resp = self.openrouter_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3, max_tokens=200,
                )
                return resp.choices[0].message.content.strip()
            else:
                resp = self.anthropic_client.messages.create(
                    model=self.claude_model,
                    max_tokens=200,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text.strip() if resp.content else self._rule_based_explanation(query, analysis, simulation, validation)
        except Exception:
            return self._rule_based_explanation(query, analysis, simulation, validation)

    def _rule_based_explanation(
        self,
        query: str,
        analysis: Dict[str, Any],
        simulation: Dict[str, Any],
        validation: Dict[str, Any],
    ) -> str:
        return self._state_summary(analysis, simulation.get("state", {}), validation.get("consequences", []), validation.get("outcome", "SAFE"))

    # ──────────────────────────────────────────────────────────────
    #  Response builders
    # ──────────────────────────────────────────────────────────────

    def _build_out_of_domain_response(self, user_message: str) -> Dict[str, Any]:
        technical_response = {
            "outcome": {
                "level": "INSUFFICIENT_STATE",
                "headline": "INSUFFICIENT STATE — Out of Domain",
                "valid": False,
                "reason": "Query is outside the supported welding domain.",
            },
            "instruction": "Provide a valid welding-related query about the Vulcan OmniPro 220.",
            "consequences": [
                {"label": "Immediate", "text": "The system cannot process queries unrelated to welding."},
                {"label": "Short term", "text": "No machine state can be determined."},
                {"label": "Continued use", "text": "The system remains unable to assist."},
            ],
            "constraint_trace": [],
            "explanation": "This system is specialized for Vulcan OmniPro 220 operations only.",
            "state": {},
            "simulation": [],
            "comparison": {},
            "artifact": None,
            "confidence": {"label": "low", "score": 0.0},
            "assumptions": ["Query is outside the welding domain."],
            "sources": [],
        }
        return {
            "text": self._format_response_text(technical_response),
            "artifacts": [],
            "images": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "technical_response": technical_response,
            "metadata": {"query_intent": "out_of_domain", "retrieved_nodes": 0, "artifact_generated": False},
        }

    def _check_missing_params(self, intent: str, context: Dict[str, Any]) -> List[str]:
        required = {
            "polarity": ["process"],
            "duty_cycle": ["amperage", "voltage"],
            "setup": ["material", "thickness", "process"],
            "troubleshooting": [],
        }.get(intent, [])
        return [p for p in required if not context.get(p)]

    def _build_insufficient_state_response(self, intent: str, missing: List[str]) -> Dict[str, Any]:
        technical_response = {
            "outcome": {
                "level": "INSUFFICIENT_STATE",
                "headline": "INSUFFICIENT STATE — Missing Information",
                "valid": False,
                "reason": f"Missing required parameters for {intent}: {', '.join(missing)}.",
            },
            "instruction": f"Specify required parameters: {', '.join(missing)}.",
            "consequences": [
                {"label": "Immediate", "text": "The system cannot proceed without the missing parameters."},
                {"label": "Short term", "text": "No machine state can be reliably computed."},
                {"label": "Continued use", "text": "The system remains unable to provide a deterministic decision."},
            ],
            "constraint_trace": [],
            "explanation": "Deterministic decisions require complete input parameters.",
            "state": {},
            "simulation": [],
            "comparison": {},
            "artifact": None,
            "confidence": {"label": "low", "score": 0.0},
            "assumptions": [f"Missing parameters: {', '.join(missing)}."],
            "sources": [],
        }
        return {
            "text": self._format_response_text(technical_response),
            "artifacts": [],
            "images": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "technical_response": technical_response,
            "metadata": {"query_intent": intent, "retrieved_nodes": 0, "artifact_generated": False},
        }

    # ──────────────────────────────────────────────────────────────
    #  Retrieval (deterministic)
    # ──────────────────────────────────────────────────────────────

    def _multi_hop_retrieval(self, query: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        plans = [query]
        plans.extend(self._expand_query(query, analysis))

        results: List[Dict[str, Any]] = []
        seen_ids = set()
        for plan in plans[:8]:
            for hit in self.vector_store.search(plan, n_results=6):
                node_id = hit["metadata"].get("node_id")
                if node_id in seen_ids:
                    continue
                seen_ids.add(node_id)
                results.append(hit)

        related = self._follow_relationships(results)
        for hit in related:
            node_id = hit["metadata"].get("node_id")
            if node_id in seen_ids:
                continue
            seen_ids.add(node_id)
            results.append(hit)

        return sorted(results, key=lambda item: item.get("distance", 1.0))[:18]

    def _expand_query(self, query: str, analysis: Dict[str, Any]) -> List[str]:
        context = analysis["context"]
        expansions = []
        if analysis["primary_intent"] == "duty_cycle":
            expansions.extend(["duty cycle table", "rated output duty cycle"])
        if analysis["primary_intent"] == "polarity":
            expansions.extend(["polarity diagram", "torch work clamp connection", "DCEN DCEP"])
        if analysis["primary_intent"] == "troubleshooting":
            expansions.extend(["troubleshooting chart", "defect cause solution", "porosity spatter causes"])
        if context.get("process"):
            expansions.append(f"{context['process']} settings")
        if context.get("material"):
            expansions.append(f"{context['material']} welding settings")
        if context.get("thickness"):
            expansions.append(f"{context['thickness']} thickness chart")
        if context.get("amperage"):
            expansions.append(f"{int(context['amperage'])}A output")
        if context.get("voltage"):
            expansions.append(context["voltage"])
        return expansions

    def _follow_relationships(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        nodes_by_id = {node["id"]: node for node in self.knowledge_base.get("knowledge_nodes", [])}
        related: List[Dict[str, Any]] = []
        seed_ids = {hit["metadata"].get("node_id") for hit in hits}
        for rel in self.knowledge_base.get("relationships", []):
            if rel["source"] not in seed_ids and rel["target"] not in seed_ids:
                continue
            other_id = rel["target"] if rel["source"] in seed_ids else rel["source"]
            node = nodes_by_id.get(other_id)
            if not node:
                continue
            related.append({
                "text": self._node_to_text(node),
                "metadata": {
                    "node_id": node["id"],
                    "node_type": node["type"],
                    "page": str(node["page"]),
                    "source": node["source"],
                    "title": node.get("title", ""),
                },
                "distance": 0.55,
            })
        return related

    def _build_evidence_bundle(self, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
        nodes_by_id = {node["id"]: node for node in self.knowledge_base.get("knowledge_nodes", [])}
        nodes = []
        grouped = defaultdict(list)
        for hit in hits:
            node_id = hit["metadata"].get("node_id")
            node = nodes_by_id.get(node_id)
            if not node:
                continue
            nodes.append({**node, "distance": hit.get("distance", 1.0)})
            grouped[node["type"]].append(node)
        return {"nodes": nodes, "grouped": grouped}

    # ──────────────────────────────────────────────────────────────
    #  Simulation (deterministic)
    # ──────────────────────────────────────────────────────────────

    def _build_simulation(self, query: str, analysis: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
        intent = analysis["primary_intent"]
        context = analysis["context"]

        if intent == "polarity":
            return self._simulate_polarity(query, context)
        elif intent == "duty_cycle":
            return self._simulate_duty_cycle(context)
        elif intent == "troubleshooting":
            return self._simulate_troubleshooting(query, context)
        else:
            return self._simulate_setup(context)

    def _simulate_polarity(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        process = context.get("process")
        material = context.get("material")
        voltage = context.get("voltage")
        amperage = int(context["amperage"]) if context.get("amperage") else None

        if not process:
            return {
                "state": {}, "baselineState": {}, "comparison": {},
                "steps": [{"step": 1, "event": "Cannot simulate", "stateKey": "state_t0", "effect": "Missing process."}],
                "effects": ["Insufficient state."], "mode": "insufficient",
            }

        correct = self.domain_knowledge.infer_polarity(process, material)["polarity"]
        reversed_req = self._query_requests_reversed_polarity(query)
        return self.simulation_engine.simulate_polarity_transition(
            process=process, material=material, voltage=voltage,
            amperage=amperage, expected_polarity=correct, reverse=reversed_req,
        )

    def _simulate_duty_cycle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        process = context.get("process", "MIG")
        material = context.get("material", "mild steel")
        voltage = context.get("voltage", "240V")
        amperage = int(context["amperage"]) if context.get("amperage") else 160

        inferred = self.domain_knowledge.infer_duty_cycle(amperage, voltage)
        duty_cycle = int(inferred["estimated_duty_cycle"])
        range_min, range_max = self.domain_knowledge.duty_cycle_patterns.get(
            voltage, self.domain_knowledge.duty_cycle_patterns["240V"]
        )["range"]

        state = {
            "components": {"powerSource": {"status": "ready", "inputVoltage": voltage}},
            "process": process, "material": material,
            "constraints": {
                "targetAmperage": amperage, "dutyCycle": duty_cycle,
                "supportedAmperageRange": [range_min, range_max],
                "lookupTable": {f"{process}_{amperage}A_{voltage.replace('V', '')}V": duty_cycle},
            },
            "derived": {
                "weldWindowMinutes": round((duty_cycle / 100) * 10, 1),
                "cooldownMinutes": round(10 - (duty_cycle / 100) * 10, 1),
                "weldOutcome": "thermal window holds" if duty_cycle > 35 else "thermal limit arrives quickly",
            },
        }
        steps = [
            {"step": 1, "event": "Set operating point", "effect": f"{amperage}A on {voltage}"},
            {"step": 2, "event": "Apply thermal envelope", "effect": f"duty cycle locks to {duty_cycle}%"},
            {"step": 3, "event": "Simulate ten-minute window", "effect": f"weld {state['derived']['weldWindowMinutes']}min, cool {state['derived']['cooldownMinutes']}min"},
        ]
        return {"state": state, "steps": steps, "effects": [s["effect"] for s in steps], "mode": "nominal"}

    def _simulate_troubleshooting(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        diagnosis = self.domain_knowledge.diagnose_weld_defect(query)[0]
        state = {
            "process": context.get("process"),
            "material": context.get("material"),
            "components": {"powerSource": {"status": "ready", "inputVoltage": context.get("voltage")}},
            "constraints": {"targetAmperage": int(context["amperage"]) if context.get("amperage") else None},
            "derived": {
                "activeDefect": diagnosis["defect"].replace("_", " "),
                "causeChain": diagnosis["likely_causes"],
                "weldOutcome": "defect state active",
            },
        }
        steps = [
            {"step": 1, "event": "Observe symptom", "effect": diagnosis["defect"].replace("_", " ")},
            {"step": 2, "event": "Expand cause chain", "effect": ", ".join(diagnosis["likely_causes"][:3])},
            {"step": 3, "event": "Prioritize intervention", "effect": "clear the defect state before restarting"},
        ]
        return {"state": state, "steps": steps, "effects": [s["effect"] for s in steps], "mode": "nominal"}

    def _simulate_setup(self, context: Dict[str, Any]) -> Dict[str, Any]:
        process = context.get("process")
        material = context.get("material")
        thickness = context.get("thickness")
        voltage = context.get("voltage")

        if material and thickness:
            amp_info = self.domain_knowledge.infer_amperage_from_material(material, thickness)
            recommended = int(amp_info["recommended_start"])
            amp_range = list(amp_info["amperage_range"])
        else:
            recommended = None
            amp_range = None

        state = {
            "process": process, "material": material, "thickness": thickness,
            "components": {"powerSource": {"status": "ready", "inputVoltage": voltage}},
            "constraints": {
                "targetAmperage": recommended,
                "recommendedAmperageRange": amp_range,
                "expectedPolarity": self.domain_knowledge.infer_polarity(process, material)["polarity"] if process and material else None,
            },
            "derived": {"weldOutcome": "stable setup block"},
        }
        steps = [
            {"step": 1, "event": "Assemble setup state", "effect": f"{process or '?'} for {thickness or '?'} {material or '?'}"},
            {"step": 2, "event": "Lock recommended operating point", "effect": f"start at {recommended}A" if recommended else "no reference point found"},
        ]
        return {"state": state, "steps": steps, "effects": [s["effect"] for s in steps], "mode": "nominal"}

    # ──────────────────────────────────────────────────────────────
    #  Artifact builders (state-driven)
    # ──────────────────────────────────────────────────────────────

    def _build_artifact(
        self, query: str, analysis: Dict[str, Any], evidence: Dict[str, Any],
        simulation: Dict[str, Any], validation: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        intent = analysis["primary_intent"]
        if intent == "polarity":
            return self._build_polarity_diagram(analysis, simulation, validation)
        if intent == "duty_cycle":
            return self._build_duty_cycle_visualizer(analysis, simulation, validation)
        if intent == "troubleshooting":
            return self._build_troubleshooting_tree(query, analysis, validation)
        if analysis["context"].get("material") or analysis["context"].get("thickness"):
            return self._build_parameter_explorer(analysis, simulation, validation)
        if evidence["grouped"].get("table"):
            return self._build_interactive_table(evidence)
        return None

    def _build_polarity_diagram(self, analysis: Dict[str, Any], simulation: Dict[str, Any], validation: Dict[str, Any]) -> Dict[str, Any]:
        process = analysis["context"].get("process") or simulation.get("state", {}).get("process", "Welder")
        material = analysis["context"].get("material") or simulation.get("state", {}).get("components", {}).get("workpiece", {}).get("material", "steel")
        polarity = self.domain_knowledge.infer_polarity(process, material)["polarity"] if process != "Welder" else "N/A"
        comparison = simulation.get("comparison", {})
        before = comparison.get("before", simulation["state"])
        after = comparison.get("after", simulation["state"])

        if not before or not after:
            before = after = simulation.get("state", {})

        after_torch = after.get("components", {}).get("torch", {}).get("terminal", "negative")
        after_work = after.get("components", {}).get("workClamp", {}).get("terminal", "positive")

        return {
            "type": "polarity_diagram",
            "title": f"{process} polarity layout",
            "data": {
                "process": process, "polarity": polarity,
                "outcome": validation["outcome"], "outcomeHeadline": validation["headline"],
                "simulationMode": simulation["mode"],
                "comparison": simulation.get("comparison"),
                "statusBadges": [
                    {"label": validation["headline"], "tone": "danger" if validation["outcome"] in {"FAILURE RISK", "DAMAGE RISK"} else "success"},
                    {"label": "Heat shifts to the torch" if simulation["mode"] == "fault" else "Heat stays in the workpiece", "tone": "warning" if simulation["mode"] == "fault" else "info"},
                ],
                "components": [
                    {"id": "welder", "type": "machine", "label": "Welder", "x": 40, "y": 130, "width": 120, "height": 90, "color": "#0f766e", "terminals": [{"id": "positive", "label": "+", "x": 160, "y": 155, "color": "#dc2626"}, {"id": "negative", "label": "-", "x": 160, "y": 195, "color": "#111827"}]},
                    {"id": "torch", "type": "torch", "label": f"{process} Torch", "x": 380, "y": 60, "width": 110, "height": 44, "color": "#2563eb"},
                    {"id": "workpiece", "type": "metal", "label": "Work Clamp", "x": 380, "y": 190, "width": 120, "height": 50, "color": "#6b7280"},
                ],
                "connections": [
                    {"id": "torch_cable", "from": {"component": "welder", "terminal": after_torch}, "to": {"component": "torch", "point": {"x": 380, "y": 82}}, "color": "#2563eb", "label": f"Torch ({after_torch})", "animated": True},
                    {"id": "work_cable", "from": {"component": "welder", "terminal": after_work}, "to": {"component": "workpiece", "point": {"x": 380, "y": 215}}, "color": "#dc2626" if after_work == "positive" else "#111827", "label": f"Work clamp ({after_work})", "animated": True},
                ],
                "notes": [f"{process} on {material} requires {polarity}.", validation["instruction"]],
            },
        }

    def _build_duty_cycle_visualizer(self, analysis: Dict[str, Any], simulation: Dict[str, Any], validation: Dict[str, Any]) -> Dict[str, Any]:
        state = simulation["state"]
        amperage = int(state["constraints"]["targetAmperage"])
        voltage = int(str(state["components"]["powerSource"]["inputVoltage"]).replace("V", ""))
        process = state["process"]
        duty_value = int(state["constraints"]["dutyCycle"])
        weld_min = round(state["derived"]["weldWindowMinutes"], 1)
        cool_min = round(10 - weld_min, 1)
        return {
            "type": "duty_cycle_visualizer",
            "title": f"{process} duty cycle at {amperage}A / {voltage}V",
            "data": {
                "process": process, "voltage": voltage, "amperage": amperage,
                "outcome": validation["outcome"], "outcomeHeadline": validation["headline"],
                "dutyCycle": duty_value, "timeWindow": 10,
                "visualization": {
                    "type": "heat_bar",
                    "segments": [
                        {"duration": weld_min, "state": "welding", "color": "#dc2626", "label": f"Weld {weld_min} min"},
                        {"duration": cool_min, "state": "cooling", "color": "#0ea5e9", "label": f"Cool {cool_min} min"},
                    ],
                    "heatLevels": {"welding": 100, "cooling": 20, "critical": 80},
                },
                "calculator": {"enabled": True, "ranges": {"voltage": [120, 240], "amperage": {"min": 30, "max": 220, "step": 10}}},
                "statusBadges": [{"label": validation["headline"], "tone": "danger" if validation["outcome"] in {"FAILURE RISK", "DAMAGE RISK"} else "success"}],
            },
        }

    def _build_troubleshooting_tree(self, query: str, analysis: Dict[str, Any], validation: Dict[str, Any]) -> Dict[str, Any]:
        diagnoses = self.domain_knowledge.diagnose_weld_defect(query)
        defect = diagnoses[0]
        return {
            "type": "troubleshooting_tree",
            "title": "Troubleshooting path",
            "data": {
                "problem": defect["defect"].replace("_", " "),
                "rootNode": "start",
                "nodes": [
                    {"id": "start", "type": "question", "text": "What symptom best matches?", "options": [{"label": defect["defect"].replace("_", " "), "next": "checks"}]},
                    {"id": "checks", "type": "solution", "text": "Required checks", "steps": defect["likely_causes"], "severity": "moderate", "icon": "wrench"},
                ],
                "statusBadges": [{"label": validation["headline"], "tone": "danger"}, {"label": "Action Required", "tone": "warning"}],
            },
        }

    def _build_parameter_explorer(self, analysis: Dict[str, Any], simulation: Dict[str, Any], validation: Dict[str, Any]) -> Dict[str, Any]:
        state = simulation["state"]
        material = state.get("material") or analysis["context"].get("material") or "mild steel"
        thickness = state.get("thickness") or analysis["context"].get("thickness") or "1/8"
        rec_amp = int(state["constraints"]["targetAmperage"]) if state.get("constraints", {}).get("targetAmperage") else 160
        amp_range = state.get("constraints", {}).get("recommendedAmperageRange") or [rec_amp - 15, rec_amp + 15]
        params = self._infer_process_parameters(material, thickness, rec_amp)
        return {
            "type": "parameter_explorer",
            "title": f"{material.title()} setup block",
            "data": {
                "defaultMaterial": material, "defaultThickness": thickness,
                "recommendedStartAmperage": rec_amp, "amperageRange": amp_range,
                "materials": ["mild steel", "stainless steel", "aluminum"],
                "thicknesses": ["1/16", "1/8", "1/4", "3/8"],
                "process": state.get("process") or analysis["context"].get("process") or "MIG",
                "outcome": validation["outcome"], "outcomeHeadline": validation["headline"],
                "parameterCards": [
                    {"label": "Voltage", "value": params["voltage"], "tone": "info"},
                    {"label": "Wire Speed", "value": params["wire_speed"], "tone": "info"},
                    {"label": "Amperage", "value": params["amperage"], "tone": "success"},
                    {"label": "Gas", "value": params["gas"], "tone": "warning"},
                ],
                "statusBadges": [{"label": validation["headline"], "tone": "success" if validation["outcome"] == "SAFE" else "danger"}],
                "notes": [validation["instruction"], "Travel speed and stickout are secondary adjustments."],
            },
        }

    def _infer_process_parameters(self, material: str, thickness: str, rec_amp: int) -> Dict[str, str]:
        tk = thickness.replace('"', '').strip()
        gas = "100% Argon" if material == "aluminum" else "Tri-mix" if "stainless" in material else "C25"
        voltage_map = {"1/4": "20-22V", "1/8": "17-19V", "3/8": "23-25V"}
        ws_map = {"1/4": "300-380 IPM", "1/8": "220-300 IPM", "3/8": "360-450 IPM"}
        return {
            "voltage": voltage_map.get(tk, "15-17V"),
            "wire_speed": ws_map.get(tk, "160-240 IPM"),
            "amperage": f"{max(30, rec_amp - 15)}-{rec_amp + 15}A",
            "gas": gas,
            "confidence_label": "Recommended Start",
        }

    def _build_interactive_table(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        table = evidence["grouped"]["table"][0]
        return {
            "type": "interactive_table",
            "title": table.get("title", "Specification table"),
            "data": {
                "headers": table.get("data", {}).get("columns", []),
                "rows": table.get("data", {}).get("rows", []),
                "page": table["page"], "source": table["source"],
            },
        }

    # ──────────────────────────────────────────────────────────────
    #  Decision + formatting helpers
    # ──────────────────────────────────────────────────────────────

    def _build_decision_package(self, analysis: Dict[str, Any], simulation: Dict[str, Any], validation: Dict[str, Any]) -> Dict[str, Any]:
        reason = validation["violations"][0]["detail"] if validation.get("violations") else "No constraint violation detected."
        return {
            "valid": validation["valid"],
            "outcome": {"level": validation["outcome"], "headline": validation["headline"], "reason": reason},
            "instruction": validation["instruction"],
            "consequences": validation["consequences"],
            "constraint_trace": validation["constraint_trace"],
            "summary": self._state_summary(analysis, simulation.get("state", {}), validation.get("consequences", []), validation.get("outcome", "SAFE")),
        }

    def _state_summary(self, analysis: Dict[str, Any], state: Dict[str, Any], consequences: List[Dict[str, str]], outcome: str) -> str:
        outcomes = {
            "SAFE": "The machine state is valid and ready for operation.",
            "SUBOPTIMAL": "The machine state is valid but operates with reduced efficiency.",
            "FAILURE RISK": "The machine state is invalid and will produce a defective weld.",
            "DAMAGE RISK": "The machine state is critically unsafe and will damage equipment.",
            "INSUFFICIENT_STATE": "The system lacks sufficient information to determine a state.",
        }
        return outcomes.get(outcome, "The machine state has been evaluated.")

    def _format_response_text(self, tr: Dict[str, Any]) -> str:
        o = tr.get("outcome", {})
        lines = [
            f"{o.get('headline', 'UNKNOWN')}",
            "",
            tr.get("instruction", ""),
            "",
        ]
        for c in tr.get("consequences", []):
            lines.append(f"{c['label']}: {c['text']}")
        lines.append("")
        lines.append(tr.get("explanation", ""))
        return "\n".join(lines)

    def _collect_assumptions(self, analysis: Dict[str, Any], evidence: Dict[str, Any]) -> List[str]:
        assumptions: List[str] = []
        context = analysis["context"]
        if not context.get("process"):
            assumptions.append("Process not specified — constraint engine requires it for setup queries.")
        if analysis["primary_intent"] == "polarity" and not evidence["grouped"].get("diagram"):
            assumptions.append("No explicit wiring diagram extracted. Decision uses the process polarity rule.")
        return assumptions

    def _collect_sources(self, evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{"source": n["source"], "page": n["page"], "node_type": n["type"], "title": n.get("title", "")} for n in evidence["nodes"][:6]]

    def _estimate_confidence(self, evidence: Dict[str, Any], assumptions: List[str]) -> Dict[str, Any]:
        score = 0.45 + min(0.25, len(evidence["grouped"].get("table", [])) * 0.12) + min(0.15, len(evidence["grouped"].get("diagram", [])) * 0.1) + min(0.15, len(evidence["nodes"]) * 0.01)
        score -= min(0.2, len(assumptions) * 0.08)
        score = max(0.2, min(0.95, score))
        return {"label": "high" if score >= 0.78 else "medium" if score >= 0.55 else "low", "score": round(score, 2)}

    def _select_supporting_images(self, evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        pages = {n["page"] for n in evidence["nodes"][:4]}
        selected = []
        for img in self.knowledge_base.get("images", []):
            if img["page"] in pages:
                selected.append(img)
            if len(selected) >= 2:
                break
        return selected

    def _node_to_text(self, node: Dict[str, Any]) -> str:
        content = node.get("content", "")
        if node.get("data"):
            content += f"\nStructured data: {node['data']}"
        if node.get("steps"):
            content += "\nSteps: " + " | ".join(node["steps"])
        return content

    def _query_requests_reversed_polarity(self, query: str) -> bool:
        lowered = query.lower()
        direct = ["wrong polarity", "reversed polarity", "reverse polarity", "polarity reversed", "polarity is reversed", "backwards polarity", "polarity backwards", "swap polarity", "swapped polarity"]
        if any(t in lowered for t in direct):
            return True
        return bool(re.search(r"(reverse|reversed|wrong|backward|backwards|swap|swapped).{0,24}polarity", lowered) or re.search(r"polarity.{0,24}(reverse|reversed|wrong|backward|backwards|swap|swapped)", lowered))

    def analyze_weld_defect(self, image_base64: str) -> Dict[str, Any]:
        return {"defects": ["visual analysis not enabled"], "severity": "medium", "causes": ["image reasoning not connected"], "solutions": ["Use the troubleshooting tree."], "settingsRecommendation": None}

    def reset_conversation(self):
        pass
