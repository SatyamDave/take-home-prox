"""
Query planning and decomposition.
Breaks down complex queries into sub-queries and plans multi-hop searches.
"""

from typing import List, Dict, Any, Optional
import re
import json


class QueryPlanner:
    """
    Decomposes user queries into structured search plans.
    Handles complex queries that need multiple lookups and reasoning steps.
    """

    def __init__(self):
        self.query_patterns = {
            "duty_cycle": [
                "duty cycle",
                "duty",
                "continuous",
                "how long",
                "runtime",
                "overheating",
                "thermal",
            ],
            "wire_speed": ["wire speed", "feed rate", "ipm", "wire feed"],
            "amperage": ["amps", "amperage", "current", "how many amps"],
            "voltage": [
                "voltage",
                "volts",
                "120v",
                "240v",
                "110v",
                "220v",
                "power",
                "inverter",
                "generator",
            ],
            "polarity": [
                "polarity",
                "dcep",
                "dcen",
                "electrode positive",
                "electrode negative",
                "cable",
                "cables",
                "flip",
                "torch",
                "work clamp",
                "reverse",
                "reversed",
                "wrong",
                "swap",
                "connection",
            ],
            "gas": ["gas", "shielding gas", "argon", "co2", "flow rate"],
            "troubleshooting": [
                "problem",
                "issue",
                "defect",
                "spatter",
                "porosity",
                "not working",
                "won't",
                "arc",
                "unstable",
                "weak",
                "crack",
                "burn through",
                "backlight",
                "screen",
                "display",
                "lcd",
                "error",
                "fault",
            ],
            "setup": [
                "setup",
                "set up",
                "configure",
                "how to",
                "settings",
                "should i use",
                "what do i use",
                "hooked up",
                "connect",
                "plug in",
            ],
            "material": ["steel", "aluminum", "stainless", "material", "metal"],
            "thickness": ["thick", "thickness", "gauge", "1/8", "1/4", "1/16"],
            "process": [
                "mig",
                "tig",
                "stick",
                "flux core",
                "process",
                "weld",
                "welding",
                "welder",
            ],
        }

    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze the user's intent and extract key entities.
        """
        query_lower = query.lower()

        # Detect query types
        detected_types = []
        for query_type, keywords in self.query_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_types.append(query_type)

        # Extract numeric values (amperage, voltage, thickness)
        numbers = re.findall(
            r"\b(\d+(?:\.\d+)?)\s*([aA]|[vV]|amps?|volts?|inches?|mm|gauge)?\b", query
        )

        # Extract fractions (for thickness)
        fractions = re.findall(r"\b(\d+)/(\d+)\b", query)

        # Determine primary intent
        primary_intent = self._determine_primary_intent(detected_types, query_lower)

        # Extract context
        context = {
            "voltage": self._extract_voltage(query_lower, numbers),
            "amperage": self._extract_amperage(query_lower, numbers),
            "thickness": self._extract_thickness(query_lower, fractions),
            "material": self._extract_material(query_lower),
            "process": self._extract_process(query_lower),
        }

        return {
            "primary_intent": primary_intent,
            "detected_types": detected_types,
            "context": context,
            "original_query": query,
            "complexity": len(detected_types),
        }

    def _determine_primary_intent(self, detected_types: List[str], query: str) -> str:
        """
        Determine what the user is primarily asking about.
        """
        # Priority order for intent
        priority = [
            "troubleshooting",  # Problems first
            "polarity",  # Connection/wiring questions before setup
            "duty_cycle",  # Then specific questions
            "wire_speed",
            "amperage",
            "setup",
            "gas",
            "process",
        ]

        for intent in priority:
            if intent in detected_types:
                return intent

        # If no specific welding-related intent is detected, classify as out_of_domain
        # This catches nonsense, unrelated, or generic queries
        return "out_of_domain"

    def _extract_voltage(self, query: str, numbers: List[tuple]) -> Optional[str]:
        """Extract voltage from query."""
        if "120" in query or "110" in query:
            return "120V"
        if "240" in query or "220" in query or "230" in query:
            return "240V"

        # Look for numbers with 'v' or 'volt'
        for num, unit in numbers:
            if unit and unit.lower() in ["v", "volt", "volts"]:
                val = float(num)
                if 100 <= val <= 130:
                    return "120V"
                elif 200 <= val <= 250:
                    return "240V"

        return None

    def _extract_amperage(self, query: str, numbers: List[tuple]) -> Optional[float]:
        """Extract amperage from query."""
        for num, unit in numbers:
            if unit and unit.lower() in ["a", "amp", "amps"]:
                return float(num)

        # If no unit but reasonable amperage range
        for num, unit in numbers:
            val = float(num)
            if not unit and 30 <= val <= 300:  # Typical welding range
                return val

        return None

    def _extract_thickness(self, query: str, fractions: List[tuple]) -> Optional[str]:
        """Extract material thickness."""
        if fractions:
            # Return first fraction found
            num, denom = fractions[0]
            return f"{num}/{denom}"

        # Look for gauge
        if "gauge" in query or "ga" in query:
            gauge_match = re.search(r"(\d+)\s*(?:gauge|ga)", query)
            if gauge_match:
                return f"{gauge_match.group(1)}ga"

        return None

    def _extract_material(self, query: str) -> Optional[str]:
        """Extract material type."""
        if "aluminum" in query or "aluminium" in query:
            return "aluminum"
        if "stainless" in query:
            return "stainless steel"
        if "steel" in query or "mild steel" in query:
            return "mild steel"

        return None

    def _extract_process(self, query: str) -> Optional[str]:
        """Extract welding process."""
        query_lower = query.lower()

        if "mig" in query_lower:
            return "MIG"
        if "tig" in query_lower:
            return "TIG"
        if "stick" in query_lower:
            return "STICK"
        if "flux" in query_lower and "core" in query_lower:
            return "FLUX CORE"

        return None

    def create_search_plan(
        self, query_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Create a multi-step search plan based on query analysis.
        Each step represents a search or reasoning operation.
        """
        plan = []
        intent = query_analysis["primary_intent"]
        context = query_analysis["context"]
        detected_types = query_analysis["detected_types"]

        # Step 1: Always do a direct search first
        plan.append(
            {
                "step": 1,
                "action": "search",
                "query": query_analysis["original_query"],
                "description": "Direct search for user query",
                "n_results": 5,
            }
        )

        # Step 2: Based on intent, add targeted searches
        if intent == "duty_cycle":
            # Search for duty cycle tables/specs
            duty_search = "duty cycle " + (context["process"] or "MIG")
            if context["voltage"]:
                duty_search += f" {context['voltage']}"

            plan.append(
                {
                    "step": 2,
                    "action": "search",
                    "query": duty_search,
                    "description": "Search for duty cycle specifications",
                    "n_results": 3,
                }
            )

            # Also search for duty cycle tables
            plan.append(
                {
                    "step": 3,
                    "action": "search",
                    "query": "duty cycle table chart specifications",
                    "description": "Find duty cycle tables",
                    "n_results": 3,
                }
            )

        elif intent == "wire_speed":
            plan.append(
                {
                    "step": 2,
                    "action": "search",
                    "query": f"wire feed speed settings {context.get('process', 'MIG')}",
                    "description": "Search for wire speed settings",
                    "n_results": 3,
                }
            )

        elif intent == "troubleshooting":
            # Search for troubleshooting section
            plan.append(
                {
                    "step": 2,
                    "action": "search",
                    "query": "troubleshooting problems defects solutions",
                    "description": "Search troubleshooting guide",
                    "n_results": 5,
                }
            )

        elif intent == "setup":
            plan.append(
                {
                    "step": 2,
                    "action": "search",
                    "query": "setup installation initial configuration",
                    "description": "Search setup instructions",
                    "n_results": 4,
                }
            )

        # Step 3: If we have material + thickness, search for settings
        if context.get("material") and context.get("thickness"):
            plan.append(
                {
                    "step": len(plan) + 1,
                    "action": "search",
                    "query": f"{context['material']} {context['thickness']} settings recommended",
                    "description": "Search for material-specific settings",
                    "n_results": 3,
                }
            )

        # Step 4: Add cross-reference step with domain knowledge
        plan.append(
            {
                "step": len(plan) + 1,
                "action": "cross_reference",
                "description": "Apply domain knowledge to infer missing specs",
                "context": context,
            }
        )

        # Step 5: Synthesize results
        plan.append(
            {
                "step": len(plan) + 1,
                "action": "synthesize",
                "description": "Combine search results with inferred knowledge",
                "intent": intent,
            }
        )

        return plan

    def should_decompose_query(self, query_analysis: Dict[str, Any]) -> bool:
        """
        Determine if query is complex enough to need decomposition.
        """
        # Complex if multiple types detected
        if query_analysis["complexity"] >= 2:
            return True

        # Complex if asking for relationships (e.g., "max wire speed for MIG")
        if any(
            word in query_analysis["original_query"].lower()
            for word in ["max", "maximum", "best", "optimal", "recommended"]
        ):
            return True

        return False

    def refine_search_queries(
        self, original_query: str, search_results: List[Dict], iteration: int = 0
    ) -> Optional[str]:
        """
        Given initial search results, generate a refined follow-up query.
        This enables iterative search refinement.
        """
        if iteration >= 3:  # Max 3 iterations
            return None

        # If we got very few results, broaden the search
        if len(search_results) < 2:
            # Extract key terms and search more broadly
            key_terms = re.findall(
                r"\b[A-Z]{2,}\b|\b\d+[AVav]\b|\b[a-z]{4,}\b", original_query
            )
            if key_terms:
                return " ".join(key_terms[:3])  # Simpler query

        # If results are low confidence (high distance), try alternative terms
        if search_results and search_results[0].get("distance", 0) > 0.7:
            # Try synonyms
            synonyms = {
                "duty cycle": "continuous operation rating",
                "wire speed": "wire feed rate",
                "amperage": "current output",
                "polarity": "electrode connection",
            }

            for term, synonym in synonyms.items():
                if term in original_query.lower():
                    return original_query.lower().replace(term, synonym)

        return None
