import json
from typing import Dict, Any, List, Optional

import google.generativeai as genai
from langgraph.graph import StateGraph, END


class FeedbackReflectionAgent:
    """
    Feedback & Reflection Agent (LangGraph + Gemini)
    - Evaluates the generated answer for correctness and completeness against the provided context
    - Produces a brief evaluation and suggests an optional follow-up question to deepen understanding
    """

    def __init__(self, gemini_api_key: str, model_name: str = "gemini-1.5-flash"):
        self.gemini_api_key = gemini_api_key
        self.model_name = model_name
        self._model = None
        self._graph = None
        self._initialize_llm()
        self._build_graph()

    def _initialize_llm(self):
        genai.configure(api_key=self.gemini_api_key)
        self._model = genai.GenerativeModel(self.model_name)

    def _initial_state(
        self,
        query: str,
        presented_answer: str,
        context_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return {
            "query": query,
            "presented_answer": presented_answer or "",
            "context": {
                "retrieved_chunks": context_chunks or [],
            },
            "evaluation": None,
            "follow_up": None,
        }

    def _evaluate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        chunks = state.get("context", {}).get("retrieved_chunks", [])
        prompt = f"""
        You are a critical teaching assistant. Evaluate the quality of the provided answer with respect to the user's query
        and the given context snippets. Identify correctness, completeness, clarity, and cite any gaps or risks.
        Then craft ONE concise follow-up question that would help the student learn better.

        Return STRICT JSON with keys:
        {{
          "evaluation": "2-4 sentences evaluating correctness/completeness/clarity",
          "follow_up": "one short question like 'Would you like an example on X?' or null if not needed"
        }}

        Query: {state.get('query')}
        Presented Answer: {state.get('presented_answer')}
        Context Snippets:
        {json.dumps([{"text": c.get("text",""), "score": c.get("final_score", 0.0)} for c in chunks][:6], indent=2)}
        """
        try:
            resp = self._model.generate_content(prompt)
            text = (resp.text or "{}").strip()
            if text.startswith("```"):
                text = text.strip("`\n")
                if text.lower().startswith("json\n"):
                    text = text[5:]
            data = json.loads(text)
        except Exception:
            data = {
                "evaluation": "Answer generated. Unable to run detailed evaluation at this time.",
                "follow_up": "Was this clear? Would you like an example problem?",
            }
        state["evaluation"] = data.get("evaluation")
        state["follow_up"] = data.get("follow_up")
        return state

    def _build_graph(self):
        g = StateGraph(dict)
        g.add_node("evaluate", self._evaluate_node)
        g.set_entry_point("evaluate")
        g.set_finish_point("evaluate")
        self._graph = g.compile()

    def evaluate_and_follow_up(
        self,
        query: str,
        presented_answer: str,
        context_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        state = self._initial_state(query, presented_answer, context_chunks)
        out = self._graph.invoke(state)
        return {
            "evaluation": out.get("evaluation"),
            "follow_up": out.get("follow_up"),
        }
