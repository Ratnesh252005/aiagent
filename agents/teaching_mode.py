import json
from typing import Dict, Any, List, Optional

import google.generativeai as genai
from langgraph.graph import StateGraph, END


class TeachingModeAgent:
    """
    Teaching Mode Agent (LangGraph + Gemini)
    - Decides how to present an answer: explain, quiz, or summary
    - Consumes upstream intent (from QueryUnderstandingAgent) but can also infer
    - Produces a structured payload with formatted output and optional quiz items
    """

    def __init__(self, gemini_api_key: str, model_name: str = "gemini-1.5-flash"):
        self.gemini_api_key = gemini_api_key
        self.model_name = model_name
        self._model = None
        self._graph = None
        self._build_graph()

    def _initialize_llm(self):
        genai.configure(api_key=self.gemini_api_key)
        self._model = genai.GenerativeModel(self.model_name)
    
    def _ensure_model(self):
        if self._model is None:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self._model = genai.GenerativeModel(self.model_name)
            except Exception:
                self._model = None

    # ----- State -----
    def _initial_state(
        self,
        query: str,
        base_answer: str,
        mode: Optional[str],
        context_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return {
            "query": query,
            "base_answer": base_answer or "",
            "mode": (mode or "explain").lower(),
            "context": {
                "retrieved_chunks": context_chunks or [],
            },
            "final_presentation": None,
            "quiz_items": [],
            "summary_points": [],
        }

    # ----- Nodes -----
    def _render_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        mode = state.get("mode", "explain")
        chunks = state.get("context", {}).get("retrieved_chunks", [])
        prompt = f"""
        You are a teaching assistant. Given the user's query, a base answer, and some context snippets,
        present the response in the requested mode. Supported modes:
        - explain: Provide a clear, step-by-step explanation with structure (headings/bullets), include key formulae if relevant.
        - quiz: Generate 4-6 practice questions (mix MCQ/short), include an answer key at the end.
        - summary: Provide a concise 5-line summary (each line <= 20 words), focusing on essentials.

        Always return STRICT JSON with keys:
        {{
          "mode": "explain|quiz|summary",
          "final_presentation": "string with markdown",
          "quiz_items": [{{"question": "...", "options": ["A","B","C","D"], "answer": "A"}}],
          "summary_points": ["point1", "point2", "point3", "point4", "point5"]
        }}

        Mode: {mode}
        Query: {state.get('query')}
        Base Answer: {state.get('base_answer')}
        Context Snippets:
        {json.dumps([{"text": c.get("text",""), "score": c.get("final_score", 0.0)} for c in chunks][:6], indent=2)}
        """
        self._ensure_model()
        try:
            resp = self._model.generate_content(prompt) if self._model else None
            if resp is None:
                raise RuntimeError("Model not available")
            text = (resp.text or "{}").strip()
            if text.startswith("```"):
                text = text.strip("`\n")
                if text.lower().startswith("json\n"):
                    text = text[5:]
            data = json.loads(text)
        except Exception:
            # Fallback minimal formatting
            data = {
                "mode": mode,
                "final_presentation": state.get("base_answer") or "",
                "quiz_items": [],
                "summary_points": [],
            }
        state["mode"] = (data.get("mode") or mode).lower()
        state["final_presentation"] = data.get("final_presentation") or state.get("base_answer")
        state["quiz_items"] = data.get("quiz_items", [])
        state["summary_points"] = data.get("summary_points", [])
        return state

    # ----- Build graph -----
    def _build_graph(self):
        g = StateGraph(dict)
        g.add_node("render", self._render_node)
        g.set_entry_point("render")
        g.set_finish_point("render")
        self._graph = g.compile()

    # ----- Public API -----
    def present_answer(
        self,
        query: str,
        base_answer: str,
        mode: Optional[str],
        context_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        state = self._initial_state(query, base_answer, mode, context_chunks)
        out = self._graph.invoke(state)
        return {
            "mode": out.get("mode", "explain"),
            "final_presentation": out.get("final_presentation", base_answer),
            "quiz_items": out.get("quiz_items", []),
            "summary_points": out.get("summary_points", []),
        }
