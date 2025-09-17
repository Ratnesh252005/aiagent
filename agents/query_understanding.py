import os
import json
from typing import Dict, List, Any

import google.generativeai as genai
from langgraph.graph import StateGraph, END
from langgraph.constants import Send


class QueryUnderstandingAgent:
    """
    Uses LangGraph + Gemini to (1) classify user intent and (2) optionally decompose
    complex questions into sub-questions to guide retrieval and answer planning.
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

    # ----- Graph state -----
    def _initial_state(self, question: str) -> Dict[str, Any]:
        return {
            "question": question,
            "intent": None,
            "sub_questions": [],
            "confidence": None
        }

    # ----- Nodes -----
    def _classify_intent_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        You are a classification helper for a study assistant. Given the user question,
        classify the intent into one of: ["explain", "quiz", "summary", "other"].
        Also, if the question is complex, decompose it into 2-5 sub-questions that will
        help retrieve relevant content. Return strict JSON with keys: intent, confidence (0-1), sub_questions (array of strings).

        Question: {state.get('question')}

        Respond ONLY with JSON.
        """
        # Retry with simple exponential backoff to handle rate limits
        data = None
        last_err = None
        for attempt in range(3):
            try:
                resp = self._model.generate_content(prompt)
                text = resp.text or "{}"
                # try to extract JSON from code block if present
                text = text.strip()
                if text.startswith("```"):
                    # remove markdown code fences
                    text = text.strip("`\n")
                    # in case it's like ```json\n{...}```
                    if text.lower().startswith("json\n"):
                        text = text[5:]
                data = json.loads(text)
                break
            except Exception as e:
                last_err = e
                # crude check for rate/quota wording; sleep and retry
                msg = str(e).lower()
                if "quota" in msg or "rate" in msg:
                    import time
                    time.sleep(2 * (attempt + 1))
                    continue
                else:
                    break
        if data is None:
            # fallback minimal
            data = {"intent": "explain", "confidence": 0.5, "sub_questions": []}

        state["intent"] = data.get("intent", "explain")
        state["confidence"] = data.get("confidence", 0.5)
        subs = data.get("sub_questions", []) or []
        if isinstance(subs, list):
            state["sub_questions"] = [s for s in subs if isinstance(s, str) and s.strip()][:5]
        else:
            state["sub_questions"] = []
        return state

    # ----- Graph build and run -----
    def _build_graph(self):
        graph = StateGraph(dict)
        graph.add_node("classify_intent", self._classify_intent_node)
        graph.set_entry_point("classify_intent")
        graph.set_finish_point("classify_intent")
        self._graph = graph.compile()

    def analyze_query(self, question: str) -> Dict[str, Any]:
        """Run the LangGraph to analyze a question and return intent + sub-questions."""
        state = self._initial_state(question)
        out = self._graph.invoke(state)
        return {
            "intent": out.get("intent", "explain"),
            "confidence": out.get("confidence", 0.5),
            "sub_questions": out.get("sub_questions", [])
        }
