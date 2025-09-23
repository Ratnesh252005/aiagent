import json
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from langgraph.graph import StateGraph, END

class ReasoningPlannerAgent:
    """
    A reasoning and planning agent that uses LangGraph + Gemini to:
    1. Analyze complex queries
    2. Break them down into logical steps
    3. Plan the retrieval and reasoning process
    4. Generate structured reasoning chains
    """

    def __init__(self, gemini_api_key: str, model_name: str = "gemini-1.5-flash", retriever_agent: Any = None):
        self.gemini_api_key = gemini_api_key
        self.model_name = model_name
        self.retriever_agent = retriever_agent
        self._model = None
        self._graph = None
        self._initialize_llm()
        self._build_graph()

    def _initialize_llm(self):
        """Initialize the Gemini model with the provided API key."""
        genai.configure(api_key=self.gemini_api_key)
        self._model = genai.GenerativeModel(self.model_name)

    def _initial_state(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Initialize the state for the reasoning graph."""
        return {
            "query": query,
            "context": context or {},
            "reasoning_steps": [],
            "retrieval_plan": [],
            "final_answer": None,
            "confidence": 0.0
        }

    def _analyze_query_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the query and plan the reasoning approach."""
        prompt = """
        You are a reasoning and planning assistant. Your task is to analyze the user's query and plan how to answer it.
        
        For the given query, please:
        1. Identify the key information needed
        2. Determine if multi-step reasoning is required
        3. Plan the retrieval strategy
        4. Outline the reasoning steps
        
        Query: {query}
        
        Respond with a JSON object containing:
        - "reasoning_required": boolean indicating if multi-step reasoning is needed
        - "retrieval_plan": list of information to retrieve
        - "reasoning_steps": list of steps to solve the problem
        - "expected_answer_format": description of the expected answer format
        """.format(query=state["query"])

        response = self._get_llm_response(prompt)
        plan = response.get("plan", {})
        
        state["reasoning_steps"] = plan.get("reasoning_steps", [])
        state["retrieval_plan"] = plan.get("retrieval_plan", [])
        state["reasoning_required"] = plan.get("reasoning_required", False)
        
        return state

    def _execute_retrieval_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the retrieval plan using the provided RetrieverAgent if available."""
        # Ensure context container exists
        if "context" not in state or state["context"] is None:
            state["context"] = {}

        # If no retriever wired, pass through
        if not getattr(self, "retriever_agent", None):
            return state

        # Determine what to retrieve
        ctx = state.get("context", {})
        questions_for_retrieval: List[str] = ctx.get("sub_questions") or state.get("retrieval_plan") or [state.get("query", "")]
        document_id: Optional[str] = ctx.get("document_id")
        top_k: int = int(ctx.get("top_k") or 5)
        base_query: str = state.get("query", "")

        try:
            top_context = self.retriever_agent.retrieve(
                questions=questions_for_retrieval,
                top_k=top_k,
                document_id=document_id,
                base_query=base_query,
            )
            # Store retrieved chunks back into context for downstream nodes / UI
            state["context"]["retrieved_chunks"] = top_context
        except Exception as e:
            # On failure, keep existing state and proceed
            state["context"]["retrieved_chunks"] = []
        return state

    def _reasoning_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the actual reasoning based on the plan and retrieved information."""
        if not state["reasoning_required"]:
            return state
            
        # Honor force_reasoning flag from context (if present)
        if state.get("context", {}).get("force_reasoning"):
            state["reasoning_required"] = True

        prompt = """
        Based on the following query and context, generate a well-reasoned answer.
        
        Query: {query}
        
        Reasoning Steps:
        {reasoning_steps}
        
        Context:
        {context}
        
        Provide your answer in the following format:
        {{
            "answer": "The final answer",
            "confidence": 0.0-1.0,
            "reasoning_chain": ["step 1", "step 2", ...]
        }}
        """.format(
            query=state["query"],
            reasoning_steps="\n".join(f"- {step}" for step in state["reasoning_steps"]),
            context=json.dumps(state.get("context", {}), indent=2)
        )
        
        response = self._get_llm_response(prompt)
        
        if isinstance(response, dict):
            state["final_answer"] = response.get("answer")
            state["confidence"] = response.get("confidence", 0.5)
            state["reasoning_chain"] = response.get("reasoning_chain", [])
        
        return state

    def _get_llm_response(self, prompt: str) -> Dict:
        """Get a response from the LLM and parse it as JSON."""
        try:
            response = self._model.generate_content(prompt)
            text = response.text.strip()
            
            # Clean up the response if it's in a code block
            if text.startswith("```"):
                text = text.strip("`\n")
                if text.lower().startswith("json\n"):
                    text = text[5:]
            
            return json.loads(text)
        except Exception as e:
            print(f"Error getting LLM response: {str(e)}")
            return {}

    def _build_graph(self):
        """Build the LangGraph for the reasoning process."""
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("execute_retrieval", self._execute_retrieval_node)
        workflow.add_node("perform_reasoning", self._reasoning_node)
        
        # Define edges
        workflow.add_edge("analyze_query", "execute_retrieval")
        workflow.add_conditional_edges(
            "execute_retrieval",
            lambda state: "perform_reasoning" if state.get("reasoning_required") else END,
            {
                "perform_reasoning": "perform_reasoning",
                END: END
            }
        )
        workflow.add_edge("perform_reasoning", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_query")
        
        # Compile the graph
        self._graph = workflow.compile()

    def process_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a query using the reasoning and planning agent.
        
        Args:
            query: The user's query
            context: Optional context or additional information
            
        Returns:
            Dict containing the reasoning process and final answer
        """
        state = self._initial_state(query, context)
        result = self._graph.invoke(state)
        
        # Ensure we have a final answer or fallback
        if not result.get("final_answer") and result.get("reasoning_steps"):
            result["final_answer"] = " ".join(result["reasoning_steps"][-1:]) if result["reasoning_steps"] else "I couldn't generate a complete answer."
        
        return result
