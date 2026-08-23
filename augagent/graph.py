"""Stateful Graph Orchestration for AugAgent.

Provides a robust DAG (Directed Acyclic Graph) engine where nodes are functions
and edges define the control flow. Supports parallel execution and typed state reducers.
"""

import asyncio
from typing import Any, Callable, Dict, Union, Coroutine, Awaitable, List

from augagent.telemetry import get_logger

State = Dict[str, Any]
NodeFunc = Callable[[State], Union[State, Coroutine[Any, Any, State]]]
ConditionFunc = Callable[[State], str]

class StateGraph:
    def __init__(self, state_schema: Any = None):
        self.nodes: Dict[str, NodeFunc] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[str, ConditionFunc] = {}
        self.entry_point: str = ""
        self.reducers: Dict[str, Callable[[Any, Any], Any]] = {}
        self.state_schema = state_schema

    def add_node(self, name: str, func: NodeFunc):
        self.nodes[name] = func

    def set_entry_point(self, name: str):
        self.entry_point = name

    def add_edge(self, from_node: str, to_node: str):
        self.edges[from_node] = to_node

    def add_conditional_edges(self, from_node: str, condition: ConditionFunc, route_map: Dict[str, str]):
        """Condition function returns a key that maps to the next node via route_map."""
        self.conditional_edges[from_node] = lambda state: route_map.get(condition(state), "__end__")

    def add_reducer(self, key: str, reducer: Callable[[Any, Any], Any]):
        """Register a custom reducer for a state key (e.g., list append)."""
        self.reducers[key] = reducer

    async def execute(self, initial_state: State, max_steps: int = 50) -> State:
        logger = get_logger()
        state = initial_state.copy()
        current_nodes = [self.entry_point]

        for step in range(max_steps):
            if not current_nodes or current_nodes == ["__end__"]:
                break

            logger.log_info(f"[Graph] Executing nodes: {current_nodes}")
            
            # Fan-out: Execute all current nodes in parallel
            tasks = []
            for node in current_nodes:
                if node == "__end__": continue
                if node not in self.nodes:
                    raise ValueError(f"Node '{node}' not found in graph.")
                
                func = self.nodes[node]
                result = func(state.copy()) # Pass copy for parallel safety
                if asyncio.iscoroutine(result):
                    tasks.append(result)
                else:
                    # Wrap sync result in a future
                    future: Any = asyncio.Future()
                    future.set_result(result)
                    tasks.append(future)  # type: ignore

            # Fan-in: gather results
            if tasks:
                results = await asyncio.gather(*tasks)
                
                # State merging with reducers
                for result_dict in results:
                    if not result_dict: continue
                    for k, v in result_dict.items():
                        if k in self.reducers and k in state:
                            state[k] = self.reducers[k](state[k], v)
                        else:
                            state[k] = v
                
                # Schema validation if provided
                if self.state_schema:
                    try:
                        state = self.state_schema.model_validate(state).model_dump()
                    except Exception as e:
                        logger.log_error(f"[Graph] State schema validation failed: {e}")
                        raise

            # Determine next nodes
            next_nodes = []
            for node in current_nodes:
                if node == "__end__": continue
                if node in self.conditional_edges:
                    nxt = self.conditional_edges[node](state)
                    if nxt not in next_nodes: next_nodes.append(nxt)
                elif node in self.edges:
                    nxt = self.edges[node]
                    if nxt not in next_nodes: next_nodes.append(nxt)
                else:
                    if "__end__" not in next_nodes: next_nodes.append("__end__")

            current_nodes = next_nodes

        return state

    def to_mermaid(self) -> str:
        """Export graph as a Mermaid diagram."""
        lines = ["graph TD"]
        for node in self.nodes:
            lines.append(f"    {node}({node})")
        for f, t in self.edges.items():
            lines.append(f"    {f} --> {t}")
        for f in self.conditional_edges:
            lines.append(f"    {f} -. conditional .-> ...")
        return "\n".join(lines)
