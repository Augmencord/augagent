"""Stateful Graph Orchestration for AugAgent.

Provides a DAG (Directed Acyclic Graph) engine where nodes are agent actions
or generic functions, and edges define the control flow. The graph shares a
state object across all nodes.
"""

from typing import Any, Callable, Dict, Union, Coroutine
import asyncio
from augagent.telemetry import get_logger

# State can be any dict
State = Dict[str, Any]

NodeFunc = Callable[[State], Union[State, Coroutine[Any, Any, State]]]
ConditionFunc = Callable[[State], str]

class StateGraph:
    def __init__(self):
        self.nodes: Dict[str, NodeFunc] = {}
        self.edges: Dict[str, Dict[str, str]] = {}
        self.conditional_edges: Dict[str, ConditionFunc] = {}
        self.entry_point: str = ""

    def add_node(self, name: str, func: NodeFunc):
        self.nodes[name] = func

    def set_entry_point(self, name: str):
        self.entry_point = name

    def add_edge(self, from_node: str, to_node: str):
        if from_node not in self.edges:
            self.edges[from_node] = {}
        self.edges[from_node]["__default__"] = to_node

    def add_conditional_edges(self, from_node: str, condition: ConditionFunc):
        """Condition function should return the name of the next node."""
        self.conditional_edges[from_node] = condition

    async def execute(self, initial_state: State, max_steps: int = 50) -> State:
        logger = get_logger()
        state = initial_state.copy()
        current_node = self.entry_point

        for step in range(max_steps):
            if current_node == "__end__":
                break

            if current_node not in self.nodes:
                raise ValueError(f"Node '{current_node}' not found in graph.")

            logger.log_info(f"[Graph] Executing node: {current_node}")
            
            # Execute node
            node_func = self.nodes[current_node]
            result = node_func(state)
            if asyncio.iscoroutine(result):
                result = await result
            
            # Update state (shallow merge)
            state.update(result)

            # Determine next node
            next_node = "__end__"
            if current_node in self.conditional_edges:
                next_node = self.conditional_edges[current_node](state)
            elif current_node in self.edges and "__default__" in self.edges[current_node]:
                next_node = self.edges[current_node]["__default__"]

            current_node = next_node

        if current_node != "__end__":
            logger.log_error(f"[Graph] Max steps ({max_steps}) reached without hitting __end__.")

        return state
