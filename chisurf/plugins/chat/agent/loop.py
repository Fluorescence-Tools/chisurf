import time
from typing import Any, Dict, List, Optional, Tuple

from .tools import parse_toolcall_block, execute_tool


class AgentLoop:
    def __init__(self, max_steps: int = 4):
        self.max_steps = max_steps

    def run(
        self,
        goal: str,
        llm_respond,
        confirm_destructive: bool = False,
        extra_whitelist: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        context_provider=None,
    ) -> Dict[str, Any]:
        """
        Planner–executor loop constrained to at most one tool call per step,
        with a maximum number of steps. Returns a trace of steps.
        llm_respond(prompt_messages) -> assistant_text
        context_provider(query) -> retrieved_context_str
        """
        trace: List[Dict[str, Any]] = []
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": f"Goal: {goal}"})

        for step in range(1, self.max_steps + 1):
            # Retrieve context for the current goal (simple approach)
            context = ""
            if callable(context_provider):
                try:
                    context = context_provider(goal)
                except Exception:
                    context = ""
            planning_hint = "You may choose and call at most one tool using the exact fenced JSON toolcall block."
            plan_prompt = f"Context:\n{context}\n\n{planning_hint}\n\nRespond with a short plan followed by optional tool call if needed."
            messages.append({"role": "user", "content": plan_prompt})
            assistant = llm_respond(messages)
            tool_name_args = parse_toolcall_block(assistant)

            step_rec: Dict[str, Any] = {
                "step": step,
                "assistant": assistant,
            }

            if tool_name_args is None:
                # No tool call: assume conclusion if assistant says done
                trace.append(step_rec)
                if step == self.max_steps:
                    break
                # Ask the LLM to conclude
                messages.append({"role": "assistant", "content": assistant})
                messages.append({"role": "user", "content": "If task is complete, say 'Concluding.' Otherwise emit a tool call."})
                continue

            tool_name, args = tool_name_args
            step_rec["tool"] = {"name": tool_name, "args": args}

            # Execute tool (respect destructive confirmation)
            result, err = execute_tool(tool_name, args, confirm_destructive=confirm_destructive, extra_whitelist=extra_whitelist)
            if err:
                observation = {"error": err}
            else:
                observation = {"result": result}
            step_rec["observation"] = observation
            trace.append(step_rec)

            # Feed observation back
            obs_text = str(observation)[:4000]
            messages.append({"role": "assistant", "content": assistant})
            messages.append({"role": "user", "content": f"Observation: {obs_text}\n\nIf the goal is complete, conclude briefly. Otherwise, plan next and call at most one tool."})

            # Heuristic: if tool executed successfully and not destructive confirmation error, allow next step
            if err and "requires confirmation" in err.lower():
                break

        return {"trace": trace, "steps": len(trace), "concluded": len(trace) >= 1 and len(trace) <= self.max_steps}
