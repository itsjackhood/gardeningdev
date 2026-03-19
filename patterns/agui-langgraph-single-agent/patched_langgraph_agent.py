"""Patched LangGraphAgent subclass with two bug fixes.

Bug 1 — Multi-turn silent failure:
    The original prepare_stream triggers prepare_regenerate_stream when
    len(checkpointed) > len(incoming). Since the frontend sends only the new
    user message (len=1), this fires on every turn after the first and the
    regenerate path produces nothing (it can't find the fresh UUID message ID
    in checkpoint history). Fix: remove that block entirely — the checkpointer
    already has history, so we just append the new message and stream normally.

Bug 2 — Parallel tool calls:
    The original _handle_single_event only processes tool_call_chunks[0], so
    the second parallel tool never gets TOOL_CALL_START/ARGS/END events and
    the frontend silently drops its result. Fix: emit TOOL_CALL_START for every
    chunk that has a name; args and results come through OnToolEnd as normal.
"""

import json
from typing import Any, AsyncGenerator

from ag_ui.core import (
    EventType,
    CustomEvent,
    RunStartedEvent,
    RunFinishedEvent,
    ReasoningEncryptedValueEvent,
    ReasoningMessageEndEvent,
    ReasoningEndEvent,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    ToolCallStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
)
from ag_ui.core.types import RunAgentInput
from ag_ui_langgraph import LangGraphAgent
from ag_ui_langgraph.agent import State, dump_json_safe
from ag_ui_langgraph.types import LangGraphEventTypes
from ag_ui_langgraph.utils import (
    agui_messages_to_langchain,
    get_stream_payload_input,
    resolve_reasoning_content,
    resolve_encrypted_reasoning_content,
    resolve_message_content,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command


class PatchedLangGraphAgent(LangGraphAgent):

    async def prepare_stream(self, input: RunAgentInput, agent_state: State, config: RunnableConfig):
        state_input = input.state or {}
        messages = input.messages or []
        forwarded_props = input.forwarded_props or {}
        thread_id = input.thread_id

        state_input["messages"] = agent_state.values.get("messages", [])
        self.active_run["current_graph_state"] = agent_state.values.copy()
        langchain_messages = agui_messages_to_langchain(messages)
        state = self.langgraph_default_merge_state(state_input, langchain_messages, input)
        self.active_run["current_graph_state"].update(state)
        config["configurable"]["thread_id"] = thread_id
        interrupts = agent_state.tasks[0].interrupts if agent_state.tasks and len(agent_state.tasks) > 0 else []
        has_active_interrupts = len(interrupts) > 0
        resume_input = forwarded_props.get('command', {}).get('resume', None)

        self.active_run["schema_keys"] = self.get_schema_keys(config)

        # BUG FIX 1: regenerate block removed — see module docstring.

        events_to_dispatch = []
        if has_active_interrupts and not resume_input:
            events_to_dispatch.append(
                RunStartedEvent(type=EventType.RUN_STARTED, thread_id=thread_id, run_id=self.active_run["id"])
            )
            for interrupt in interrupts:
                events_to_dispatch.append(
                    CustomEvent(
                        type=EventType.CUSTOM,
                        name=LangGraphEventTypes.OnInterrupt.value,
                        value=dump_json_safe(interrupt.value),
                        raw_event=interrupt,
                    )
                )
            events_to_dispatch.append(
                RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=self.active_run["id"])
            )
            return {
                "stream": None,
                "state": None,
                "config": None,
                "events_to_dispatch": events_to_dispatch,
            }

        if self.active_run["mode"] == "continue":
            await self.graph.aupdate_state(config, state, as_node=self.active_run.get("node_name"))

        if resume_input:
            if isinstance(resume_input, str):
                try:
                    resume_input = json.loads(resume_input)
                except json.JSONDecodeError:
                    pass
            stream_input = Command(resume=resume_input)
        else:
            payload_input = get_stream_payload_input(
                mode=self.active_run["mode"],
                state=state,
                schema_keys=self.active_run["schema_keys"],
            )
            stream_input = {**forwarded_props, **payload_input} if payload_input else None

        subgraphs_stream_enabled = input.forwarded_props.get('stream_subgraphs') if input.forwarded_props else False

        kwargs = self.get_stream_kwargs(
            input=stream_input,
            config=config,
            subgraphs=bool(subgraphs_stream_enabled),
            version="v2",
        )

        stream = self.graph.astream_events(**kwargs)
        return {"stream": stream, "state": state, "config": config}

    async def _handle_single_event(self, event: Any, state: State) -> AsyncGenerator[str, None]:
        event_type = event.get("event")

        if event_type != LangGraphEventTypes.OnChatModelStream:
            # All non-stream events handled by parent unchanged
            async for ev in super()._handle_single_event(event, state):
                yield ev
            return

        # --- OnChatModelStream ---
        should_emit_messages = event["metadata"].get("emit-messages", True)
        should_emit_tool_calls = event["metadata"].get("emit-tool-calls", True)

        if event["data"]["chunk"].response_metadata.get('finish_reason', None):
            return

        all_tool_chunks = event["data"]["chunk"].tool_call_chunks or []

        # BUG FIX 2: parallel tool call support.
        # When the model calls N tools in parallel, each OnChatModelStream event
        # contains N chunks (one per tool), each with its own index and id.
        # The original code only looks at chunks[0], so tools 2..N never get
        # TOOL_CALL_START/ARGS/END events.
        #
        # Fix: maintain a per-index tracker in active_run["parallel_tool_streams"]
        # so each parallel tool streams independently.
        if all_tool_chunks:
            self.active_run["has_function_streaming"] = True
            if "parallel_tool_streams" not in self.active_run:
                self.active_run["parallel_tool_streams"] = {}

            parallel = self.active_run["parallel_tool_streams"]

            for chunk in all_tool_chunks:
                idx = chunk.get("index", 0)
                tool_id = chunk.get("id")
                tool_name = chunk.get("name")
                args_delta = chunk.get("args", "")

                if tool_name and tool_id:
                    # New tool starting — emit TOOL_CALL_START
                    parallel[idx] = {"id": tool_id, "name": tool_name}
                    if should_emit_tool_calls:
                        yield self._dispatch_event(
                            ToolCallStartEvent(
                                type=EventType.TOOL_CALL_START,
                                tool_call_id=tool_id,
                                tool_call_name=tool_name,
                                parent_message_id=event["data"]["chunk"].id,
                                raw_event=event,
                            )
                        )
                elif idx in parallel and args_delta:
                    # Args streaming for an already-started tool
                    if should_emit_tool_calls:
                        yield self._dispatch_event(
                            ToolCallArgsEvent(
                                type=EventType.TOOL_CALL_ARGS,
                                tool_call_id=parallel[idx]["id"],
                                delta=args_delta,
                                raw_event=event,
                            )
                        )

            # Keep messages_in_process in sync with chunk[0] so the parent's
            # OnChatModelEnd handler can emit TOOL_CALL_END for the primary tool.
            first = all_tool_chunks[0]
            first_idx = first.get("index", 0)
            if first_idx in parallel:
                self.set_message_in_progress(
                    self.active_run["id"],
                    {"id": event["data"]["chunk"].id, "tool_call_id": parallel[first_idx]["id"], "tool_call_name": parallel[first_idx]["name"]}
                )
            return

        # --- No tool chunks: text message or end-of-tool-stream ---
        current_stream = self.get_message_in_progress(self.active_run["id"])
        has_current_stream = bool(current_stream and current_stream.get("id"))

        reasoning_data = resolve_reasoning_content(event["data"]["chunk"]) if event["data"]["chunk"] else None
        encrypted_reasoning_data = resolve_encrypted_reasoning_content(event["data"]["chunk"]) if event["data"]["chunk"] else None
        message_content = resolve_message_content(event["data"]["chunk"].content) if event["data"]["chunk"] and event["data"]["chunk"].content else None
        is_message_content_event = message_content is not None
        is_tool_call_end_event = has_current_stream and current_stream.get("tool_call_id")
        is_message_end_event = has_current_stream and not current_stream.get("tool_call_id") and not is_message_content_event

        if reasoning_data:
            for event_str in self.handle_reasoning_event(reasoning_data):
                yield event_str
            return

        if encrypted_reasoning_data and self.active_run.get('reasoning_process') is not None:
            reasoning_message_id = self.active_run["reasoning_process"]["message_id"]
            yield self._dispatch_event(
                ReasoningEncryptedValueEvent(
                    type=EventType.REASONING_ENCRYPTED_VALUE,
                    subtype="message",
                    entity_id=reasoning_message_id,
                    encrypted_value=encrypted_reasoning_data,
                )
            )
            return

        if reasoning_data is None and self.active_run.get('reasoning_process') is not None:
            reasoning_message_id = self.active_run["reasoning_process"]["message_id"]
            if self.active_run["reasoning_process"].get("signature"):
                yield self._dispatch_event(
                    ReasoningEncryptedValueEvent(
                        type=EventType.REASONING_ENCRYPTED_VALUE,
                        subtype="message",
                        entity_id=reasoning_message_id,
                        encrypted_value=self.active_run["reasoning_process"]["signature"],
                    )
                )
            yield self._dispatch_event(ReasoningMessageEndEvent(type=EventType.REASONING_MESSAGE_END, message_id=reasoning_message_id))
            yield self._dispatch_event(ReasoningEndEvent(type=EventType.REASONING_END, message_id=reasoning_message_id))
            self.active_run["reasoning_process"] = None

        if is_tool_call_end_event:
            # Emit TOOL_CALL_END for all parallel tools that were tracked
            parallel = self.active_run.get("parallel_tool_streams", {})
            if parallel:
                for tracked in parallel.values():
                    yield self._dispatch_event(
                        ToolCallEndEvent(type=EventType.TOOL_CALL_END, tool_call_id=tracked["id"], raw_event=event)
                    )
                self.active_run["parallel_tool_streams"] = {}
            else:
                yield self._dispatch_event(
                    ToolCallEndEvent(type=EventType.TOOL_CALL_END, tool_call_id=current_stream["tool_call_id"], raw_event=event)
                )
            self.messages_in_process[self.active_run["id"]] = None
            return

        if is_message_end_event:
            yield self._dispatch_event(
                TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=current_stream["id"], raw_event=event)
            )
            self.messages_in_process[self.active_run["id"]] = None
            return

        if is_message_content_event and should_emit_messages:
            if not (current_stream and current_stream.get("id")):
                yield self._dispatch_event(
                    TextMessageStartEvent(
                        type=EventType.TEXT_MESSAGE_START,
                        role="assistant",
                        message_id=event["data"]["chunk"].id,
                        raw_event=event,
                    )
                )
                self.set_message_in_progress(
                    self.active_run["id"],
                    {"id": event["data"]["chunk"].id, "tool_call_id": None, "tool_call_name": None}
                )
                current_stream = self.get_message_in_progress(self.active_run["id"])

            yield self._dispatch_event(
                TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=current_stream["id"],
                    delta=message_content,
                    raw_event=event,
                )
            )
