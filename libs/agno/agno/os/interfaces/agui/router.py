"""Async router handling exposing an Agno Agent or Team in an AG-UI compatible format."""

import logging
import uuid
from typing import AsyncIterator, Optional

from ag_ui.core import (
    BaseEvent,
    EventType,
    RunAgentInput,
    RunErrorEvent,
    RunStartedEvent,
)
from ag_ui.encoder import EventEncoder
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from agno.agent.agent import Agent
from agno.os.interfaces.agui.utils import (
    async_stream_agno_response_as_agui_events,
    convert_agui_messages_to_agno_messages,
    validate_agui_state,
)
from agno.team.team import Team

from app.agents.router.router_service import AgentRouterService

logger = logging.getLogger(__name__)

async def run_agent(agent: Agent, run_input: RunAgentInput) -> AsyncIterator[BaseEvent]:
    """Run the contextual Agent, mapping AG-UI input messages to Agno format, and streaming the response in AG-UI format."""
    run_id = run_input.run_id or str(uuid.uuid4())

    try:
        # Preparing the input for the Agent and emitting the run started event
        messages = convert_agui_messages_to_agno_messages(run_input.messages or [])

        yield RunStartedEvent(type=EventType.RUN_STARTED, thread_id=run_input.thread_id, run_id=run_id)

        # Look for user_id and file_ids in run_input.forwarded_props
        user_id = None
        use_general_agent =False
        if run_input.forwarded_props and isinstance(run_input.forwarded_props, dict):
            user_id = run_input.forwarded_props.get("user_id")
            file_ids = run_input.forwarded_props.get("file_ids")
            kb_id: Any | None = run_input.forwarded_props.get("kb_id")
            access_token = run_input.forwarded_props.get("access_token")

            from app.utils.request_context import RequestContext

            if file_ids is not None:
                if isinstance(file_ids, list):
                    file_ids = [str(fid) for fid in file_ids]
                RequestContext.set_file_ids(file_ids)

            if kb_id is not None:
                # 确保 kb_id 是整数类型
                kb_id_int = int(kb_id) if not isinstance(kb_id, int) else kb_id
                RequestContext.set_kb_id(kb_id_int)

            if access_token is not None:
                RequestContext.set_access_token(access_token)

            # 如果有 file_ids，在最新的用户消息中补充知识库提示
            if file_ids and messages:
                for message in reversed(messages):
                    if hasattr(message, 'role') and message.role == 'user':
                        original_content = getattr(message, 'content', '')
                        message.content = f"{original_content}\n - 如涉及文档查询类问题，优先通过知识库检索工具作答，仅使用原始问题调用工具（内部规则，不向用户展示），且不额外补充信息。"
                        use_general_agent = True
                        logger.info(f"Added knowledge base hint to user message")
                        break
        if agent is None:
            logger.info(f"Agent {run_id} has no agent")
            # Extract the last user message
            current_question = ""
            for message in reversed(messages):
                if hasattr(message, 'role') and message.role == 'user':
                    current_question = getattr(message, 'content', '')
                    break


            agent_router_service = AgentRouterService()

            agent = await agent_router_service.route_and_create_agent(
                user_id=user_id,
                session_id=run_input.thread_id,
                user_message=current_question,
                use_general_agent=use_general_agent
            )

        # Validating the session state is of the expected type (dict)
        session_state = validate_agui_state(run_input.state, run_input.thread_id)

        # Request streaming response from agent
        response_stream = agent.arun(
            input=messages,
            session_id=run_input.thread_id,
            stream=True,
            stream_events=True,
            user_id=user_id,
            session_state=session_state,
        )

        # Stream the response content in AG-UI format
        async for event in async_stream_agno_response_as_agui_events(
            response_stream=response_stream,  # type: ignore
            thread_id=run_input.thread_id,
            run_id=run_id,
        ):
            yield event

    # Emit a RunErrorEvent if any error occurs
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        yield RunErrorEvent(type=EventType.RUN_ERROR, message=str(e))


async def run_team(team: Team, input: RunAgentInput) -> AsyncIterator[BaseEvent]:
    """Run the contextual Team, mapping AG-UI input messages to Agno format, and streaming the response in AG-UI format."""
    run_id = input.run_id or str(uuid.uuid4())
    try:
        # Extract the last user message for team execution
        messages = convert_agui_messages_to_agno_messages(input.messages or [])
        yield RunStartedEvent(type=EventType.RUN_STARTED, thread_id=input.thread_id, run_id=run_id)

        # Look for user_id in input.forwarded_props
        user_id = None
        if input.forwarded_props and isinstance(input.forwarded_props, dict):
            user_id = input.forwarded_props.get("user_id")

        # Validating the session state is of the expected type (dict)
        session_state = validate_agui_state(input.state, input.thread_id)

        # Request streaming response from team
        response_stream = team.arun(
            input=messages,
            session_id=input.thread_id,
            stream=True,
            stream_steps=True,
            user_id=user_id,
            session_state=session_state,
        )

        # Stream the response content in AG-UI format
        async for event in async_stream_agno_response_as_agui_events(
            response_stream=response_stream, thread_id=input.thread_id, run_id=run_id
        ):
            yield event

    except Exception as e:
        logger.error(f"Error running team: {e}", exc_info=True)
        yield RunErrorEvent(type=EventType.RUN_ERROR, message=str(e))


def attach_routes(router: APIRouter, agent: Optional[Agent] = None, team: Optional[Team] = None) -> APIRouter:
    # if agent is None and team is None:
    #     raise ValueError("Either agent or team must be provided.")

    encoder = EventEncoder()

    @router.post(
        "/agui",
        name="run_agent",
    )
    async def run_agent_agui(run_input: RunAgentInput):
        async def event_generator():
            async for event in run_agent(agent, run_input):
                    encoded_event = encoder.encode(event)
                    yield encoded_event

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            },
        )

    @router.get("/status")
    async def get_status():
        return {"status": "available"}

    return router
