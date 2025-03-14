import logging

import os
from dap_types import (
    SetBreakpointsResponse,
    ErrorResponse,
    SetBreakpointsRequest,
    SetBreakpointsArguments,
    Source,
    SourceBreakpoint,
    ContinueRequest,
    ContinueArguments,
    ContinueResponse,
    EvaluateRequest,
    EvaluateArguments,
    EvaluateResponse,
    LaunchRequest,
    LaunchRequestArguments,
    InitializeRequest,
    InitializeRequestArguments,
    InitializeResponse,
    LaunchResponse,
    StoppedEvent,
    TerminatedEvent,
    Event,
    ConfigurationDoneRequest,
    StackTraceRequest,
    StackTraceArguments,
    ThreadsResponse,
    Thread,
    StackFrame,
    ThreadsRequest,
    StackTraceResponse,
    ModuleEvent,
    Module,
    Request,
    Response,
    ScopesResponse,
    ScopesRequest,
    ScopesArguments,
    VariablesRequest,
    VariablesArguments,
    VariablesResponse,
    Scope,
    Variable,
    ExceptionInfoRequest,
    ExceptionInfoArguments,
    ExceptionInfoResponse,
    ExceptionInfoResponseBody,
    SetExceptionBreakpointsRequest,
    SetExceptionBreakpointsArguments,
    SetExceptionBreakpointsResponse,
)
from dataclasses import dataclass
from pydantic import BaseModel
from typing import Literal, Optional, List, Tuple, Protocol, Callable, Any, Self

from dap_mcp.dap import DAPClient
from dap_mcp.factory import DAPFactory


class RenderableContent(Protocol):
    def render(self) -> str: ...


class FunctionCallError(BaseModel):
    message: str

    def render(self) -> str:
        return f"<error>{self.message}</error>"


@dataclass
class EventListView:
    events: List[Event]

    def render(self) -> str:
        return f"""<events>
{"\n".join([str(event) for event in self.events])}
</events>"""


def render_variable(variable: Variable) -> str:
    return f'<variable name="{variable.name}" type="{variable.type}" variablesReference={variable.variablesReference}>{variable.value}</variable>'


def render_scope(scope: Scope, variables: List[Variable]) -> str:
    return f"""<scope name={scope.name}>
{"\n".join([render_variable(variable) for variable in variables])}
</scope>"""


@dataclass
class StoppedDebuggerView:
    source: str
    source_first_line: int
    source_active_line: Optional[int]
    frames: List[StackFrame]
    frame_active_id: int
    threads: List[Thread]
    thread_active_id: int
    # module_map: dict[int, Module]
    variables: List[Tuple[Scope, List[Variable]]]
    events: EventListView
    exception_info: Optional[ExceptionInfoResponseBody]

    @staticmethod
    def render_table(active_id: Optional[int], lines: List[Tuple[int, str]]) -> str:
        # print lines with active line marked
        # active_id is the index of the active line in lines.
        # lines: [(line_number, line_content)].
        # If active_id equals to the index of the line in lines, that line is active.
        # It will print in the following format:
        # {line_number} -> {line_content}
        # {line_number}    {line_content} (otherwise)
        if len(lines) == 0:
            return ""
        max_line_number = max([line_number for line_number, _ in lines])
        max_line_number_length = len(str(max_line_number))
        formatted_lines = []
        for line_number, line_content in lines:
            if line_number == active_id:
                formatted_lines.append(
                    f"{line_number:>{max_line_number_length}} -> {line_content}"
                )
            else:
                formatted_lines.append(
                    f"{line_number:>{max_line_number_length}}    {line_content}"
                )
        return "\n".join(formatted_lines)

    def render(self) -> str:
        return f"""<debugger_view>
<source>
{self.render_table(self.source_active_line, [(i + self.source_first_line, line) for i, line in enumerate(self.source.splitlines())])}
</source>
<variables>
{"\n".join([render_scope(scope, variables) for scope, variables in self.variables])}
</variables>
<frames>
{self.render_table(self.frame_active_id, [(frame.id, f"{frame.name}\t{frame.source} line={frame.line} col={frame.column}") for frame in self.frames])}
</frames>
<threads>
{self.render_table(self.thread_active_id, [(thread.id, thread.name) for thread in self.threads])}
</threads>
{self.events.render()}
{f"<exception_info>{self.exception_info}</exception_info>" if self.exception_info else ""}
</debugger_view>
"""


DebuggerState = Literal[
    "before_initialization", "before_launch", "launched", "errored", "stopped"
]


class MethodWithAvailableStates:
    available_states: List[DebuggerState]

    async def __call__(self, *arg, **kwargs) -> Any: ...


class Debugger:
    # launched is an internal state that is not exposed to the LLM
    def __init__(
        self,
        factory: DAPFactory,
        project_root: str,
        launch_arguments: LaunchRequestArguments,
    ):
        self.state: DebuggerState = "before_initialization"
        self._factory = factory
        self._client: Optional[DAPClient] = None
        self.project_root = project_root
        self.breakpoints: dict[str, list[SourceBreakpoint]] = {}
        self.launch_arguments = launch_arguments
        self.launch_request: Optional[LaunchRequest] = None
        self.modules: dict[int, Module] = {}
        self.variables: dict[int, Variable] = {}
        self.events: list[Event] = []
        self.frames: dict[int, StackFrame] = {}
        self.active_frame_id: Optional[int] = None
        self.active_thread_id: Optional[int] = None

    def _get_available_actions(self) -> List[str]:
        func_names: List[str] = []
        for func_name in dir(self):
            func = getattr(self, func_name)
            if hasattr(func, "available_states") and isinstance(
                func.available_states, list
            ):
                if self.state in func.available_states:
                    func_names.append(func_name)
        return func_names

    @staticmethod
    def available_states(operation: str, states: List[DebuggerState]):
        def decorator(function: Callable[..., Any]):
            async def wrapper(self: Self, *args, **kwargs):
                if self.state not in states:
                    return FunctionCallError(
                        message=f"Cannot {operation} in {self.state} state\nAvailable actions for {self.state}: {self._get_available_actions()}."
                    )
                return await function(self, *args, **kwargs)

            wrapper: MethodWithAvailableStates  # type: ignore[no-redef]
            wrapper.available_states = states  # type: ignore
            return wrapper

        return decorator

    async def _handle_events(self, events: list[Event]):
        for event in events:
            if isinstance(event, ModuleEvent):
                if event.body.reason != "remove":
                    self.modules[event.body.module.id] = event.body.module
                else:
                    self.modules.pop(event.body.module.id, None)
            if isinstance(event, StoppedEvent):
                self.active_thread_id = event.body.threadId
                self.state = "stopped"
                if event.body.reason == "exception":
                    self.exception_raised = True
            elif isinstance(event, TerminatedEvent):
                self.state = "before_launch"
                await self.terminate()
        self.events += events

    def _pop_events(self) -> list[Event]:
        events = self.events
        self.events = []
        return events

    async def _send_request(self, request: Request):
        assert self._client is not None, "Client is not initialized"
        return await self._client.send_request(request)

    async def _wait_for_request(self, request: Request) -> Response | ErrorResponse:
        assert self._client is not None, "Client is not initialized"
        response, events = await self._client.wait_for_request(request)
        await self._handle_events(events)
        if isinstance(response, StackTraceResponse):
            self.active_frame_id = response.body.stackFrames[0].id
        elif isinstance(response, StackTraceResponse):
            for frame in response.body.stackFrames:
                self.frames[frame.id] = frame
        elif isinstance(response, LaunchResponse) or isinstance(
            response, ContinueResponse
        ):
            self.active_frame_id = None
            self.active_thread_id = None
            self.frames = {}
            self.state = "launched"
        return response

    async def _wait_for_event_types(self, event_types: set[str]):
        assert self._client is not None, "Client is not initialized"
        events = await self._client.wait_for_event_types(event_types)
        await self._handle_events(events)

    async def initialize(self) -> InitializeResponse:
        self._client = await self._factory.create_instance()
        request = InitializeRequest(
            seq=0,
            type="request",
            command="initialize",
            arguments=InitializeRequestArguments(
                clientID="dap_mcp",
                clientName="dap_mcp",
                adapterID="debugpy",
                locale="en",
            ),
        )
        await self._send_request(request)
        response: ErrorResponse | InitializeResponse = await self._wait_for_request(
            request
        )
        if isinstance(response, ErrorResponse):
            logging.fatal(f"Failed to initialize debugger: {response.message}")
            raise RuntimeError("Failed to initialize debugger")
        self.state = "before_launch"
        self.launch_request = await self._send_launch_request()
        return response

    async def _update_breakpoints(
        self, path: str
    ) -> SetBreakpointsResponse | ErrorResponse:
        breakpoints = self.breakpoints.get(path, [])
        name = os.path.basename(path)
        path = os.path.join(self.project_root, path)
        request = SetBreakpointsRequest(
            seq=0,
            type="request",
            command="setBreakpoints",
            arguments=SetBreakpointsArguments(
                source=Source(
                    name=name,
                    path=path,
                ),
                breakpoints=breakpoints,
            ),
        )
        await self._send_request(request)
        response: ErrorResponse | SetBreakpointsResponse = await self._wait_for_request(
            request
        )
        return response

    async def _get_threads(self) -> ThreadsResponse:
        if self.state != "stopped":
            raise RuntimeError(f"Cannot get threads in {self.state} state")
        request = ThreadsRequest(seq=0, type="request", command="threads")
        await self._send_request(request)
        response: ThreadsResponse | ErrorResponse = await self._wait_for_request(
            request
        )
        if isinstance(response, ErrorResponse):
            raise RuntimeError("Error handing for threads is not implemented")
        return response

    async def _get_stack_trace(self, threadId: int) -> StackTraceResponse:
        if self.state != "stopped":
            raise RuntimeError(f"Cannot get stack trace in {self.state} state")
        request = StackTraceRequest(
            seq=0,
            type="request",
            command="stackTrace",
            arguments=StackTraceArguments(
                threadId=threadId,
                startFrame=0,
            ),
        )
        await self._send_request(request)
        response: ErrorResponse | StackTraceResponse = await self._wait_for_request(
            request
        )
        if isinstance(response, ErrorResponse):
            raise RuntimeError("Error handing for stack trace is not implemented")
        return response

    async def _get_scopes(self, frameId: int) -> ScopesResponse:
        request = ScopesRequest(
            seq=0,
            type="request",
            command="scopes",
            arguments=ScopesArguments(frameId=frameId),
        )
        await self._send_request(request)
        response: ErrorResponse | ScopesResponse = await self._wait_for_request(request)
        if isinstance(response, ErrorResponse):
            raise RuntimeError("Error handing for scopes is not implemented")
        return response

    async def _get_variables(self, variablesReference: int) -> VariablesResponse:
        request = VariablesRequest(
            seq=0,
            type="request",
            command="variables",
            arguments=VariablesArguments(variablesReference=variablesReference),
        )
        await self._send_request(request)
        response: ErrorResponse | VariablesResponse = await self._wait_for_request(
            request
        )
        if isinstance(response, ErrorResponse):
            raise RuntimeError("Error handing for variables is not implemented")
        for variable in response.body.variables:
            if variable.variablesReference != 0:
                self.variables[variable.variablesReference] = variable
        return response

    async def _get_exception_info(self) -> ExceptionInfoResponse:
        request = ExceptionInfoRequest(
            seq=0,
            type="request",
            command="exceptionInfo",
            arguments=ExceptionInfoArguments(threadId=self.active_thread_id),
        )
        await self._send_request(request)
        response: ErrorResponse | ExceptionInfoResponse = await self._wait_for_request(
            request
        )
        if isinstance(response, ErrorResponse):
            raise RuntimeError("Error handing for exception info is not implemented")
        return response

    async def _get_stopped_debugger_view(self):
        threads = await self._get_threads()
        stack_trace = await self._get_stack_trace(self.active_thread_id)
        scopes = await self._get_scopes(self.active_frame_id)
        variables = [
            (
                scope,
                (await self._get_variables(scope.variablesReference)).body.variables,
            )
            for scope in scopes.body.scopes
        ]
        if self.state == "errored":
            exception_info = (await self._get_exception_info()).body
        else:
            exception_info = None
        return StoppedDebuggerView(
            source="",
            source_first_line=0,
            source_active_line=None,
            frames=stack_trace.body.stackFrames,
            frame_active_id=self.active_frame_id,
            threads=threads.body.threads,
            thread_active_id=self.active_thread_id,
            variables=variables,
            events=EventListView(events=self._pop_events()),
            exception_info=exception_info,
        )

    @available_states("set breakpoint", ["before_launch", "stopped", "errored"])
    async def set_breakpoint(
        self, path: str, line: int, condition: Optional[str] = None
    ) -> FunctionCallError | SetBreakpointsResponse | ErrorResponse:
        breakpoints = self.breakpoints.get(path, [])
        for b in breakpoints:
            if b.condition == condition and b.line == line:
                return FunctionCallError(message="Breakpoint already exists")
        breakpoints.append(SourceBreakpoint(line=line, condition=condition))
        self.breakpoints[path] = breakpoints
        return await self._update_breakpoints(path)

    @available_states("remove breakpoint", ["before_launch", "stopped", "errored"])
    async def remove_breakpoint(
        self, path: str, line: int
    ) -> FunctionCallError | SetBreakpointsResponse | ErrorResponse:
        breakpoints = self.breakpoints.get(path, [])
        breakpoints = [b for b in breakpoints if b.line != line]
        self.breakpoints[path] = breakpoints
        return await self._update_breakpoints(path)

    @available_states("list all breakpoints", ["before_launch", "stopped", "errored"])
    async def list_all_breakpoints(self) -> dict[str, list[SourceBreakpoint]]:
        return self.breakpoints

    @available_states("continue execution", ["stopped"])
    async def continue_execution(
        self,
    ) -> FunctionCallError | StoppedDebuggerView | EventListView:
        request = ContinueRequest(
            seq=0,
            type="request",
            command="continue",
            arguments=ContinueArguments(threadId=1),
        )
        await self._send_request(request)
        response = await self._wait_for_request(request)
        if isinstance(response, ErrorResponse):
            raise RuntimeError("Error handing for continue is not implemented")
        await self._wait_for_event_types({"stopped", "terminated"})
        if self.state != "stopped":
            return EventListView(events=self._pop_events())
        return await self._get_stopped_debugger_view()

    @available_states("evaluate an expression", ["stopped"])
    async def evaluate(
        self, expression: str
    ) -> FunctionCallError | EvaluateResponse | ErrorResponse:
        request = EvaluateRequest(
            seq=0,
            type="request",
            command="evaluate",
            arguments=EvaluateArguments(
                expression=expression, frameId=self.active_frame_id
            ),
        )
        await self._send_request(request)
        response: ErrorResponse | EvaluateResponse = await self._wait_for_request(
            request
        )
        return response

    async def _send_launch_request(self) -> LaunchRequest:
        if self.state != "before_launch":
            raise RuntimeError(f"Cannot launch in {self.state} state")
        request = LaunchRequest(
            seq=0, type="request", command="launch", arguments=self.launch_arguments
        )
        await self._send_request(request)
        return request

    @available_states("launch", ["before_launch"])
    async def launch(self) -> FunctionCallError | StoppedDebuggerView | EventListView:
        for path in self.breakpoints:
            await self._update_breakpoints(path)
        set_exception_breakpoints_request = SetExceptionBreakpointsRequest(
            seq=0,
            type="request",
            command="setExceptionBreakpoints",
            arguments=SetExceptionBreakpointsArguments(filters=["uncaught"]),
        )
        await self._send_request(set_exception_breakpoints_request)
        set_exception_breakpoints_response: (
            SetExceptionBreakpointsResponse | ErrorResponse
        ) = await self._wait_for_request(set_exception_breakpoints_request)
        if isinstance(set_exception_breakpoints_response, ErrorResponse):
            raise RuntimeError(
                "Error handing for set exception breakpoints is not implemented"
            )
        configure_done_request = ConfigurationDoneRequest(
            seq=0, type="request", command="configurationDone"
        )
        await self._send_request(configure_done_request)
        response: (
            ErrorResponse | ConfigurationDoneRequest
        ) = await self._wait_for_request(configure_done_request)
        if isinstance(response, ErrorResponse):
            raise RuntimeError(
                "Error handing for configuration done is not implemented"
            )
        response: ErrorResponse | LaunchResponse = await self._wait_for_request(
            self.launch_request
        )  # type: ignore
        if isinstance(response, ErrorResponse):
            raise RuntimeError("Error handing for launch is not implemented")
        await self._wait_for_event_types({"stopped", "terminated"})
        if self.state != "stopped":
            return EventListView(events=self._pop_events())
        return await self._get_stopped_debugger_view()

    @available_states("change frame", ["stopped", "errored"])
    async def change_frame(
        self, frame_id: int
    ) -> FunctionCallError | StoppedDebuggerView:
        if frame_id not in self.frames:
            return FunctionCallError(
                message=f"Frame {frame_id} not found, available frames: {list(self.frames.keys())}"
            )
        self.active_frame_id = frame_id
        return await self._get_stopped_debugger_view()

    @available_states("terminate", ["stopped", "errored"])
    async def terminate(self) -> str:
        if self._client:
            await self._factory.destroy_instance(self._client)
            self._client = None
        # Keep breakpoints
        # self.breakpoints = {}
        self.launch_request = None
        self.modules = {}
        self.variables = {}
        self.events = []
        self.frames = {}
        self.active_frame_id = None
        self.active_thread_id = None
        self.exception_raised = False
        await self.initialize()
        return "Debugger terminated"
