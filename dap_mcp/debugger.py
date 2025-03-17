import logging

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
    StepInRequest,
    StepInArguments,
    StepOutRequest,
    StepOutArguments,
    NextRequest,
    NextArguments,
)
from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel
from typing import Literal, Optional, List, Tuple, Callable, Any

from dap_mcp.config import DAPToolsConfig
from dap_mcp.dap import DAPClient
from dap_mcp.factory import DAPFactory
from dap_mcp.render import render_table, render_scope, render_xml, try_dump_base_model


class FunctionCallError(BaseModel):
    message: str

    def render(self) -> str:
        return render_xml("error", self.message)


@dataclass
class EventListView:
    events: List[Event]

    def render(self) -> str:
        return render_xml(
            "events",
            [
                render_xml("event", try_dump_base_model(event.body), type=event.event)
                for event in self.events
            ],
        )


@dataclass
class SourceCodeView:
    source: Path | None
    # the line is in the center of the code view
    source_center_line: int
    # the line has an arrow pointing to it (active line)
    source_active_line: Optional[int]

    def get_source(self) -> Tuple[List[Tuple[int, str]], int]:
        if self.source is None:
            return [], 0
        with open(self.source, "r") as f:
            lines = f.readlines()
        span = 20
        self.source_center_line = min(len(lines), self.source_center_line)
        span_begin = max(1, self.source_center_line - span)
        span_end = min(len(lines), self.source_center_line + span)
        source = list(
            zip(range(span_begin, span_end + 1), lines[span_begin - 1 : span_end])
        )
        if source:
            source[-1] = (source[-1][0], source[-1][1].rstrip())
        return source, len(lines)

    def render(self) -> str:
        source, line_count = self.get_source()
        source_view_content = (
            "!!!source not available!!!"
            if source == []
            else render_table(self.source_active_line, source, line_delimiter="")
        )
        return render_xml(
            "source",
            source_view_content,
            path=str(self.source.resolve()) if self.source else None,
            total_lines=line_count,
        )


@dataclass
class StoppedDebuggerView:
    source: SourceCodeView
    frames: List[StackFrame]
    frame_active_id: int
    threads: List[Thread]
    thread_active_id: int
    # module_map: dict[int, Module]
    variables: List[Tuple[Scope, List[Variable]]]
    events: EventListView
    exception_info: Optional[ExceptionInfoResponseBody]
    render_variables: bool = True
    max_variable_expr_length: Optional[int] = None

    def render(self) -> str:
        return render_xml(
            "debugger_view",
            [self.source.render()]
            + (
                [
                    render_xml(
                        "variables",
                        [
                            render_scope(
                                scope, variables, self.max_variable_expr_length
                            )
                            for scope, variables in self.variables
                        ],
                    )
                ]
                if self.render_variables
                else []
            )
            + [
                render_xml(
                    "frames",
                    render_table(
                        self.frame_active_id,
                        [
                            (
                                frame.id,
                                f"{frame.name}\tpath={frame.source.path}\tline={frame.line}\tcol={frame.column}",
                            )
                            for frame in self.frames
                        ],
                    ),
                ),
                render_xml(
                    "threads",
                    render_table(
                        self.thread_active_id,
                        [(thread.id, thread.name) for thread in self.threads],
                    ),
                ),
                self.events.render(),
            ]
            + (
                [
                    render_xml(
                        "exception_info",
                        f"{self.exception_info.exceptionId} {self.exception_info.description}",
                    )
                ]
                if self.exception_info
                else []
            ),
        )


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
        launch_arguments: LaunchRequestArguments,
        config_or_none: DAPToolsConfig | None = None,
    ):
        if config_or_none is None:
            self._config = DAPToolsConfig()
        else:
            self._config = config_or_none
        self.state: DebuggerState = "before_initialization"
        self._factory = factory
        self._client: Optional[DAPClient] = None
        self.breakpoints: dict[Path, list[SourceBreakpoint]] = {}
        self.launch_arguments = launch_arguments
        self.launch_request: Optional[LaunchRequest] = None
        self.modules: dict[int, Module] = {}
        self.variables: dict[int, Variable] = {}
        self.events: list[Event] = []
        self.frames: dict[int, StackFrame] = {}
        self.active_frame_id: Optional[int] = None
        self.active_thread_id: Optional[int] = None
        self.alternative_center_line_to_view: Optional[int] = None
        self.alternative_file_to_view: Optional[Path] = None

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
            async def wrapper(self, *args, **kwargs):
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
                logging.debug(f"Debugger state changed from {self.state} to stopped")
                self.state = "stopped"
                if event.body.reason == "exception":
                    self.exception_raised = True
            elif isinstance(event, TerminatedEvent):
                logging.debug(
                    f"Debugger state changed from {self.state} to before_initialization"
                )
                self.state = "before_initialization"
                await self._terminate()
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
            logging.debug(f"Debugger state changed from {self.state} to launched")
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
        logging.debug(f"Debugger state changed from {self.state} to before_launch")
        self.state = "before_launch"
        self.launch_request = await self._send_launch_request()
        return response

    async def _update_breakpoints(
        self, path: Path
    ) -> SetBreakpointsResponse | ErrorResponse:
        breakpoints = self.breakpoints.get(path, [])
        request = SetBreakpointsRequest(
            seq=0,
            type="request",
            command="setBreakpoints",
            arguments=SetBreakpointsArguments(
                source=Source(
                    name=path.name,
                    path=str(path.resolve()),
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
        assert self.active_thread_id is not None, (
            "Active thread is not set, this should be set via Events"
        )
        stack_trace = await self._get_stack_trace(self.active_thread_id)
        assert self.active_frame_id is not None, (
            "Active frame is not set, this should be set via Events"
        )
        if self._config.debuggerView.showVariables:
            scopes = await self._get_scopes(self.active_frame_id)
            variables = [
                (
                    scope,
                    (
                        await self._get_variables(scope.variablesReference)
                    ).body.variables,
                )
                for scope in scopes.body.scopes
            ]
        else:
            variables = []
        if self.state == "errored":
            exception_info = (await self._get_exception_info()).body
        else:
            exception_info = None
        current_frame = next(
            f for f in stack_trace.body.stackFrames if f.id == self.active_frame_id
        )
        return StoppedDebuggerView(
            source=SourceCodeView(
                source=self.alternative_file_to_view or Path(current_frame.source.path),
                source_center_line=self.alternative_center_line_to_view
                or current_frame.line,
                source_active_line=current_frame.line,
            ),
            frames=stack_trace.body.stackFrames,
            frame_active_id=self.active_frame_id,
            threads=threads.body.threads,
            thread_active_id=self.active_thread_id,
            variables=variables,
            events=EventListView(events=self._pop_events()),
            exception_info=exception_info,
            render_variables=self._config.debuggerView.showVariables,
        )

    @available_states("set breakpoint", ["before_launch", "stopped", "errored"])
    async def set_breakpoint(
        self, path: Path, line: int, condition: Optional[str] = None
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
        self, path: Path, line: int
    ) -> FunctionCallError | SetBreakpointsResponse | ErrorResponse:
        breakpoints = self.breakpoints.get(path, [])
        breakpoints = [b for b in breakpoints if b.line != line]
        self.breakpoints[path] = breakpoints
        return await self._update_breakpoints(path)

    @available_states("list all breakpoints", ["before_launch", "stopped", "errored"])
    async def list_all_breakpoints(
        self,
    ) -> dict[Path, list[SourceBreakpoint]] | FunctionCallError:
        return self.breakpoints

    @available_states("remove all breakpoints", ["before_launch", "stopped", "errored"])
    async def remove_all_breakpoints(
        self,
    ) -> None | FunctionCallError:
        for path in self.breakpoints:
            await self._update_breakpoints(path)
        self.breakpoints = {}
        return None

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
        self.alternative_center_line_to_view = None
        self.alternative_file_to_view = None
        await self._wait_for_event_types({"stopped", "terminated"})
        if self.state != "stopped":
            return EventListView(events=self._pop_events())
        return await self._get_stopped_debugger_view()

    @available_states("step in", ["stopped"])
    async def step_in(
        self,
    ) -> FunctionCallError | StoppedDebuggerView | EventListView:
        request = StepInRequest(
            seq=0,
            type="request",
            command="stepIn",
            arguments=StepInArguments(threadId=self.active_thread_id),
        )
        await self._send_request(request)
        response = await self._wait_for_request(request)
        if isinstance(response, ErrorResponse):
            raise RuntimeError("Error handing for step in is not implemented")
        self.alternative_center_line_to_view = None
        self.alternative_file_to_view = None
        await self._wait_for_event_types({"stopped", "terminated"})
        if self.state != "stopped":
            return EventListView(events=self._pop_events())
        return await self._get_stopped_debugger_view()

    @available_states("step out", ["stopped"])
    async def step_out(
        self,
    ) -> FunctionCallError | StoppedDebuggerView | EventListView:
        request = StepOutRequest(
            seq=0,
            type="request",
            command="stepOut",
            arguments=StepOutArguments(threadId=self.active_thread_id),
        )
        await self._send_request(request)
        response = await self._wait_for_request(request)
        if isinstance(response, ErrorResponse):
            raise RuntimeError("Error handing for step out is not implemented")
        self.alternative_center_line_to_view = None
        self.alternative_file_to_view = None
        await self._wait_for_event_types({"stopped", "terminated"})
        if self.state != "stopped":
            return EventListView(events=self._pop_events())
        return await self._get_stopped_debugger_view()

    @available_states("next", ["stopped"])
    async def next(
        self,
    ) -> FunctionCallError | StoppedDebuggerView | EventListView:
        request = NextRequest(
            seq=0,
            type="request",
            command="next",
            arguments=NextArguments(threadId=self.active_thread_id),
        )
        await self._send_request(request)
        response = await self._wait_for_request(request)
        if isinstance(response, ErrorResponse):
            raise RuntimeError("Error handing for next is not implemented")
        self.alternative_center_line_to_view = None
        self.alternative_file_to_view = None
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
        configure_done_response: (
            ErrorResponse | ConfigurationDoneRequest
        ) = await self._wait_for_request(configure_done_request)
        if isinstance(configure_done_response, ErrorResponse):
            raise RuntimeError(
                "Error handing for configuration done is not implemented"
            )
        launch_response: ErrorResponse | LaunchResponse = await self._wait_for_request(
            self.launch_request
        )
        if isinstance(launch_response, ErrorResponse):
            raise RuntimeError("Error handing for launch is not implemented")
        self.alternative_center_line_to_view = None
        self.alternative_file_to_view = None
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
        self.alternative_center_line_to_view = None
        self.alternative_file_to_view = None
        return await self._get_stopped_debugger_view()

    @available_states(
        "view file around line",
        ["before_initialization", "before_launch", "launched", "errored", "stopped"],
    )
    async def view_file_around_line(
        self, file: Optional[Path], line: int
    ) -> SourceCodeView | FunctionCallError:
        if (
            file is None
            and self.alternative_file_to_view is None
            and self.state in ["before_initialization", "before_launch", "launched"]
        ):
            return FunctionCallError(
                message=f"You must provide a file path to view in {self.state} state"
            )
        self.alternative_center_line_to_view = line
        if file:
            self.alternative_file_to_view = file
        current_active_line = (
            self.frames[self.active_frame_id].line if self.active_frame_id else None
        )
        source = SourceCodeView(
            source=self.alternative_file_to_view,
            source_center_line=line,
            source_active_line=current_active_line,
        )
        return source

    @available_states("terminate", ["stopped", "errored"])
    async def terminate(self) -> str | FunctionCallError:
        return await self._terminate()

    async def _terminate(self) -> str:
        logging.debug(
            f"terminating debugger... self._client is not None: {self._client is not None}"
        )
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
        self.alternative_center_line_to_view = None
        self.alternative_file_to_view = None
        logging.debug(
            f"Debugger initialized, state before initialization: {self.state}"
        )
        await self.initialize()
        logging.debug(f"Debugger initialized, state after initialization: {self.state}")
        return "Debugger terminated"
