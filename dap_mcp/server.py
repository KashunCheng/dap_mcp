import json
import sys

import anyio
import click
import mcp.types as types
from dap_types import LaunchRequestArguments, SourceBreakpoint
from mcp.server.lowlevel import Server
from pathlib import Path
from pydantic import TypeAdapter
from typing import Optional, TextIO, Literal

from dap_mcp.config import DebuggerSpecificConfig
from dap_mcp.debugger import Debugger, FunctionCallError
from dap_mcp.factory import DAPClientSingletonFactory
from dap_mcp.render import render_xml, try_render

import logging

logger = logging.getLogger(__name__)


@click.command()
@click.option("--port", default=8000, help="Port to listen on for SSE")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Enable verbose logging", default=False
)
@click.option(
    "--config",
    "-c",
    "config_file",
    help="Path to the configuration file",
    required=True,
    type=click.File("r"),
)
def main(
    port: int,
    transport: str,
    verbose: bool,
    config_file: TextIO,
) -> int:
    if verbose:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    debug_config_type_adapter = TypeAdapter(DebuggerSpecificConfig)
    try:
        json_config = json.load(config_file)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON configuration: {e}")
        return 1
    config = debug_config_type_adapter.validate_python(json_config)
    app: Server[object] = Server("dap_mcp")
    dap_factory = DAPClientSingletonFactory(config.debuggerPath, config.debuggerArgs)
    launch_arguments = LaunchRequestArguments(
        noDebug=False,
        **config.model_dump(
            exclude_none=True,
            exclude={"type", "debuggerPath", "debuggerArgs", "sourceDirs", "tools"},
        ),
    )
    debugger = Debugger(dap_factory, launch_arguments, config.tools)

    def ensure_file_path(str_path: str) -> Path | None:
        path = Path(str_path)
        if not path.is_file():
            if path.is_absolute():
                return None
            for potential_dir in config.sourceDirs:
                potential_path = Path(potential_dir) / path
                if potential_path.is_file():
                    return potential_path
            return None
        return path

    async def launch():
        return (await debugger.launch()).render()

    async def set_breakpoint(path: str, line: int, condition: Optional[str] = None):
        file_path = ensure_file_path(path)
        if file_path is None:
            return FunctionCallError(message=f"File ({path}) not found").render()
        response = await debugger.set_breakpoint(file_path, line, condition)
        return try_render(response)

    async def remove_breakpoint(path: str, line: int):
        file_path = ensure_file_path(path)
        if file_path is None:
            return FunctionCallError(message=f"File ({path}) not found").render()
        response = await debugger.remove_breakpoint(file_path, line)
        return try_render(response)

    async def view_file_around_line(line: int, path: Optional[str] = None):
        if path:
            file_path = ensure_file_path(path)
            if file_path is None:
                return FunctionCallError(message=f"File ({path}) not found").render()
        else:
            file_path = None
        response = await debugger.view_file_around_line(file_path, line)
        return try_render(response)

    async def remove_all_breakpoints():
        response = await debugger.remove_all_breakpoints()
        if isinstance(response, FunctionCallError):
            return response.render()
        return "All breakpoints removed"

    async def list_all_breakpoints():
        response = await debugger.list_all_breakpoints()
        if isinstance(response, FunctionCallError):
            return response.render()

        def render_file(file: str, breakpoints: list[SourceBreakpoint]) -> str:
            return render_xml(
                "file",
                [
                    render_xml("breakpoint", None, **sb.model_dump())
                    for sb in breakpoints
                ],
                path=file,
            )

        return render_xml(
            "breakpoints",
            [
                render_file(str(file), breakpoints)
                for file, breakpoints in response.items()
            ],
        )

    async def continue_execution():
        response = await debugger.continue_execution()
        return try_render(response)

    async def step_in():
        response = await debugger.step_in()
        return try_render(response)

    async def step_out():
        response = await debugger.step_out()
        return try_render(response)

    async def next():
        response = await debugger.next()
        return try_render(response)

    async def evaluate(expression: str):
        response = await debugger.evaluate(expression)
        return try_render(response)

    async def change_frame(frameId: int):
        response = await debugger.change_frame(frameId)
        return response.render()

    async def terminate():
        response = await debugger.terminate()
        if isinstance(response, FunctionCallError):
            return response.render()
        return response

    @app.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        if name == "get_launch_config" and config.tools.getLaunchConfig.enabled:
            config_schema = json.dumps(type(config).model_json_schema())
            config_json = json.dumps(config.model_dump(exclude_none=True))
            return [
                types.TextContent(
                    type="text",
                    text=render_xml(
                        "config",
                        [
                            render_xml("schema", config_schema),
                            render_xml("data", config_json),
                        ],
                    ),
                )
            ]
        if name == "launch" and config.tools.launch.enabled:
            return [types.TextContent(type="text", text=await launch())]
        if name == "set_breakpoint" and config.tools.setBreakpoint.enabled:
            return [
                types.TextContent(
                    type="text",
                    text=await set_breakpoint(
                        arguments["path"], arguments["line"], arguments.get("condition")
                    ),
                )
            ]
        if name == "remove_breakpoint" and config.tools.removeBreakpoint.enabled:
            return [
                types.TextContent(
                    type="text",
                    text=await remove_breakpoint(arguments["path"], arguments["line"]),
                )
            ]
        if name == "view_file_around_line" and config.tools.viewFileAroundLine.enabled:
            return [
                types.TextContent(
                    type="text",
                    text=await view_file_around_line(
                        arguments["line"], arguments.get("path")
                    ),
                )
            ]
        if (
            name == "remove_all_breakpoints"
            and config.tools.removeAllBreakpoints.enabled
        ):
            return [types.TextContent(type="text", text=await remove_all_breakpoints())]
        if name == "list_all_breakpoints" and config.tools.listAllBreakpoints.enabled:
            return [types.TextContent(type="text", text=await list_all_breakpoints())]
        if name == "continue_execution" and config.tools.continueExecution.enabled:
            return [types.TextContent(type="text", text=await continue_execution())]
        if name == "step_in" and config.tools.stepIn.enabled:
            return [types.TextContent(type="text", text=await step_in())]
        if name == "step_out" and config.tools.stepOut.enabled:
            return [types.TextContent(type="text", text=await step_out())]
        if name == "next" and config.tools.next.enabled:
            return [types.TextContent(type="text", text=await next())]
        if name == "evaluate" and config.tools.evaluate.enabled:
            return [
                types.TextContent(
                    type="text", text=await evaluate(arguments["expression"])
                )
            ]
        if name == "change_frame" and config.tools.changeFrame.enabled:
            return [
                types.TextContent(
                    type="text", text=await change_frame(arguments["frameId"])
                )
            ]
        if name == "terminate" and config.tools.terminate.enabled:
            return [types.TextContent(type="text", text=await terminate())]
        raise ValueError(f"Unknown tool: {name}")

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        tool_list: list[types.Tool | Literal[False]] = [
            config.tools.getLaunchConfig.enabled
            and types.Tool(
                name="get_launch_config",
                description="Returns the user provided launch configuration along with its detailed schema for a DAP-compatible debugger. The schema includes descriptions for each field.",
                inputSchema={"type": "object", "properties": {}},
            ),
            config.tools.launch.enabled
            and types.Tool(
                name="launch",
                description="Launch the debuggee program. Set breakpoints before launching if necessary.",
                inputSchema={"type": "object", "properties": {}},
            ),
            config.tools.setBreakpoint.enabled
            and types.Tool(
                name="set_breakpoint",
                description="Set a breakpoint at the specified file and line with an optional condition.",
                inputSchema={
                    "type": "object",
                    "required": ["path", "line"],
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The file path where the breakpoint should be set.",
                        },
                        "line": {
                            "type": "integer",
                            "description": "The line number at which to set the breakpoint.",
                        },
                        "condition": {
                            "type": "string",
                            "description": "Optional condition to trigger the breakpoint.",
                        },
                    },
                },
            ),
            config.tools.removeBreakpoint.enabled
            and types.Tool(
                name="remove_breakpoint",
                description="Remove a breakpoint from the specified file and line.",
                inputSchema={
                    "type": "object",
                    "required": ["path", "line"],
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The file path from which to remove the breakpoint.",
                        },
                        "line": {
                            "type": "integer",
                            "description": "The line number of the breakpoint to remove.",
                        },
                    },
                },
            ),
            config.tools.listAllBreakpoints.enabled
            and types.Tool(
                name="list_all_breakpoints",
                description="List all breakpoints currently set in the debugger.",
                inputSchema={"type": "object", "properties": {}},
            ),
            config.tools.removeAllBreakpoints.enabled
            and types.Tool(
                name="remove_all_breakpoints",
                description="Remove all breakpoints currently set in the debugger.",
                inputSchema={"type": "object", "properties": {}},
            ),
            config.tools.continueExecution.enabled
            and types.Tool(
                name="continue_execution",
                description="Continue execution in the debugger after hitting a breakpoint.",
                inputSchema={"type": "object", "properties": {}},
            ),
            config.tools.stepIn.enabled
            and types.Tool(
                name="step_in",
                description="Step into the function call in the debugger.",
                inputSchema={"type": "object", "properties": {}},
            ),
            config.tools.stepOut.enabled
            and types.Tool(
                name="step_out",
                description="Step out of the current function in the debugger.",
                inputSchema={"type": "object", "properties": {}},
            ),
            config.tools.next.enabled
            and types.Tool(
                name="next",
                description="Step over to the next line of code in the debugger.",
                inputSchema={"type": "object", "properties": {}},
            ),
            config.tools.evaluate.enabled
            and types.Tool(
                name="evaluate",
                description="Evaluate an expression in the current debugging context.",
                inputSchema={
                    "type": "object",
                    "required": ["expression"],
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The expression to evaluate.",
                        }
                    },
                },
            ),
            config.tools.changeFrame.enabled
            and types.Tool(
                name="change_frame",
                description="Change the current debugging frame to the specified frame ID.",
                inputSchema={
                    "type": "object",
                    "required": ["frameId"],
                    "properties": {
                        "frameId": {
                            "type": "integer",
                            "description": "The ID of the frame to switch to.",
                        }
                    },
                },
            ),
            config.tools.terminate.enabled
            and types.Tool(
                name="terminate",
                description="Terminate the current debugging session.",
                inputSchema={"type": "object", "properties": {}},
            ),
            config.tools.viewFileAroundLine.enabled
            and types.Tool(
                name="view_file_around_line",
                description="Returns the lines of source code and the source code around the specified line. You should ALWAYS prefer this tool if you are reading code. Because it will show the line number, which is crucial for debugging. If 'path' is provided, it opens that file; otherwise, it uses the last specified file. You must provide a file to read the source before launch as no prefilled paths exist.",
                inputSchema={
                    "type": "object",
                    "required": ["line"],
                    "properties": {
                        "line": {
                            "type": "integer",
                            "description": "The line number around which the source code will be displayed.",
                        },
                        "path": {
                            "type": "string",
                            "description": "Optional file path. Provide this to open a specific file; otherwise, the last specified file is used.",
                        },
                    },
                },
            ),
        ]
        return [tool for tool in tool_list if tool]

    if transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")
        debugger_inited = False

        async def handle_sse(request):
            nonlocal debugger_inited
            if not debugger_inited:
                await debugger.initialize()
                debugger_inited = True
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        import uvicorn

        uvicorn.run(starlette_app, host="0.0.0.0", port=port)
    else:
        from mcp.server.stdio import stdio_server

        async def arun():
            await debugger.initialize()
            async with stdio_server() as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        anyio.run(arun)

    return 0


if __name__ == "__main__":
    main()
