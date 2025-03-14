import sys

import anyio
import click
import mcp.types as types
from dap_types import LaunchRequestArguments, Response, SourceBreakpoint, ErrorResponse
from mcp.server.lowlevel import Server
from typing import Optional

from dap_mcp.debugger import Debugger, RenderableContent
from dap_mcp.factory import DAPClientSingletonFactory
from dap_mcp.render import render_xml

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename="example.log", encoding="utf-8", level=logging.DEBUG)


def render_response(r: Response) -> str:
    if isinstance(r, ErrorResponse):
        return f"<error>{r}</error>"
    return f"<response>{r}</response>"


def try_render(r: Response | RenderableContent) -> str:
    if isinstance(r, Response):
        return render_response(r)
    else:
        return r.render()


@click.command()
@click.option("--port", default=8000, help="Port to listen on for SSE")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
@click.option("--debuggee-cwd", help="Working directory of the debuggee", required=True)
@click.option(
    "--debuggee-python", help="Python executable to use", default=sys.executable
)
@click.option(
    "--debug-program-path", help="Path to the program to debug", required=True
)
def main(
    port: int,
    transport: str,
    debuggee_cwd: str,
    debuggee_python: str,
    debug_program_path: str,
) -> int:
    # find_active_loop()
    app: Server[object] = Server("mcp-website-fetcher")
    dap_factory = DAPClientSingletonFactory("python -m debugpy.adapter")
    launch_arguments = LaunchRequestArguments(
        noDebug=False,
        program=debug_program_path,
        python=[debuggee_python],
        args=[],
        cwd=debuggee_cwd,
        env={"PYTHONIOENCODING": "UTF-8", "PYTHONUNBUFFERED": "1"},
    )
    # find_active_loop()
    debugger = Debugger(dap_factory, debuggee_cwd, launch_arguments)

    # find_active_loop()

    async def launch():
        return (await debugger.launch()).render()

    async def set_breakpoint(path: str, line: int, condition: Optional[str] = None):
        response = await debugger.set_breakpoint(path, line, condition)
        return try_render(response)

    async def remove_breakpoint(path: str, line: int):
        response = await debugger.remove_breakpoint(path, line)
        return try_render(response)

    async def list_all_breakpoints():
        response = await debugger.list_all_breakpoints()

        def render_file(file: str, breakpoints: list[SourceBreakpoint]) -> str:
            return f"""<file path="{file}">
{"\n".join([render_xml("breakpoint", None, **sb.model_dump()) for sb in breakpoints])}
</file>"""

        return f"""<breakpoints>
{"\n".join([render_file(file, breakpoints) for file, breakpoints in response.items()])}
</breakpoints>
"""

    async def continue_execution():
        response = await debugger.continue_execution()
        return try_render(response)

    async def evaluate(expression: str):
        response = await debugger.evaluate(expression)
        return try_render(response)

    async def change_frame(frameId: int):
        response = await debugger.change_frame(frameId)
        return response.render()

    async def terminate():
        response = await debugger.terminate()
        return response

    @app.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        # find_active_loop()
        if name == "launch":
            return [types.TextContent(type="text", text=await launch())]
        if name == "set_breakpoint":
            return [
                types.TextContent(
                    type="text",
                    text=await set_breakpoint(
                        arguments["path"], arguments["line"], arguments.get("condition")
                    ),
                )
            ]
        if name == "remove_breakpoint":
            return [
                types.TextContent(
                    type="text",
                    text=await remove_breakpoint(arguments["path"], arguments["line"]),
                )
            ]
        if name == "list_all_breakpoints":
            return [types.TextContent(type="text", text=await list_all_breakpoints())]
        if name == "continue_execution":
            return [types.TextContent(type="text", text=await continue_execution())]
        if name == "evaluate":
            return [
                types.TextContent(
                    type="text", text=await evaluate(arguments["expression"])
                )
            ]
        if name == "change_frame":
            return [
                types.TextContent(
                    type="text", text=await change_frame(arguments["frameId"])
                )
            ]
        if name == "terminate":
            return [types.TextContent(type="text", text=await terminate())]
        raise ValueError(f"Unknown tool: {name}")

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="launch",
                description="Launch the debuggee program.",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
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
            types.Tool(
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
            types.Tool(
                name="list_all_breakpoints",
                description="List all breakpoints currently set in the debugger.",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="continue_execution",
                description="Continue execution in the debugger after hitting a breakpoint.",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
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
            types.Tool(
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
            types.Tool(
                name="terminate",
                description="Terminate the current debugging session.",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

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
