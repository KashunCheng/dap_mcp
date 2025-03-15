import asyncio.subprocess

from asyncio.subprocess import Process
from typing import Protocol, Optional

from dap_mcp.dap import DAPClient


class DAPFactory(Protocol):
    async def create_instance(self) -> DAPClient: ...

    async def destroy_instance(self, client: DAPClient): ...


class DAPClientSingletonFactory:
    def __init__(self, cmd: str, args: list[str], **kwargs):
        self.cmd = cmd
        self.args = args
        self.kwargs = kwargs
        self.debugger_process: Optional[Process] = None

    async def create_instance(self) -> DAPClient:
        if self.debugger_process is not None:
            raise Exception(
                "DAPClientSingletonFactory can only create one instance of DAPClient"
            )
        adapter = await asyncio.subprocess.create_subprocess_exec(
            self.cmd,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            **self.kwargs,
        )
        reader = adapter.stdout
        writer = adapter.stdin
        assert reader is not None, "Connection closed"
        assert writer is not None, "Connection closed"
        self.debugger_process = adapter
        return DAPClient(reader, writer)

    async def destroy_instance(self, client: DAPClient):
        if self.debugger_process is None:
            return
        if client.stream_reader != self.debugger_process.stdout:
            raise Exception("Client does not belong to this factory")
        self.debugger_process.terminate()
        self.debugger_process = None
