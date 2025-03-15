import json
import pydantic_core
from asyncio import StreamReader, StreamWriter
from dap_types import Request, Response, Event, DiscriminatedProtocolMessage
from pydantic import TypeAdapter, ValidationError, Field
from typing import Tuple, Union, Annotated
import logging

logger = logging.getLogger(__name__)


class DAPClient:
    def __init__(self, stream_reader: StreamReader, stream_writer: StreamWriter):
        self.seq = 1
        self.request_sent: dict[int, Request] = {}
        self.response_received: dict[int, Response] = {}
        self.stream_reader = stream_reader
        self.stream_writer = stream_writer
        self.protocol_message_adapter = TypeAdapter(DiscriminatedProtocolMessage)
        self.lossy_protocol_message_adapter: TypeAdapter[
            Annotated[Union[Request, Event, Response], Field(discriminator="type")]
        ] = TypeAdapter(
            Annotated[Union[Request, Event, Response], Field(discriminator="type")]
        )

    async def send_request(self, request: Request):
        request.seq = self.seq
        self.request_sent[self.seq] = request
        self.seq += 1

        encoded = json.dumps(
            pydantic_core.to_jsonable_python(request, exclude_none=True)
        ).encode("utf-8")
        headers = f"Content-Length: {len(encoded)}\r\n\r\n".encode("utf-8")
        message = headers + encoded

        self.stream_writer.write(message)
        await self.stream_writer.drain()
        logger.debug(f"client -> server: {encoded.decode('utf-8')}")

    async def receive(self) -> Request | Response | Event:
        while True:
            content_length_bytes = await self.stream_reader.readuntil(b"\r\n\r\n")
            assert content_length_bytes is not None, "Connection closed"
            content_length = int(content_length_bytes.decode("utf-8").split(": ")[1])
            message = await self.stream_reader.readexactly(content_length)
            logger.debug(f"server -> client: {message.decode('utf-8')}")
            message = json.loads(message)
            try:
                protocol_message: Request | Response | Event = (
                    self.protocol_message_adapter.validate_python(message)
                )
                return protocol_message
            except ValidationError:
                try:
                    lossy_protocol_message: Request | Response | Event = (
                        self.lossy_protocol_message_adapter.validate_python(message)
                    )
                    return lossy_protocol_message
                except ValidationError:
                    logger.warning(f"Invalid message: {message.decode('utf-8')}")

    @staticmethod
    def _try_discriminate_response(request: Request, response: Response) -> Response:
        if hasattr(request, "discriminate_response"):
            return request.discriminate_response(response)
        return response

    async def wait_for_request(self, request: Request) -> Tuple[Response, list[Event]]:
        if request.seq in self.response_received:
            response = self.response_received.pop(request.seq)
            return self._try_discriminate_response(request, response), []

        events: list[Event] = []
        while True:
            message = await self.receive()
            if isinstance(message, Response):
                if message.request_seq == request.seq:
                    return self._try_discriminate_response(request, message), events
                else:
                    self.response_received[message.seq] = message
            elif isinstance(message, Event):
                events.append(message)
            else:
                logger.warning(f"Unexpected message: {message}")

    async def wait_for_event_types(self, event_types: set[str]) -> list[Event]:
        events = []
        while True:
            message = await self.receive()
            if isinstance(message, Event):
                events.append(message)
                if message.event in event_types:
                    return events
            elif isinstance(message, Response):
                self.response_received[message.seq] = message
            else:
                logger.warning(f"Unexpected message: {message}")
