import asyncio

from mcp.server.stdio import stdio_server

from mindlm.mcp.server import server


async def main() -> None:  # pragma: no cover
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
