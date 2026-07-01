import argparse
import logging
import shutil
import subprocess
import sys
from mcp.server.fastmcp import FastMCP

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def generate_image(prompt: str) -> str:
    """Use this tool to generate an image using the local FLUX.2 Klein model."""
    logger.debug("Received prompt: %s", prompt)
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        error_message = "Ollama executable not found in PATH. Install Ollama or add it to PATH."
        logger.error(error_message)
        return f"Error generating image: {error_message}"

    try:
        logger.debug("Starting subprocess call to Ollama (%s)...", ollama_path)
        result = subprocess.run(
            [ollama_path, "run", "x/flux2-klein:4b", prompt],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        logger.debug("Successfully received response from Ollama.")
        return f"Success! The image was generated. Output details: {result.stdout.strip()}"
    except subprocess.CalledProcessError as e:
        logger.error("Ollama command failed. Stdout: %s, Stderr: %s", e.stdout, e.stderr)
        return f"Error generating image (Ollama error): {e.stderr or e.stdout or str(e)}"
    except subprocess.TimeoutExpired:
        logger.error("Ollama generation timed out after 300 seconds.")
        return "Error: The generation took longer than 5 minutes and timed out."
    except Exception as e:
        logger.exception("An unexpected Python error occurred.")
        return f"System Error: {e}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local MCP server exposing a FLUX.2 Klein image generation tool."
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="streamable-http",
        help="MCP transport protocol to use. Defaults to streamable-http.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8888, help="Port to listen on.")
    parser.add_argument(
        "--mount-path",
        default="/",
        help="Mount path for SSE or streamable-http transport.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(
        "Starting MCP server with transport=%s host=%s port=%s mount_path=%s",
        args.transport,
        args.host,
        args.port,
        args.mount_path,
    )
    mcp = FastMCP(
        "flux-klein",
        log_level="DEBUG",
        host=args.host,
        port=args.port,
        mount_path=args.mount_path,
    )
    mcp.tool()(generate_image)

    if args.transport == "stdio":
        print("Starting MCP server on stdio transport. Waiting for MCP client input...", file=sys.stderr)
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        print(
            f"Starting MCP server on SSE transport at http://{args.host}:{args.port}{args.mount_path}",
            file=sys.stderr,
        )
        mcp.run(transport="sse", mount_path=args.mount_path)
    else:
        print(
            f"Starting MCP server on streamable-http transport at http://{args.host}:{args.port}{args.mount_path}",
            file=sys.stderr,
        )
        mcp.run(transport="streamable-http", mount_path=args.mount_path)


if __name__ == "__main__":
    main()
