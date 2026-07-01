#!/bin/bash
# Run the local MCP server in the 'ai' conda environment.
# Use --transport streamable-http for an HTTP-accessible MCP endpoint.
~/miniforge3/bin/conda run -n ai python "/Users/sugandha/Library/CloudStorage/Dropbox/UT Austin/aiw394D deep learning/klein_mcp.py" --transport streamable-http
