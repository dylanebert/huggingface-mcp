import os
from typing import Literal

from fastmcp import FastMCP
from huggingface_hub import HfApi

mcp = FastMCP("Hugging Face MCP")

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")
hf_api = HfApi(token=HF_TOKEN)


@mcp.tool()
def list_models(
    sort: Literal[
        "trending_score", "last_modified", "created_at", "downloads", "likes"
    ] = "trending_score",
    direction: Literal[-1, 1] = -1,
    limit: int = 100,
) -> list[str]:
    """
    List models on Hugging Face.

    Args:
        sort: The key with which to sort the models. Possible values are:
            - "trending_score" (sorted by recent popularity)
            - "last_modified" (sorted by last modified date)
            - "created_at" (sorted by creation date)
            - "downloads" (sorted by number of downloads)
            - "likes" (sorted by number of likes)
        direction: The direction to sort the models. Possible values are:
            - -1 (descending)
            - 1 (ascending)
        limit: The maximum number of models to list

    Returns:
        A list of model ids
    """
    try:
        models = hf_api.list_models(sort=sort, direction=int(direction), limit=limit)
        return [model.modelId for model in models]
    except Exception as e:
        return [f"Error: {e}"]


if __name__ == "__main__":
    mcp.run(
        transport="sse", host="127.0.0.1", port=8000, log_level="debug", path="/mcp"
    )
