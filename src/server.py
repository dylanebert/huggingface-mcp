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
    search: str = None,
    library: list[str] = None,
    tags: list[str] = None,
    pipeline_tag: str = None,
    sort: Literal[
        "trending_score", "last_modified", "created_at", "downloads", "likes"
    ] = "trending_score",
    direction: Literal[-1, 1] = -1,
    limit: int = 20,
) -> list[str]:
    """
     List models on Hugging Face Hub.

    Use this tool to search for models by name, tags, or other filters, and to get a list of model IDs.
    This is the first step when you need to find a specific model before retrieving its details.

    Parameters:
        search (str, optional): A string to search for in model IDs or names (e.g., "deepseek").
        library (list[str], optional): List of libraries the models use (e.g., ["pytorch", "tensorflow"]).
        tags (list[str], optional): List of tags to filter models by (e.g., ["text-generation", "llama"]).
        pipeline_tag (str, optional): Filter by pipeline tag (e.g., "text-generation").
        sort (Literal["trending_score", "last_modified", "created_at", "downloads", "likes"], default="trending_score"): Sort models by the specified key.
        direction (int, default=-1): Sort direction: -1 for descending, 1 for ascending.
        limit (int, default=20): Maximum number of models to return.

    Returns:
        list[str]: A list of model IDs matching the search criteria.

    Examples:
        - To find trending models: list_models(sort="trending_score", limit=10)
        - To search for models related to "deepseek": list_models(search="deepseek", sort="likes", limit=5)
        - To filter by tag: list_models(tags=["text-generation"], pipeline_tag="text-generation")
    """
    try:
        models = hf_api.list_models(
            library=library,
            tags=tags,
            search=search,
            pipeline_tag=pipeline_tag,
            sort=sort,
            direction=int(direction),
            limit=limit,
        )
        return [model.modelId for model in models]
    except Exception as e:
        return [f"Error: {e}"]


@mcp.tool()
def get_model_info(model_id: str) -> dict:
    """
    Get detailed information about a specific model on Hugging Face Hub.

    This tool requires the exact model ID, which can be obtained using `list_models`.
    If you have a partial name or tag, use `list_models` first to find the exact ID.

    Parameters:
        model_id (str): The exact model ID in the format "organization/model-name" (e.g., "DeepSeek/DeepSeek-R1").

    Returns:
        dict: A dictionary containing model information including available fields such as:
            - id: The model ID
            - author: The author of the model
            - created_at: The creation date
            - last_modified: The last modified date
            - downloads: Number of downloads
            - likes: Number of likes
            - tags: List of tags
            - pipeline_tag: The pipeline tag
            - library_name: The library name
            - license: The model license
            - base_model: The base model (if available)
            - siblings: List of repository files (if available)
            - datasets: Datasets used to train the model (if available)
            - spaces: List of spaces using this model (if available)
            - xet_enabled: Whether XET is enabled (if available)

    Raises:
        Exception: If the model_id is invalid or not found. Use list_models to find the correct ID.

    Example:
        - First, find the model ID: list_models(search="deepseek", sort="likes", limit=1)
        - Then, get the model info: get_model_info("DeepSeek/DeepSeek-R1")
    """
    try:
        model = hf_api.model_info(model_id)
        model_info = {}

        if hasattr(model, "id") and model.id is not None:
            model_info["id"] = model.id

        if hasattr(model, "author") and model.author is not None:
            model_info["author"] = model.author

        if hasattr(model, "created_at") and model.created_at is not None:
            model_info["created_at"] = model.created_at

        if hasattr(model, "last_modified") and model.last_modified is not None:
            model_info["last_modified"] = model.last_modified

        if hasattr(model, "downloads") and model.downloads is not None:
            model_info["downloads"] = model.downloads

        if hasattr(model, "likes") and model.likes is not None:
            model_info["likes"] = model.likes

        if hasattr(model, "tags") and model.tags is not None:
            model_info["tags"] = model.tags

        if hasattr(model, "pipeline_tag") and model.pipeline_tag is not None:
            model_info["pipeline_tag"] = model.pipeline_tag

        if hasattr(model, "library_name") and model.library_name is not None:
            model_info["library_name"] = model.library_name

        if hasattr(model, "card_data") and model.card_data is not None:
            if (
                hasattr(model.card_data, "license")
                and model.card_data.license is not None
            ):
                model_info["license"] = model.card_data.license

            if (
                hasattr(model.card_data, "base_model")
                and model.card_data.base_model is not None
            ):
                model_info["base_model"] = model.card_data.base_model

            if (
                hasattr(model.card_data, "datasets")
                and model.card_data.datasets is not None
            ):
                model_info["datasets"] = model.card_data.datasets

        if hasattr(model, "siblings") and model.siblings is not None:
            model_info["siblings"] = model.siblings

        if hasattr(model, "spaces") and model.spaces is not None:
            model_info["spaces"] = model.spaces

        if hasattr(model, "xet_enabled") and model.xet_enabled is not None:
            model_info["xet_enabled"] = model.xet_enabled

        return model_info
    except Exception:
        return {
            "error": f"Failed to get model info for '{model_id}'. Use list_models to find the exact ID."
        }


if __name__ == "__main__":
    mcp.run(
        transport="sse", host="127.0.0.1", port=8000, log_level="debug", path="/mcp"
    )
