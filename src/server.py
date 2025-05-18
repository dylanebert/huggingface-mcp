import os
from typing import Literal
import re

from fastmcp import FastMCP
from huggingface_hub import HfApi, CommitOperationAdd

mcp = FastMCP("Hugging Face MCP")

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")
hf_api = HfApi(token=HF_TOKEN)


@mcp.tool()
def search_models(
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
    Search models on Hugging Face Hub.

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
        - To find trending models: search_models(sort="trending_score", limit=10)
        - To search for models related to "deepseek": search_models(search="deepseek", sort="likes", limit=5)
        - To filter by tag: search_models(tags=["text-generation"], pipeline_tag="text-generation")
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
    Get structured metadata about a model on the Hugging Face Hub.

    Use this when you need specific fields like downloads, tags, or other metadata.
    For comprehensive model information, use `get_model_card`.

    This tool requires the exact model ID, which can be obtained using `search_models`.
    If you have a partial name or tag, use `search_models` first to find the exact ID.

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
        Exception: If the model_id is invalid or not found. Use search_models to find the correct ID.

    Example:
        - First, find the model ID: search_models(search="deepseek", sort="likes", limit=1)
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
    except Exception as e:
        return {"error": f"Failed to get model info for '{model_id}': {e}"}


@mcp.tool()
def get_model_card(model_id: str) -> str:
    """
    Get the complete model card (README.md) for a specific model on Hugging Face Hub.

    Use this when you need comprehensive model documentation including usage examples, model limitations, etc.
    For only structured metadata, use `get_model_info` instead.

    This tool requires the exact model ID, which can be obtained using `search_models`.
    If you have a partial name or tag, use `search_models` first to find the exact ID.

    Args:
        model_id (str): The model ID in the format "organization/model-name" (e.g., "DeepSeek/DeepSeek-R1").

    Returns:
        str: The markdown content of the model card.

    Example:
        - First, find the model ID: search_models(search="deepseek", sort="likes", limit=1)
        - Then, get the model card: get_model_card("DeepSeek/DeepSeek-R1")
    """
    try:
        filepath = hf_api.hf_hub_download(model_id, "README.md")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        return content
    except Exception as e:
        return {"error": f"Failed to get model card for '{model_id}': {e}"}


@mcp.tool()
def update_metadata(
    model_id: str,
    pipeline_tag: str = None,
    library_name: str = None,
) -> str:
    """
    Create a pull request to modify specific metadata fields in the model card (README.md) for a model on Hugging Face Hub.

    This tool only modifies the pipeline_tag and library_name fields in the YAML header of the README.
    It will create a pull request with a fixed commit message "Update Metadata".

    Args:
        model_id (str): The model ID in the format "organization/model-name" (e.g., "DeepSeek/DeepSeek-R1").
        pipeline_tag (str, optional): The pipeline tag to set (e.g., "text-generation").
        library_name (str, optional): The library name to set (e.g., "pytorch").

    Returns:
        str: A message indicating whether the pull request was created successfully.
    """
    if not pipeline_tag and not library_name:
        return "No changes requested. At least one of pipeline_tag or library_name must be provided."

    try:
        filepath = hf_api.hf_hub_download(model_id, "README.md")

        with open(filepath, "rb") as f:
            content_bytes = f.read()

        content = content_bytes.decode("utf-8")
        line_ending = "\r\n" if "\r\n" in content else "\n"

        has_yaml_header = re.match(r"^---\s*[\r\n]", content) is not None

        if not has_yaml_header:
            header = "---" + line_ending
            if pipeline_tag is not None:
                header += f"pipeline_tag: {pipeline_tag}{line_ending}"
            if library_name is not None:
                header += f"library_name: {library_name}{line_ending}"
            header += "---" + line_ending

            content = header + content
        else:
            header_match = re.match(r"^---(.*?)---\s*", content, re.DOTALL)
            if header_match:
                header = header_match.group(1)
                rest_of_content = content[header_match.end() :]

                if pipeline_tag is not None:
                    if re.search(r"^\s*pipeline_tag:", header, re.MULTILINE):
                        header = re.sub(
                            r"(^\s*pipeline_tag:).*?(\r?\n)",
                            f"\\1 {pipeline_tag}\\2",
                            header,
                            flags=re.MULTILINE,
                        )
                    else:
                        header += f"pipeline_tag: {pipeline_tag}{line_ending}"

                if library_name is not None:
                    if re.search(r"^\s*library_name:", header, re.MULTILINE):
                        header = re.sub(
                            r"(^\s*library_name:).*?(\r?\n)",
                            f"\\1 {library_name}\\2",
                            header,
                            flags=re.MULTILINE,
                        )
                    else:
                        header += f"library_name: {library_name}{line_ending}"

                content = f"---{header}---{line_ending}{rest_of_content}"
            else:
                return "Malformed YAML header structure in README."

        with open(filepath, "wb") as f:
            f.write(content.encode("utf-8"))

        operation = CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=filepath,
        )

        hf_api.create_commit(
            repo_id=model_id,
            commit_message="Update Metadata",
            commit_description="",
            operations=[operation],
            create_pr=True,
        )

        return f"Pull request created successfully for '{model_id}'"
    except Exception as e:
        return f"Failed to create pull request for '{model_id}': {e}"


if __name__ == "__main__":
    mcp.run(
        transport="sse", host="127.0.0.1", port=8000, log_level="debug", path="/mcp"
    )
