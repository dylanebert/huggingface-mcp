# Hugging Face MCP

Hugging Face MCP is an MCP server on top of the [Hugging Face Hub API](https://huggingface.co/docs/huggingface_hub/v0.31.2/package_reference/hf_api).

## Tools

-   **search_models**: Search models on the Hugging Face Hub. Equivalent to `hf_api.list_models`.
-   **get_model_info**: Get structured metadata about a model on the Hugging Face Hub. Equivalent to `hf_api.model_info`.
-   **get_model_card**: Uses `hf_api.hf_hub_download` to get the full model card (README.md) of a model on the Hugging Face Hub.
-   **update_metadata**: Opens a pull request using `hf_api.create_commit` to update the metadata of a model on the Hugging Face Hub. Currently only supports updating `library_name` and `pipeline_tag`.

## Usage

This MCP is tested using Cursor.

### Run the Server

1. Clone the repository.
2. Create a virtual environment and install the dependencies.

```bash
uv venv
.venv\Scripts\activate # Windows
source .venv/bin/activate # Linux/MacOS
uv pip install fastmcp huggingface_hub
```

3. Run the MCP server.

```bash
python src/server.py
```

### Use the Server in Cursor

1. Press `Ctrl+Shift+J` -> `MCP` -> `Add new global MCP server` to open `mcp.json`.
2. Add the following configuration:

```json
{
    "mcpServers": {
        "huggingface": {
            "url": "http://127.0.0.1:8000/mcp"
        }
    }
}
```

3. The server should now be available in the MCP server list, with the tools `search_models`, `get_model_info`, `get_model_card`, and `update_metadata`.
4. Use Cursor Chat with a frontier model like `claude-3.7-sonnet` and Agent mode to use the tools.

## Examples

-   _List the top 50 trending models on Hugging Face_
-   _Get the model card for the `meta-llama/Meta-Llama-3-8B-Instruct` model_
-   _Find the top trending models with `gguf` in the name and write them to a CSV called `gguf_trending_models.csv` with the header `model_id,pipeline_tag`._
-   _Read the data in `gguf_trending_models.csv`. Use that data to update the `pipeline_tag` for all models in `gguf_trending_models.csv` using the `update_metadata` tool._

## Limitations

-   Agents make mistakes, especially on complex operations. Break it down into simple tasks with manual verification.
-   The Hugging Face Hub API has rate limits. Avoid massive-scale operations.
