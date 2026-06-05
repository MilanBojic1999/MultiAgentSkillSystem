from langchain_mcp_adapters.client import MultiServerMCPClient

MCP_ACCESS_AGENT: dict[str, dict[str, str]] = {
    "researcher": {"yotta_mcp": "http://207.189.105.118:8001/mcp"},
}

def create_mcp_client(agent_name: str) -> tuple:
    """
    Returns (client, tools) for the given agent.
    The caller MUST use `async with client:` around tool usage to keep
    the MCP transport alive for the duration of the agent call.

    Returns (None, []) if the agent has no MCP servers configured.
    """
    server_map = MCP_ACCESS_AGENT.get(agent_name, {})
    if not server_map:
        return None, []

    client = MultiServerMCPClient(
        {name: {"url": url, "transport": "streamable_http"}
         for name, url in server_map.items()}
    )
    return client, client.get_tools()
