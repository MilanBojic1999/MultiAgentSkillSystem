from langchain_mcp_adapters.client import MultiServerMCPClient

MCP_ACCESS_AGENT: dict[str, dict[str, str]] = {
    "researcher": {"yotta_mcp": "http://207.189.105.118:8001/mcp"},
}

async def get_mcp_tools_for_agent(agent_name: str) -> list:
    """
    Returns activated MCP tools scoped to this agent.
    Call inside an async context and pass the tools to create_react_agent.
    """
    server_map = MCP_ACCESS_AGENT.get(agent_name, [])
    if not server_map:
        return []

    client = MultiServerMCPClient(
        {name: {"url": url, "transport": "streamable_http"}
         for name, url in server_map.items()}
    )
    async with client:
        return client.get_tools()
