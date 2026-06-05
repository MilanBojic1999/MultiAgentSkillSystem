from tools import calculate, plotting_tool, run_bash

AGENT_TOOLS = {
    "mathematician": [calculate, plotting_tool, run_bash],
    "researcher": [run_bash],
    "writer": [],
}