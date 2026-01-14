import json

class DocsAgent:
    """
    Mock agent that simulates an MCP-enabled agent navigating Microsoft documentation.
    """
    def process(self, query: str) -> dict:
        # Simulate searching Microsoft Learn docs

        results = [
            {
                "title": "Microsoft Agent Framework Overview",
                "url": "https://learn.microsoft.com/en-us/agent-framework/overview",
                "snippet": "Agent Framework offers two primary categories of capabilities: AI agents and Workflows."
            },
            {
                "title": "Model Context Protocol (MCP) on Azure",
                "url": "https://learn.microsoft.com/en-us/azure/api-management/mcp-server-overview",
                "snippet": "Learn how Azure API Management enables secure, scalable access to remote MCP servers for AI agents."
            }
        ]

        return {
            "intent": "doc_search_results",
            "data": {
                "query": query,
                "source": "Microsoft Learn (Mock MCP)",
                "results": results
            }
        }
