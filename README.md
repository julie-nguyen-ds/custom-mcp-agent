# custom-mcp-agent

## Overview
This repository provides tools and code to build agents using the LangGraph framework on Databricks. Agents created with this repo can be deployed to Databricks and evaluated for quality. 
The system integrates with an MCP server, which is deployed using the [`custom-mcp-to-dbx`](https://github.com/julie-nguyen-ds/custom-mcp-to-dbx) repository for agent orchestration and communication.

## Features
- **Create LangGraph Agents**: Define and configure autonomous agents for workflow automation within Databricks[1].
- **Easy Deployment**: Deploy agents singly or in batches to Databricks workspaces.
- **Agent Quality Evaluation**: Built-in tools for evaluating agent performance metrics and quality.
- **MCP Server Integration**: Plug agents into an MCP server for agent management and message passing, which should be set up using the referenced repo.

## Prerequisites
- Databricks workspace
- Deployed MCP server (example taken from [`custom-mcp-to-dbx`](https://github.com/julie-nguyen-ds/custom-mcp-to-dbx))
- This example was run on a Machine Learning Runtime Cluster 16.4 LTS on Databricks

## Getting Started

### Clone the Repository
```sh
git clone https://github.com/julie-nguyen-ds/custom-mcp-agent.git
cd custom-mcp-agent
```

### Configure Agent 
Edit the environment files .env to set up agent and environment variable / connection details for Databricks and the MCP server.
Then walk through the notebook:
- 01. Setups secrets in your Databricks environment (for secure secret governance).
- 02. Creates the agent using langgraph, connects it with your MCP server and deploys it as an endpoint.
- 03. Evaluates the performance of your agents.
     
### Deploy to Databricks
Follow the deployment scripts or guidelines included to deploy your agent to the target Databricks workspace.

### Connect to MCP Server
Ensure the MCP server from [`custom-mcp-to-dbx`](https://github.com/julie-nguyen-ds/custom-mcp-to-dbx) is running. Configure this repository to communicate with the MCP endpoint for agent orchestration.

## Agent Quality Evaluation
Use included notebooks to run agent evaluation using mlflow 3. Results can be seen in the interface.

## Related Repositories

- MCP Server Deployment: [`custom-mcp-to-dbx`](https://github.com/julie-nguyen-ds/custom-mcp-to-dbx)
