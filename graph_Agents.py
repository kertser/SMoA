import openai
import httpx
import logging
from flask import jsonify
import asyncio
import os
from dotenv import load_dotenv
import networkx as nx
import matplotlib.pyplot as plt

# Loading environment variables, like API keys
load_dotenv()

# Add detailed logging configuration
logging.basicConfig(level=logging.INFO)

# Create graph to track agent interactions
interaction_graph = nx.DiGraph()

async def openai_post_request(messages, model_name, max_tokens, temperature):
    """Send a request to the OpenAI API."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    retries = 3  # Number of attempts
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Check for successful request
            return response.json()  # Return JSON response from OpenAI API
        except Exception as e:
            logging.error(f"Error during OpenAI request: {str(e)}")
            raise


class Agent:
    """Base Agent class to process lexical elements via GPT-4"""
    def __init__(self, name, system_prompt):
        self.name = name
        self.system_prompt = system_prompt

    async def run(self, user_input):
        logging.info(f"Agent {self.name} processing input: {user_input}")
        try:
            # Call GPT-4 model through OpenAI API
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input}
            ]
            response = await openai_post_request(messages, "gpt-4", 500, 0.7)
            return response['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"Error in {self.name}: {e}")
            return f"Error: {str(e)}"


# Agents for specific lexical elements
class AssertionAgent(Agent):
    def __init__(self):
        super().__init__("Assertion Agent", "You are an expert in processing statements. Analyze and respond to this statement.")


class QuestionAgent(Agent):
    def __init__(self):
        super().__init__("Question Agent", "You are an expert in processing questions. Answer the following question.")


class HypothesisAgent(Agent):
    def __init__(self):
        super().__init__("Hypothesis Agent", "You are an expert in evaluating hypotheses. Analyze and respond to this hypothesis.")


class ThesisAgent(Agent):
    def __init__(self):
        super().__init__("Thesis Agent", "You are an expert in evaluating arguments and thesis statements. Respond to the following thesis.")


class ConditionAgent(Agent):
    def __init__(self):
        super().__init__("Condition Agent", "You are an expert in evaluating conditional statements. Analyze the following condition.")


class ConjunctionAgent(Agent):
    def __init__(self):
        super().__init__("Conjunction Agent", "You are an expert in merging similar responses. Combine the following contexts into one.")


class SplitterAgent(Agent):
    def __init__(self):
        super().__init__("Splitter Agent", "You are an expert in dividing tasks into sub-tasks. Split the following task into smaller tasks.")


class DefinitionAgent(Agent):
    def __init__(self):
        super().__init__("Definition Agent", "You are an expert in defining terms. Define the following term.")


class ExplorationAgent(Agent):
    def __init__(self):
        super().__init__("Exploration Agent", "You are an expert in exploring concepts. Investigate and explain the following concept.")


class ComparisonAgent(Agent):
    def __init__(self):
        super().__init__("Comparison Agent", "You are an expert in comparing concepts. Compare the following concepts.")


class AnalysisAgent(Agent):
    def __init__(self):
        super().__init__("Analysis Agent", "You are an expert in analyzing concepts. Analyze the following statement.")


### ------- Graph Structure

class AgentGraph:
    """Graph structure to manage agents and their interactions"""

    def __init__(self):
        self.agents = {}  # Dictionary to hold agents
        self.graph = nx.DiGraph()  # The graph object to hold dynamic interactions
        self.edge_labels = {}  # Dictionary to hold edge labels (prompts)

    def add_agent(self, agent_name, agent):
        """Add an agent to the graph"""
        self.agents[agent_name] = agent

    async def run(self, input_prompt):
        """Run the graph by breaking the input into sub-tasks and passing them through agents"""
        sub_tasks = await self.decompose_task(input_prompt)

        # Process subtasks through relevant agents
        context = None
        previous_task = input_prompt  # Store the original input as the first node
        self.graph.add_node(previous_task)  # Add the input as the first node

        for task in sub_tasks:
            agent_type = self.identify_agent_type(task)
            if agent_type in self.agents:
                logging.info(f"Running task '{task}' with agent: {agent_type}")
                context = await self.agents[agent_type].run(context or input_prompt)

                # Add agents and interactions dynamically to the graph
                if agent_type not in self.graph:
                    self.graph.add_node(agent_type)
                self.graph.add_edge(previous_task, agent_type)  # Add the interaction
                self.edge_labels[(previous_task, agent_type)] = task  # Label the edge with the task prompt
                previous_task = agent_type  # Move to the next node
            else:
                logging.warning(f"No agent found for task: {task}")

        # Visualize the graph
        self.visualize_graph()

        # Finally return the result
        return context

    async def decompose_task(self, prompt):
        """Use the GPT-4 model to break a complex prompt into multiple components."""
        messages = [
            {"role": "system", "content": "Break this prompt into sub-tasks."},
            {"role": "user", "content": prompt}
        ]
        response = await openai_post_request(messages, "gpt-4", 500, 0.7)
        components = response['choices'][0]['message']['content'].split('\n')
        return [component.strip() for component in components if component.strip()]

    def identify_agent_type(self, task):
        """Identify the type of agent required based on the task"""
        task_lower = task.lower()

        if "assertion" in task_lower:
            return "assertion"
        elif "question" in task_lower or "what" in task_lower:
            return "question"
        elif "hypothesis" in task_lower or "suppose" in task_lower:
            return "hypothesis"
        elif "thesis" in task_lower:
            return "thesis"
        elif "condition" in task_lower or "if" in task_lower:
            return "condition"
        elif "define" in task_lower:
            return "definition"  # Add a definition agent for 'define' tasks
        elif "investigate" in task_lower or "explore" in task_lower:
            return "exploration"  # Add an exploration agent for 'investigate' tasks
        elif "compare" in task_lower:
            return "comparison"  # Add a comparison agent
        elif "analyze" in task_lower or "explain" in task_lower:
            return "analysis"  # Add an analysis agent
        elif "conclusion" in task_lower:
            return "conjunction"  # Use conjunction for conclusion
        else:
            return None  # No matching agent found

    def visualize_graph(self):
        """Visualize the interaction graph with edge labels"""
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', font_weight='bold', node_size=2000, font_size=10)

        # Draw edge labels (prompts between agents)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=self.edge_labels, font_color='red')

        plt.title('Agent Interaction Graph with Prompts')
        plt.show()


async def process_user_prompt(prompt):
    """Process a user input by running it through the agent graph"""
    # Initialize the graph
    agent_graph = AgentGraph()

    # Add existing agents
    agent_graph.add_agent("assertion", AssertionAgent())
    agent_graph.add_agent("question", QuestionAgent())
    agent_graph.add_agent("hypothesis", HypothesisAgent())
    agent_graph.add_agent("thesis", ThesisAgent())
    agent_graph.add_agent("condition", ConditionAgent())
    agent_graph.add_agent("conjunction", ConjunctionAgent())
    agent_graph.add_agent("splitter", SplitterAgent())
    agent_graph.add_agent("definition", DefinitionAgent())
    agent_graph.add_agent("exploration", ExplorationAgent())
    agent_graph.add_agent("comparison", ComparisonAgent())
    agent_graph.add_agent("analysis", AnalysisAgent())

    # Run the user prompt through the graph
    final_output = await agent_graph.run(prompt)

    return final_output


# Example usage
async def main():
    user_input = "What are the benefits of quantum computing compared to classical computing?"
    final_output = await process_user_prompt(user_input)
    print("Final Output:", final_output)


# Entry point for the asyncio event loop
if __name__ == "__main__":
    # Run the main event loop for asynchronous operations
    asyncio.run(main())
