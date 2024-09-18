
# SMoA (Smart Mixture of Agents) - Intelligent Task Orchestration with Specialized Large Language Models (LLMs)

## Overview

This project implements an AI system that decomposes complex user requests into simpler subtasks, intelligently orchestrating a variety of specialized LLMs to process each task. The goal is to enhance efficiency, minimize operational costs, and optimize computational load by selecting the best-suited models for specific tasks. 

This system leverages:
- **Parallel and Asynchronous Task Processing**: For optimal resource utilization.
- **Cost and Load Optimization**: By balancing the use of free local models and paid APIs.
- **Mixture of Experts (MoE)** and **Retrieval-Augmented Generation (RAG)**: To combine expert outputs and enrich responses with data.

---

## Features

1. **User Request Analysis**: Decomposes complex queries into simpler subtasks.
2. **Subtask Assignment**: Allocates subtasks to specialized models (both local and paid APIs).
3. **Asynchronous Execution**: Processes tasks concurrently, reducing response time.
4. **Caching Mechanisms**: Avoids redundant computations, optimizing performance.
5. **Response Aggregation**: Combines multiple model outputs into a coherent final response.
6. **Retrieval-Augmented Generation (RAG)**: Enriches responses by retrieving relevant information from vector databases.

---

## Technology Stack

- **Programming Language**: Python 3.x
- **Frameworks/Libraries**:
  - `Flask`: Provides the REST API to interact with the system.
  - `LangChain`: For integrating LLMs with dynamic data sources.
  - `openai`, `httpx`, `sentence_transformers`: Access and interaction with language models.
  - `asyncio`: For asynchronous task management.
  - `aiocache`: Caching mechanism for optimizing repeated requests.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/https://github.com/kertser/SMoA.git
   cd SMoA
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file and add your **OpenAI API Key**:
   ```bash
   OPENAI_API_KEY=your-openai-api-key
   ```

---

## Configuration

- **config.ini**: This file stores various configuration parameters for logging, server settings, and model behavior. Ensure that this file is correctly set up before running the system.
  
- **models_config.ini**: Defines the models used for various subtasks based on the thematic categories. This file is critical for task decomposition and expert model selection.
  (will be replaced by a proper recommender system)
---

## Running the System

1. **Start the Server**:
   ```bash
   python server_API.py
   ```
   You should see logs indicating that the server is running on `localhost:5000`.

2. **Client Interaction**:
   The client can be used to interact with the server. Run the client script:
   ```bash
   python client.py
   ```
   The client will prompt you to enter a message, which will be sent to the server. The server processes the message, decomposes it into subtasks, and responds with the final aggregated answer.

---

## Example Workflow

### Request Example:

**User Request**:  
"Explain the significance of the Pythagorean theorem in modern physics and provide a Python code example that demonstrates its application."

**Steps**:
1. **Request Analysis**: The request is analyzed by the senior LLM and broken down into subtasks:
   - Explain the significance of the Pythagorean theorem in physics.
   - Provide a Python code example.
   
2. **Subtask Assignment**:
   - Physics subtask is assigned to a model specializing in physics.
   - The coding subtask is assigned to a programming-specific model.

3. **Response Aggregation**: The senior LLM aggregates the results from the specialized models into a final answer and sends it back to the user.

---

## Future Enhancements

- **Model Pool Expansion**: Integration with more specialized models to cover additional domains.
- **Optimized Caching**: More efficient caching strategies to reduce API calls and costs.
- **Improved Task Handling**: More advanced decomposition and aggregation methods for even better performance in complex queries.

---

## Contributions

TBD

---

## License

TBD