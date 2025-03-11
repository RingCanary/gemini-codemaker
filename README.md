# Gemini Codemaker

A Rust CLI application that interfaces with Google's Gemini API to generate and execute code.

## Features

- **Interactive Chat with Gemini**: Continuous chat experience with the Gemini 2.0 Flash Thinking model
- **Execute Code**: Use the Gemini 1.5 Flash model to execute code snippets
- **Create Codebases**: Generate complete codebases from natural language descriptions
- **Command feedback loop for iterative improvements**
- **Support for file and folder creation, code writing, and command execution**
- **Direct code execution using Gemini 1.5 Flash model**
- **Configurable API endpoint and model selection**
- **Robust error handling** with custom error types and proper error propagation
- **Configurable logging** for better debugging and verbosity control

## Prerequisites

- Rust and Cargo installed
- Google Gemini API key (environment variable: `GEMINI_API_KEY`)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/gemini-codemaker.git
   cd gemini-codemaker
   ```
2. Build the project:
   ```
   cargo build --release
   ```
3. Set your Gemini API key:
   ```
   export GEMINI_API_KEY="your-api-key-here"
   ```

## Usage

### Chat Mode

To start an interactive chat session with the Gemini model:

```bash
# Start chat with an initial query
cargo run -- chat --query "What is Rust programming language?"

# Start chat without an initial query (will prompt for input)
cargo run -- chat
```

In the interactive chat mode:
- Type your queries and receive responses
- The chat maintains context across multiple interactions
- Type `exit` or `quit` to end the chat session

### Execute Mode

To execute code using the Gemini 1.5 Flash model:

```bash
cargo run -- execute --query "Write a Python function to calculate the factorial of a number and show its usage"
```

### Create Codebase Mode

To create a complete codebase from a description:

```bash
cargo run -- create-codebase --description "A React application with a Node.js backend that provides a simple todo list functionality" --output-dir my_project
```

This will generate a complete codebase based on your description in the specified output directory.

### Logging

The application uses the `env_logger` crate for logging. You can control the log level using the `RUST_LOG` environment variable:

```bash
# Show only error messages
RUST_LOG=error cargo run -- chat --query "What is Rust?"

# Show info and above (info, warn, error)
RUST_LOG=info cargo run -- execute --query "Write a Python hello world"

# Show all log messages including debug and trace
RUST_LOG=trace cargo run -- create-codebase --description "Simple web app" --output-dir test_app
```

Common log levels from least to most verbose: error, warn, info, debug, trace

## Configuration

The application requires a valid Gemini API key to function. You can obtain one from the [Google AI Studio](https://ai.google.dev/).

### Environment Variables

- `GEMINI_API_KEY`: Required for authenticating API requests
- `GEMINI_MODEL`: Optional variable to specify which model to use (defaults to gemini-2.0-flash-thinking-exp-01-21)
- `GEMINI_API_ENDPOINT`: Optional variable to specify a custom API endpoint

## Supported Commands

The application supports three main modes:

### Chat Mode

In the interactive chat mode:
1. The Gemini model responds with structured commands that this CLI can execute
2. Feedback from command execution is sent back to Gemini in subsequent queries
3. The chat maintains context across multiple interactions
4. Commands that can be executed:
   - `create_folder`: Create a new directory
   - `create_file`: Create an empty file
   - `write_code_to_file`: Write code to a specified file
   - `execute_command`: Execute a shell command

### Execute Mode

In execute mode, the application uses Gemini 1.5 Flash with code execution capabilities:

1. The model generates code based on your query
2. The model executes the code and returns the results
3. All execution happens on Google's servers, not locally

### Create Codebase Mode

In create codebase mode, the application generates a complete codebase based on your description.

## Feedback Loop

After executing commands in chat mode, the application sends feedback to Gemini in subsequent queries, allowing it to adjust its approach based on command success or failure. This feedback loop is maintained throughout the chat session.

## Dependencies

- serde, serde_json: For JSON serialization/deserialization
- clap: For command-line argument parsing
- reqwest: For making HTTP requests
- tokio: For asynchronous runtime
- log, env_logger: For configurable logging
- thiserror: For custom error types
- regex: For pattern matching in code extraction

## License

[MIT License](LICENSE)
