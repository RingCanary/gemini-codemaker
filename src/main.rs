use clap::Parser;
use log::{debug, error, info, trace, warn};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{env, fs, path::Path, process::Command as ProcessCommand};
use thiserror::Error;

// Custom error type for the application
#[derive(Error, Debug)]
pub enum AppError {
    #[error("API error: {0}")]
    ApiError(String),
    
    #[error("JSON parsing error: {0}")]
    JsonParseError(#[from] serde_json::Error),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),
    
    #[error("Environment error: {0}")]
    EnvError(String),
    
    #[error("Command execution error: {0}")]
    CommandError(String),
    
    #[error("Response error: {0}")]
    ResponseError(String),
}

impl From<String> for AppError {
    fn from(error: String) -> Self {
        AppError::ResponseError(error)
    }
}

#[derive(Parser, Debug)]
#[command(version = "1.0", about = "Interactive CLI with Gemini")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, clap::Subcommand)]
enum Commands {
    Chat {
        #[arg(long)]
        query: String,
    },
    Execute {
        #[arg(long)]
        query: String,
    },
    CreateCodebase {
        #[arg(long)]
        description: String,
        #[arg(long, default_value = ".")]
        output_dir: String,
    },
}

// Response structures for Gemini API
#[derive(Debug, Deserialize)]
struct GeminiApiResponse {
    candidates: Option<Vec<Candidate>>,
    prompt_feedback: Option<PromptFeedback>,
}

#[derive(Debug, Deserialize)]
struct PromptFeedback {
    block_reason: Option<String>,
    #[allow(dead_code)]
    safety_ratings: Option<Vec<SafetyRating>>,
}

#[derive(Debug, Deserialize)]
struct SafetyRating {
    #[allow(dead_code)]
    category: String,
    #[allow(dead_code)]
    probability: String,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    content: Content,
    #[allow(dead_code)]
    finish_reason: Option<String>,
    #[allow(dead_code)]
    index: Option<i32>,
    #[allow(dead_code)]
    safety_ratings: Option<Vec<SafetyRating>>,
}

#[derive(Debug, Deserialize)]
struct Content {
    #[allow(dead_code)]
    role: Option<String>,
    parts: Vec<Part>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Part {
    ExecutableCode {
        executable_code: ExecutableCode,
    },
    CodeExecutionResult {
        code_execution_result: CodeExecutionResult,
    },
    Text {
        text: String,
    },
}

#[derive(Debug, Deserialize)]
struct ExecutableCode {
    language: String,
    code: String,
}

#[derive(Debug, Deserialize)]
struct CodeExecutionResult {
    outcome: String,
    output: String,
}

// Structure for command processing
#[derive(Debug, Deserialize)]
struct GeminiResponse {
    commands: Vec<GeminiCommand>,
    user_message: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum GeminiCommand {
    CreateFolder { path: String },
    CreateFile { path: String, content: String },
    ExecuteCommand { command: String, args: Vec<String> },
}

#[derive(Serialize, Deserialize, Debug)]
enum CommandStatus {
    Success,
    Failure,
}

#[derive(Serialize, Deserialize, Debug)]
struct CommandFeedback {
    command_type: String,
    command_details: String,
    status: CommandStatus,
    message: String,
}

async fn chat_with_gemini(
    query: &str,
    system_info: &str,
    api_key: &str,
    feedback: &str,
) -> Result<GeminiApiResponse, AppError> {
    let client = Client::new();
    let gemini_api_endpoint =
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-thinking-exp-01-21:generateContent";

    let prompt_content = format!(
        "You are a helpful coding assistant. You will receive system information and user queries. Respond with a JSON object containing 'commands' and 'user_message'. 'commands' is an array of command objects, each with a 'type' and command-specific fields. Supported commands:\n- 'create_folder': {{ \"type\": \"create_folder\", \"path\": \"<folder_path>\" }}\n- 'create_file': {{ \"type\": \"create_file\", \"path\": \"<file_path>\" }}\n- 'write_code_to_file': {{ \"type\": \"write_code_to_file\", \"path\": \"<file_path>\", \"code\": \"<code_string>\" }}\n- 'execute_command': {{ \"type\": \"execute_command\", \"command\": \"<command_string>\" }}\n'user_message' is a string for user feedback after execution.\n\n**Feedback Loop:** After I execute your commands, I will provide feedback on their success or failure in subsequent queries. Use this feedback to improve your command generation. If a command fails, try to correct it or adjust your approach in the next turn.\n\nExample response for 'please build a hello-world python app for me':\n{{\n  \"commands\": [\n    {{\"type\": \"create_folder\", \"path\": \"user_projects\"}},\n    {{\"type\": \"create_file\", \"path\": \"user_projects/hello_world.py\"}},\n    {{\"type\": \"write_code_to_file\", \"path\": \"user_projects/hello_world.py\", \"code\": \"print('Hello, World!')\"}},\n    {{\"type\": \"execute_command\", \"command\": \"python user_projects/hello_world.py\"}}\n  ],\n  \"user_message\": \"Here is a hello-world Python app in 'user_projects'. It has been created and executed.\" \n}}\n\nSystem Information:\n{}\n\nPrevious Command Feedback (if any):\n{}\n\nUser Query:\n{}",
        system_info, feedback, query
    );

    let request_body = json!({
        "contents": [{
            "parts": [{"text": prompt_content}]
        }]
    });

    info!("Sending request to Gemini Pro API...");

    let response = client
        .post(gemini_api_endpoint)
        .header("Content-Type", "application/json")
        .query(&[("key", api_key)])
        .json(&request_body)
        .send()
        .await?;

    let status = response.status();
    let response_text = response.text().await?;

    info!("API Response Status: {}", status);

    if !status.is_success() {
        error!("API Error Response: {}", response_text);
        return Err(AppError::ApiError(format!(
            "API request failed with status {}: {}",
            status, response_text
        )));
    }

    match serde_json::from_str::<GeminiApiResponse>(&response_text) {
        Ok(api_response) => Ok(api_response),
        Err(e) => {
            error!("Failed to parse API response: {}", e);
            error!("Response text: {}", response_text);
            Err(AppError::JsonParseError(e))
        }
    }
}

async fn execute_with_gemini(
    query: &str,
    api_key: &str,
) -> Result<GeminiApiResponse, AppError> {
    let client = Client::new();
    let gemini_api_endpoint =
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-thinking-exp-01-21:generateContent";

    let request_body = json!({
        "tools": [{"code_execution": {}}],
        "contents": [
            {
                "role": "user",
                "parts": [{"text": query}]
            }
        ]
    });

    info!("Sending request to Gemini API...");

    let response = client
        .post(gemini_api_endpoint)
        .header("Content-Type", "application/json")
        .query(&[("key", api_key)])
        .json(&request_body)
        .send()
        .await?;

    let status = response.status();
    let response_text = response.text().await?;

    info!("API Response Status: {}", status);

    if !status.is_success() {
        error!("API Error Response: {}", response_text);
        return Err(AppError::ApiError(format!(
            "API request failed with status {}: {}",
            status, response_text
        )));
    }

    info!("API Response received. Processing...");

    match serde_json::from_str::<GeminiApiResponse>(&response_text) {
        Ok(api_response) => Ok(api_response),
        Err(e) => {
            error!("Failed to parse API response: {}", e);
            error!("Response text: {}", response_text);
            Err(AppError::JsonParseError(e))
        }
    }
}

async fn create_codebase_with_gemini(
    description: &str,
    output_dir: &str,
    api_key: &str,
) -> Result<GeminiApiResponse, AppError> {
    let client = Client::new();
    let gemini_api_endpoint =
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-thinking-exp-01-21:generateContent";

    // Create the output directory if it doesn't exist
    let output_path = Path::new(output_dir);
    if !output_path.exists() {
        fs::create_dir_all(output_path)?;
    }

    let prompt = format!(
        "Create a complete codebase based on this description: {}\n\n\
        Generate all necessary files for a working application. For each file:\n\
        1. Use a clear header with the filename (e.g., '## app.py' or 'File: app.py')\n\
        2. Provide the complete code content in a markdown code block with the appropriate language\n\
        3. Briefly explain what the file does after the code block\n\n\
        IMPORTANT: Make sure to include the actual code in markdown code blocks with the appropriate language tag, not just explanations.\n\
        For example, for a Python file:\n\
        ## app.py\n\
        ```python\n\
        # Your actual Python code here\n\
        print('Hello, world!')\n\
        ```\n\
        This file is the main entry point of the application.\n\n\
        Include a README.md with setup instructions, dependencies, and usage examples.\n\
        Make sure the codebase is well-structured, follows best practices, and is ready to run.\n\
        Format your response as markdown with code blocks for each file.",
        description
    );

    // Use the same request format as execute_with_gemini
    let request_body = json!({
        "tools": [{"code_execution": {}}],
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
    });

    info!("Sending request to Gemini API to create codebase...");

    let response = client
        .post(gemini_api_endpoint)
        .header("Content-Type", "application/json")
        .query(&[("key", api_key)])
        .json(&request_body)
        .send()
        .await?;

    let status = response.status();
    let response_text = response.text().await?;

    info!("API Response Status: {}", status);

    if !status.is_success() {
        error!("API Error Response: {}", response_text);
        return Err(AppError::ApiError(format!(
            "API request failed with status {}: {}",
            status, response_text
        )));
    }

    info!("API Response received. Processing...");

    match serde_json::from_str::<GeminiApiResponse>(&response_text) {
        Ok(api_response) => Ok(api_response),
        Err(e) => {
            error!("Failed to parse API response: {}", e);
            error!("Response text: {}", response_text);
            Err(AppError::JsonParseError(e))
        }
    }
}

/// Infers a file extension based on the content of the code
fn infer_extension_from_content(content: &str) -> String {
    if content.contains("<?php") {
        return "php".to_string();
    } else if content.contains("<!DOCTYPE html") || content.contains("<html") {
        return "html".to_string();
    } else if content.contains("import React") || content.contains("from 'react'") {
        return "jsx".to_string();
    } else if content.contains("import ") && content.contains("from '") && content.contains("export ") {
        return "js".to_string();
    } else if content.contains("#include <") {
        if content.contains("iostream") {
            return "cpp".to_string();
        } else {
            return "c".to_string();
        }
    } else if content.contains("package ") && content.contains("import ") && content.contains("public class ") {
        return "java".to_string();
    } else if content.contains("def ") && content.contains("import ") {
        return "py".to_string();
    } else if content.contains("fn ") && content.contains("pub ") && content.contains("use ") {
        return "rs".to_string();
    }

    // Default to txt if we can't infer
    trace!("Could not infer extension from content, defaulting to txt");
    "txt".to_string()
}

/// Returns a file extension based on the language name
fn get_extension_from_language(language: &str) -> String {
    match language.to_lowercase().as_str() {
        "python" | "py" => "py",
        "javascript" | "js" => "js",
        "typescript" | "ts" => "ts",
        "jsx" => "jsx",
        "tsx" => "tsx",
        "html" => "html",
        "css" => "css",
        "rust" | "rs" => "rs",
        "go" => "go",
        "java" => "java",
        "c" => "c",
        "cpp" | "c++" => "cpp",
        "csharp" | "cs" => "cs",
        "php" => "php",
        "ruby" | "rb" => "rb",
        "shell" | "sh" | "bash" => "sh",
        "sql" => "sql",
        "json" => "json",
        "yaml" | "yml" => "yml",
        "markdown" | "md" => "md",
        "dockerfile" => "Dockerfile",
        "makefile" => "Makefile",
        _ => "txt",
    }.to_string()
}

/// Extracts files from markdown text using headers and code blocks
fn extract_files_from_markdown(text: &str) -> Vec<(String, String)> {
    let mut files = Vec::new();
    let mut current_file = None;
    let mut current_content = String::new();

    // Split the text into lines for processing
    let lines: Vec<&str> = text.lines().collect();

    for (_i, line) in lines.iter().enumerate() {
        // Check for file header pattern: "```filename" or "```language:filename"
        if line.starts_with("```") && !line.trim_start_matches('`').is_empty() {
            // If we were already collecting a file, save it before starting a new one
            if let Some(filename) = current_file.take() {
                debug!("Extracted file from markdown: {}", filename);
                files.push((filename, current_content.clone()));
                current_content.clear();
            }

            let header = line.trim_start_matches('`').trim();
            
            // Handle different code block formats
            if header.contains(':') {
                // Format: ```language:filename
                let parts: Vec<&str> = header.splitn(2, ':').collect();
                if parts.len() == 2 {
                    let filename = parts[1].trim();
                    current_file = Some(filename.to_string());
                    trace!("Found file in markdown with language prefix: {}", filename);
                }
            } else if !header.contains(' ') {
                // Format: ```filename
                current_file = Some(header.to_string());
                trace!("Found file in markdown: {}", header);
            }
        }
        // Check for end of code block
        else if line.trim() == "```" && current_file.is_some() {
            if let Some(filename) = current_file.take() {
                debug!("Completed extraction of file: {}", filename);
                files.push((filename, current_content.clone()));
                current_content.clear();
            }
        }
        // If we're inside a code block, collect the content
        else if current_file.is_some() {
            current_content.push_str(line);
            current_content.push('\n');
        }
    }

    // Handle case where the last code block doesn't have a closing ```
    if let Some(filename) = current_file {
        debug!("Extracted file from markdown (unclosed block): {}", filename);
        files.push((filename, current_content));
    }

    info!("Extracted {} files from markdown", files.len());
    files
}

/// Extracts files from code blocks
fn extract_files_from_code_blocks(text: &str) -> Vec<(String, String)> {
    let mut files = Vec::new();
    let re = regex::Regex::new(r"(?m)^```(\w+)?\s*\n([\s\S]*?)^```").unwrap();

    // Counter for generating unique filenames
    let mut counter = 1;

    for cap in re.captures_iter(text) {
        let language = cap.get(1).map_or("txt", |m| m.as_str());
        let content = cap.get(2).map_or("", |m| m.as_str());

        // Generate a filename based on the language and counter
        let extension = get_extension_from_language(language);
        let filename = format!("file_{}.{}", counter, extension);
        counter += 1;

        debug!("Extracted code block: {} (language: {})", filename, language);
        files.push((filename, content.to_string()));
    }

    info!("Extracted {} files from code blocks", files.len());
    files
}

/// Cleans and validates a file path
fn clean_and_validate_file_path(file_path: &str) -> Result<String, AppError> {
    let path = file_path.trim();
    
    // Check for suspicious paths
    if path.contains("..") || path.starts_with('/') || path.starts_with('\\') {
        warn!("Suspicious file path detected: {}", path);
        return Err(AppError::ResponseError(format!("Suspicious file path: {}", path)));
    }
    
    // Normalize path separators
    let normalized_path = path.replace('\\', "/");
    
    trace!("Normalized file path: {} -> {}", path, normalized_path);
    Ok(normalized_path)
}

/// Writes files to disk and returns a list of created file paths
fn write_files_to_disk(
    files: Vec<(String, String)>,
    output_dir: &str,
) -> Result<Vec<String>, AppError> {
    let mut created_files = Vec::new();
    let mut file_counter = 0;

    for (file_path, content) in files {
        file_counter += 1;
        
        // Clean and validate the file path
        let clean_path = clean_and_validate_file_path(&file_path)?;
        
        // If the file doesn't have an extension, try to infer one from the content
        let final_path = if !clean_path.contains('.') {
            let extension = infer_extension_from_content(&content);
            format!("{}.{}", clean_path, extension)
        } else {
            clean_path
        };
        
        // Create the full path
        let full_path = Path::new(output_dir).join(&final_path);
        
        // Create parent directories if they don't exist
        if let Some(parent) = full_path.parent() {
            debug!("Creating parent directory: {}", parent.display());
            fs::create_dir_all(parent)?;
        }

        // Write the file
        fs::write(&full_path, content)?;
        info!("Created file: {}", full_path.display());
        created_files.push(full_path.to_string_lossy().to_string());
    }

    info!("Successfully created {} files", file_counter);
    Ok(created_files)
}

/// Creates files from a text response containing code blocks and file information
fn create_files_from_response(
    text: &str,
    output_dir: &str,
) -> Result<Vec<String>, AppError> {
    // First, try to extract files based on markdown patterns
    let mut files = extract_files_from_markdown(text);

    // If no files were found using markdown pattern, try to extract code blocks
    if files.is_empty() {
        debug!("No files found using markdown pattern, trying code blocks extraction");
        files = extract_files_from_code_blocks(text);
    }

    // If still no files, treat the entire response as a single file
    if files.is_empty() {
        warn!("No files found in structured format, treating entire response as a single file");
        files.push(("README.md".to_string(), text.to_string()));
    }

    write_files_to_disk(
        files,
        output_dir
    )
}

async fn execute_command(command: &str) -> Result<String, AppError> {
    let parts: Vec<&str> = command.split_whitespace().collect();
    if parts.is_empty() {
        error!("Empty command provided");
        return Err(AppError::CommandError("Empty command".to_string()));
    }
    let cmd = parts[0];
    let args = &parts[1..];
    
    debug!("Executing command: {} with args: {:?}", cmd, args);
    
    let output = ProcessCommand::new(cmd)
        .args(args)
        .output()
        .map_err(|e| {
            error!("Failed to execute command: {}", e);
            AppError::IoError(e)
        })?;
        
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        debug!("Command executed successfully");
        trace!("Command output: {}", stdout);
        Ok(stdout)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        error!("Command execution failed: {}", stderr);
        Err(AppError::CommandError(stderr))
    }
}

#[tokio::main]
async fn main() -> Result<(), AppError> {
    // Initialize the logger
    env_logger::init();
    
    let cli = Cli::parse();

    // Get API key from environment variable or prompt user if not set
    let api_key = match env::var("GEMINI_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            error!("GEMINI_API_KEY environment variable not set");
            return Err(AppError::EnvError(
                "GEMINI_API_KEY environment variable not set. Please set it with: export GEMINI_API_KEY=your_api_key_here".to_string()
            ));
        }
    };

    let system_info = get_system_info();
    let mut feedback_messages = Vec::new();
    let feedback_string = String::new();

    match &cli.command {
        Commands::Chat { query } => {
            info!("User Query: '{}'", query);
            
            let gemini_response = chat_with_gemini(query, &system_info, &api_key, &feedback_string)
                .await
                .map_err(|e| AppError::ApiError(format!("Error communicating with Gemini API: {}", e)))?;
            
            let candidates = gemini_response.candidates.ok_or_else(|| {
                if let Some(prompt_feedback) = gemini_response.prompt_feedback {
                    if let Some(block_reason) = prompt_feedback.block_reason {
                        error!("Request was blocked: {}", block_reason);
                        AppError::ResponseError(format!("Request was blocked: {}", block_reason))
                    } else {
                        error!("No candidates received from Gemini API");
                        AppError::ResponseError("No candidates received from Gemini API".to_string())
                    }
                } else {
                    error!("No candidates received from Gemini API");
                    AppError::ResponseError("No candidates received from Gemini API".to_string())
                }
            })?;
            
            let candidate = candidates.get(0).ok_or_else(|| {
                error!("No candidates in response");
                AppError::ResponseError("No candidates in response".to_string())
            })?;
            
            // Find the text part in the response
            let mut text_content = String::new();
            for part in &candidate.content.parts {
                if let Part::Text { text } = part {
                    text_content.push_str(&text);
                    break;
                }
            }

            if text_content.is_empty() {
                error!("No text content in response");
                return Err(AppError::ResponseError("No text content in response".to_string()));
            }
            
            debug!("Received text content: {}", text_content);
            
            let gemini_response = serde_json::from_str::<GeminiResponse>(&text_content)
                .map_err(|e| {
                    error!("Failed to parse JSON: {}\nRaw: {}", e, text_content);
                    AppError::JsonParseError(e)
                })?;
            
            feedback_messages.clear();
            for cmd in gemini_response.commands {
                let feedback = match cmd {
                    GeminiCommand::CreateFolder { path } => {
                        info!("Creating folder: {}", path);
                        let result = fs::create_dir_all(&path);
                        CommandFeedback {
                            command_type: "create_folder".to_string(),
                            command_details: format!("path: {}", path),
                            status: if result.is_ok() {
                                CommandStatus::Success
                            } else {
                                error!("Failed to create folder: {}", path);
                                CommandStatus::Failure
                            },
                            message: result
                                .map(|_| "Folder created".to_string())
                                .unwrap_or_else(|e| e.to_string()),
                        }
                    }
                    GeminiCommand::CreateFile { path, content } => {
                        info!("Creating file: {}", path);
                        let result = fs::write(&path, &content);
                        CommandFeedback {
                            command_type: "create_file".to_string(),
                            command_details: format!("path: {}", path),
                            status: if result.is_ok() {
                                CommandStatus::Success
                            } else {
                                error!("Failed to create file: {}", path);
                                CommandStatus::Failure
                            },
                            message: result
                                .map(|_| "File created".to_string())
                                .unwrap_or_else(|e| e.to_string()),
                        }
                    }
                    GeminiCommand::ExecuteCommand { command, args } => {
                        info!("Executing: {}", command);
                        let result = execute_command(&format!(
                            "{} {}",
                            command,
                            args.join(" ")
                        ))
                        .await;
                        CommandFeedback {
                            command_type: "execute_command".to_string(),
                            command_details: format!(
                                "command: {}",
                                command
                            ),
                            status: if result.is_ok() {
                                CommandStatus::Success
                            } else {
                                error!("Failed to execute command: {}", command);
                                CommandStatus::Failure
                            },
                            message: result.unwrap_or_else(|e| e.to_string()),
                        }
                    }
                };
                feedback_messages.push(feedback);
            }
            info!("User message: {}", gemini_response.user_message);
            println!("\n{}", gemini_response.user_message);
        }
        Commands::Execute { query } => {
            info!("User Query for Code Execution: '{}'", query);
            
            let gemini_response = execute_with_gemini(query, &api_key)
                .await
                .map_err(|e| AppError::ApiError(format!("Error communicating with Gemini API: {}", e)))?;
            
            let candidates = gemini_response.candidates.ok_or_else(|| {
                if let Some(prompt_feedback) = gemini_response.prompt_feedback {
                    if let Some(block_reason) = prompt_feedback.block_reason {
                        error!("Request was blocked: {}", block_reason);
                        AppError::ResponseError(format!("Request was blocked: {}", block_reason))
                    } else {
                        error!("No candidates received from Gemini API");
                        AppError::ResponseError("No candidates received from Gemini API".to_string())
                    }
                } else {
                    error!("No candidates received from Gemini API");
                    AppError::ResponseError("No candidates received from Gemini API".to_string())
                }
            })?;
            
            let candidate = candidates.get(0).ok_or_else(|| {
                error!("No candidates in response");
                AppError::ResponseError("No candidates in response".to_string())
            })?;
            
            println!("\n--- Gemini Response ---");

            // Process each part of the response
            for part in &candidate.content.parts {
                match part {
                    Part::Text { text } => {
                        if !text.is_empty() {
                            info!("{}", text);
                        }
                    }
                    Part::ExecutableCode { executable_code } => {
                        info!(
                            "\n--- Generated Code ({}): ---",
                            executable_code.language
                        );
                        info!("{}", executable_code.code);
                        info!("--- End of Generated Code ---\n");
                    }
                    Part::CodeExecutionResult {
                        code_execution_result,
                    } => {
                        info!(
                            "\n--- Execution Result: {} ---",
                            code_execution_result.outcome
                        );
                        info!("{}", code_execution_result.output);
                        info!("--- End of Execution Result ---\n");
                    }
                }
            }
        }
        Commands::CreateCodebase {
            description,
            output_dir,
        } => {
            info!("Creating codebase with description: '{}'", description);
            info!("Output directory: '{}'", output_dir);

            let gemini_response = create_codebase_with_gemini(description, output_dir, &api_key)
                .await
                .map_err(|e| AppError::ApiError(format!("Error communicating with Gemini API: {}", e)))?;
            
            let candidates = gemini_response.candidates.ok_or_else(|| {
                if let Some(prompt_feedback) = gemini_response.prompt_feedback {
                    if let Some(block_reason) = prompt_feedback.block_reason {
                        error!("Request was blocked: {}", block_reason);
                        AppError::ResponseError(format!("Request was blocked: {}", block_reason))
                    } else {
                        error!("No candidates received from Gemini API");
                        AppError::ResponseError("No candidates received from Gemini API".to_string())
                    }
                } else {
                    error!("No candidates received from Gemini API");
                    AppError::ResponseError("No candidates received from Gemini API".to_string())
                }
            })?;
            
            let candidate = candidates.get(0).ok_or_else(|| {
                error!("No candidates in response");
                AppError::ResponseError("No candidates in response".to_string())
            })?;
            
            // Find the text part in the response
            let mut text_content = String::new();
            for part in &candidate.content.parts {
                if let Part::Text { text } = part {
                    text_content.push_str(&text);
                }
            }

            if text_content.is_empty() {
                error!("No text content in response");
                return Err(AppError::ResponseError("No text content in response".to_string()));
            }
            
            info!("Received text content: {}", text_content);
            
            info!("--- Creating Files from Gemini Response ---");
            let created_files = create_files_from_response(&text_content, output_dir)
                .map_err(|e| AppError::ResponseError(format!("Error creating files: {}", e)))?;
            
            info!("--- Codebase Creation Complete ---");
            info!("Created {} files in {}", created_files.len(), output_dir);
            info!("Files created:");
            for file in created_files {
                info!("- {}", file);
            }
        }
    }
    Ok(())
}

fn get_system_info() -> String {
    format!(
        "OS: {}\nArch: {}\nDir: {:?}",
        env::consts::OS,
        env::consts::ARCH,
        env::current_dir().unwrap_or_default()
    )
}
