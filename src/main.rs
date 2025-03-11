use clap::Parser;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{env, fs, path::Path, process::Command as ProcessCommand};

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
        query: String 
    },
    Execute { 
        #[arg(long)]
        query: String 
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
    safety_ratings: Option<Vec<SafetyRating>>,
}

#[derive(Debug, Deserialize)]
struct SafetyRating {
    category: String,
    probability: String,
}

#[derive(Debug, Deserialize)]
struct Candidate { 
    content: Content,
    finish_reason: Option<String>,
    index: Option<i32>,
    safety_ratings: Option<Vec<SafetyRating>>,
}

#[derive(Debug, Deserialize)]
struct Content { 
    role: Option<String>,
    parts: Vec<Part> 
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Part {
    ExecutableCode { executable_code: ExecutableCode },
    CodeExecutionResult { code_execution_result: CodeExecutionResult },
    Text { text: String },
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
enum CommandStatus { Success, Failure }

#[derive(Serialize, Deserialize, Debug)]
struct CommandFeedback {
    command_type: String,
    command_details: String,
    status: CommandStatus,
    message: String,
}

async fn chat_with_gemini(query: &str, system_info: &str, api_key: &str, feedback: &str) -> Result<GeminiApiResponse, Box<dyn std::error::Error>> {
    let client = Client::new();
    let gemini_api_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent";

    let prompt_content = format!(
        "You are a helpful coding assistant. You will receive system information and user queries. Respond with a JSON object containing 'commands' and 'user_message'. 'commands' is an array of command objects, each with a 'type' and command-specific fields. Supported commands:\n- 'create_folder': {{ \"type\": \"create_folder\", \"path\": \"<folder_path>\" }}\n- 'create_file': {{ \"type\": \"create_file\", \"path\": \"<file_path>\" }}\n- 'write_code_to_file': {{ \"type\": \"write_code_to_file\", \"path\": \"<file_path>\", \"code\": \"<code_string>\" }}\n- 'execute_command': {{ \"type\": \"execute_command\", \"command\": \"<command_string>\" }}\n'user_message' is a string for user feedback after execution.\n\n**Feedback Loop:** After I execute your commands, I will provide feedback on their success or failure in subsequent queries. Use this feedback to improve your command generation. If a command fails, try to correct it or adjust your approach in the next turn.\n\nExample response for 'please build a hello-world python app for me':\n{{\n  \"commands\": [\n    {{\"type\": \"create_folder\", \"path\": \"user_projects\"}},\n    {{\"type\": \"create_file\", \"path\": \"user_projects/hello_world.py\"}},\n    {{\"type\": \"write_code_to_file\", \"path\": \"user_projects/hello_world.py\", \"code\": \"print('Hello, World!')\"}},\n    {{\"type\": \"execute_command\", \"command\": \"python user_projects/hello_world.py\"}}\n  ],\n  \"user_message\": \"Here is a hello-world Python app in 'user_projects'. It has been created and executed.\" \n}}\n\nSystem Information:\n{}\n\nPrevious Command Feedback (if any):\n{}\n\nUser Query:\n{}",
        system_info, feedback, query
    );

    let request_body = json!({
        "contents": [{
            "parts": [{"text": prompt_content}]
        }]
    });

    println!("Sending request to Gemini Pro API...");
    
    let response = client.post(gemini_api_endpoint)
        .header("Content-Type", "application/json")
        .query(&[("key", api_key)])
        .json(&request_body)
        .send()
        .await?;
    
    let status = response.status();
    let response_text = response.text().await?;
    
    println!("API Response Status: {}", status);
    
    if !status.is_success() {
        println!("API Error Response: {}", response_text);
        return Err(format!("API request failed with status {}: {}", status, response_text).into());
    }
    
    match serde_json::from_str::<GeminiApiResponse>(&response_text) {
        Ok(api_response) => Ok(api_response),
        Err(e) => {
            println!("Failed to parse API response: {}", e);
            println!("Response text: {}", response_text);
            Err(format!("Failed to parse API response: {}", e).into())
        }
    }
}

async fn execute_with_gemini(query: &str, api_key: &str) -> Result<GeminiApiResponse, Box<dyn std::error::Error>> {
    let client = Client::new();
    let gemini_api_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent";

    let request_body = json!({
        "tools": [{"code_execution": {}}],
        "contents": [
            {
                "role": "user",
                "parts": [{"text": query}]
            }
        ]
    });

    println!("Sending request to Gemini API...");
    
    let response = client.post(gemini_api_endpoint)
        .header("Content-Type", "application/json")
        .query(&[("key", api_key)])
        .json(&request_body)
        .send()
        .await?;
    
    let status = response.status();
    let response_text = response.text().await?;
    
    println!("API Response Status: {}", status);
    
    if !status.is_success() {
        println!("API Error Response: {}", response_text);
        return Err(format!("API request failed with status {}: {}", status, response_text).into());
    }
    
    println!("API Response: {}", response_text);
    
    match serde_json::from_str::<GeminiApiResponse>(&response_text) {
        Ok(api_response) => Ok(api_response),
        Err(e) => {
            println!("Failed to parse API response: {}", e);
            println!("Response text: {}", response_text);
            Err(format!("Failed to parse API response: {}", e).into())
        }
    }
}

async fn create_codebase_with_gemini(description: &str, output_dir: &str, api_key: &str) -> Result<GeminiApiResponse, Box<dyn std::error::Error>> {
    let client = Client::new();
    let gemini_api_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent";

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
        3. Explain what the file does\n\n\
        Include a README.md with setup instructions, dependencies, and usage examples.\n\
        Make sure the codebase is well-structured, follows best practices, and is ready to run.\n\
        Format your response as markdown with code blocks for each file.",
        description
    );

    let request_body = json!({
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
    });

    println!("Sending request to Gemini API to create codebase...");
    
    let response = client.post(gemini_api_endpoint)
        .header("Content-Type", "application/json")
        .query(&[("key", api_key)])
        .json(&request_body)
        .send()
        .await?;
    
    let status = response.status();
    let response_text = response.text().await?;
    
    println!("API Response Status: {}", status);
    
    if !status.is_success() {
        println!("API Error Response: {}", response_text);
        return Err(format!("API request failed with status {}: {}", status, response_text).into());
    }
    
    println!("API Response received. Processing...");
    
    match serde_json::from_str::<GeminiApiResponse>(&response_text) {
        Ok(api_response) => Ok(api_response),
        Err(e) => {
            println!("Failed to parse API response: {}", e);
            println!("Response text: {}", response_text);
            Err(format!("Failed to parse API response: {}", e).into())
        }
    }
}

fn create_files_from_response(text: &str, output_dir: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    // Try to find patterns like "## filename.py" or "File: filename.py" in the text
    let mut files = Vec::new();
    let mut file_counter = 0;
    
    // First, try to extract files based on markdown patterns
    let mut lines = text.lines().peekable();
    let mut current_filename = String::new();
    let mut in_code_block = false;
    let mut current_code = String::new();
    let mut current_language = String::new();
    
    while let Some(line) = lines.next() {
        // Check for file headers
        if (line.starts_with("## ") || line.starts_with("### ")) && !in_code_block {
            // If we have a previous file, save it
            if !current_filename.is_empty() && !current_code.is_empty() {
                files.push((current_filename.clone(), current_code.clone()));
                current_code.clear();
            }
            
            // Extract filename from header
            let header_parts: Vec<&str> = line.trim_start_matches('#').trim().split_whitespace().collect();
            if !header_parts.is_empty() {
                current_filename = header_parts[0].to_string();
            }
        }
        // Check for "File:" pattern
        else if line.contains("File:") && !in_code_block {
            // If we have a previous file, save it
            if !current_filename.is_empty() && !current_code.is_empty() {
                files.push((current_filename.clone(), current_code.clone()));
                current_code.clear();
            }
            
            // Extract filename
            let parts: Vec<&str> = line.split("File:").collect();
            if parts.len() > 1 {
                current_filename = parts[1].trim().to_string();
            }
        }
        // Handle code blocks
        else if line.trim().starts_with("```") {
            if !in_code_block {
                in_code_block = true;
                // Extract language if specified
                current_language = line.trim().strip_prefix("```").unwrap_or("").trim().to_string();
            } else {
                in_code_block = false;
                
                // If we have a filename and code, save it
                if !current_code.is_empty() {
                    // If no filename was found, generate one based on the language
                    if current_filename.is_empty() {
                        file_counter += 1;
                        let extension = match current_language.to_lowercase().as_str() {
                            "python" | "py" => ".py",
                            "javascript" | "js" => ".js",
                            "html" => ".html",
                            "css" => ".css",
                            "json" => ".json",
                            "rust" | "rs" => ".rs",
                            "markdown" | "md" => ".md",
                            _ => ".txt",
                        };
                        current_filename = format!("file_{}{}", file_counter, extension);
                    }
                    
                    files.push((current_filename.clone(), current_code.clone()));
                    current_code.clear();
                    current_filename.clear();
                }
            }
        }
        // Collect code content
        else if in_code_block {
            current_code.push_str(line);
            current_code.push('\n');
        }
    }
    
    // Add the last file if there is one
    if !current_filename.is_empty() && !current_code.is_empty() {
        files.push((current_filename, current_code));
    }
    
    // If no files were found with the above method, use a more generic approach
    if files.is_empty() {
        // Look for code blocks with language specifiers
        let mut in_code_block = false;
        let mut language = String::new();
        let mut code_content = String::new();
        
        for line in text.lines() {
            if line.trim().starts_with("```") {
                if !in_code_block {
                    // Start of code block
                    in_code_block = true;
                    language = line.trim().strip_prefix("```").unwrap_or("").trim().to_string();
                    code_content.clear();
                } else {
                    // End of code block
                    in_code_block = false;
                    
                    // Generate a filename based on the language if we have code
                    if !code_content.is_empty() {
                        file_counter += 1;
                        let extension = match language.to_lowercase().as_str() {
                            "python" | "py" => ".py",
                            "javascript" | "js" => ".js",
                            "html" => ".html",
                            "css" => ".css",
                            "json" => ".json",
                            "rust" | "rs" => ".rs",
                            "markdown" | "md" => ".md",
                            _ => ".txt",
                        };
                        
                        let filename = format!("file_{}{}", file_counter, extension);
                        files.push((filename, code_content.clone()));
                    }
                }
            } else if in_code_block {
                code_content.push_str(line);
                code_content.push('\n');
            }
        }
    }
    
    // Process the extracted files
    let mut created_files = Vec::new();
    
    for (mut file_path, content) in files {
        // Clean up file path (remove quotes, etc.)
        file_path = file_path.trim_matches(|c: char| c == '"' || c == '\'' || c == '`' || c.is_whitespace()).to_string();
        
        // If the file path doesn't have an extension, try to infer one from the content
        if !file_path.contains('.') {
            let extension = if content.contains("def ") || content.contains("import ") {
                ".py"
            } else if content.contains("function") || content.contains("const ") || content.contains("let ") || content.contains("var ") {
                ".js"
            } else if content.contains("<html") || content.contains("<!DOCTYPE html") {
                ".html"
            } else if content.contains("body {") || content.contains("margin:") {
                ".css"
            } else if content.trim().starts_with("{") && content.trim().ends_with("}") {
                ".json"
            } else if content.contains("fn ") || content.contains("struct ") || content.contains("impl ") {
                ".rs"
            } else if content.contains("# ") || content.contains("## ") {
                ".md"
            } else {
                ".txt"
            };
            file_path = format!("{}{}", file_path, extension);
        }
        
        // If we still don't have a valid filename, generate one
        if file_path.is_empty() || file_path == "." || file_path == ".." || file_path.starts_with("File:") {
            file_counter += 1;
            
            // Try to infer file type from content
            let extension = if content.contains("def ") || content.contains("import ") {
                ".py"
            } else if content.contains("function") || content.contains("const ") || content.contains("let ") || content.contains("var ") {
                ".js"
            } else if content.contains("<html") || content.contains("<!DOCTYPE html") {
                ".html"
            } else if content.contains("body {") || content.contains("margin:") {
                ".css"
            } else if content.trim().starts_with("{") && content.trim().ends_with("}") {
                ".json"
            } else if content.contains("fn ") || content.contains("struct ") || content.contains("impl ") {
                ".rs"
            } else if content.contains("# ") || content.contains("## ") {
                ".md"
            } else {
                ".txt"
            };
            
            file_path = format!("file_{}{}", file_counter, extension);
        }
        
        // Create the full path
        let full_path = Path::new(output_dir).join(&file_path);
        
        // Create parent directories if they don't exist
        if let Some(parent) = full_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)?;
            }
        }
        
        // Write the file
        fs::write(&full_path, content)?;
        println!("Created file: {}", full_path.display());
        created_files.push(full_path.to_string_lossy().to_string());
    }
    
    Ok(created_files)
}

fn get_system_info() -> String {
    format!("OS: {}\nArch: {}\nDir: {:?}", env::consts::OS, env::consts::ARCH, env::current_dir().unwrap_or_default())
}

async fn execute_command(command: &str) -> Result<String, String> {
    let parts: Vec<&str> = command.split_whitespace().collect();
    if parts.is_empty() { return Err("Empty command".to_string()); }
    let cmd = parts[0];
    let args = &parts[1..];
    let output = ProcessCommand::new(cmd).args(args).output()
        .map_err(|e| e.to_string())?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    // Get API key from environment variable or prompt user if not set
    let api_key = match env::var("GEMINI_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!("Error: GEMINI_API_KEY environment variable not set");
            eprintln!("Please set it with: export GEMINI_API_KEY=your_api_key_here");
            return Ok(());
        }
    };
    
    let system_info = get_system_info();
    let mut feedback_messages = Vec::new();
    let mut feedback_string = String::new();

    match &cli.command {
        Commands::Chat { query } => {
            println!("User Query: '{}'", query);
            match chat_with_gemini(query, &system_info, &api_key, &feedback_string).await {
                Ok(gemini_response) => {
                    if let Some(candidates) = gemini_response.candidates {
                        if let Some(candidate) = candidates.get(0) {
                            // Find the text part in the response
                            let mut text_content = String::new();
                            for part in &candidate.content.parts {
                                if let Part::Text { text } = part {
                                    text_content.push_str(&text);
                                    break;
                                }
                            }
                            
                            if !text_content.is_empty() {
                                match serde_json::from_str::<GeminiResponse>(&text_content) {
                                    Ok(gemini_response) => {
                                        feedback_messages.clear();
                                        for cmd in gemini_response.commands {
                                            let feedback = match cmd {
                                                GeminiCommand::CreateFolder { path } => {
                                                    println!("Creating folder: {}", path);
                                                    let result = fs::create_dir_all(&path);
                                                    CommandFeedback {
                                                        command_type: "create_folder".to_string(),
                                                        command_details: format!("path: {}", path),
                                                        status: if result.is_ok() { CommandStatus::Success } else { CommandStatus::Failure },
                                                        message: result.map(|_| "Folder created".to_string())
                                                            .unwrap_or_else(|e| e.to_string()),
                                                    }
                                                }
                                                GeminiCommand::CreateFile { path, content } => {
                                                    println!("Creating file: {}", path);
                                                    let result = fs::write(&path, &content);
                                                    CommandFeedback {
                                                        command_type: "create_file".to_string(),
                                                        command_details: format!("path: {}", path),
                                                        status: if result.is_ok() { CommandStatus::Success } else { CommandStatus::Failure },
                                                        message: result.map(|_| "File created".to_string())
                                                            .unwrap_or_else(|e| e.to_string()),
                                                    }
                                                }
                                                GeminiCommand::ExecuteCommand { command, args } => {
                                                    println!("Executing: {}", command);
                                                    let result = execute_command(&format!("{} {}", command, args.join(" "))).await;
                                                    CommandFeedback {
                                                        command_type: "execute_command".to_string(),
                                                        command_details: format!("command: {}", command),
                                                        status: if result.is_ok() { CommandStatus::Success } else { CommandStatus::Failure },
                                                        message: result.unwrap_or_else(|e| e),
                                                    }
                                                }
                                            };
                                            feedback_messages.push(feedback);
                                        }
                                        println!("\n{}", gemini_response.user_message);
                                    },
                                    Err(e) => eprintln!("Failed to parse JSON: {}\nRaw: {}", e, text_content),
                                }
                            } else {
                                eprintln!("No text content in response");
                            }
                        } else {
                            eprintln!("No candidates in response");
                        }
                    } else if let Some(prompt_feedback) = gemini_response.prompt_feedback {
                        if let Some(block_reason) = prompt_feedback.block_reason {
                            eprintln!("Request was blocked: {}", block_reason);
                        } else {
                            eprintln!("No candidates received from Gemini API");
                        }
                    } else {
                        eprintln!("No candidates received from Gemini API");
                    }
                },
                Err(e) => {
                    eprintln!("Error communicating with Gemini API: {}", e);
                }
            }
        },
        Commands::Execute { query } => {
            println!("User Query for Code Execution: '{}'", query);
            match execute_with_gemini(query, &api_key).await {
                Ok(gemini_response) => {
                    if let Some(candidates) = gemini_response.candidates {
                        if let Some(candidate) = candidates.get(0) {
                            println!("\n--- Gemini Response ---");
                            
                            // Process each part of the response
                            for part in &candidate.content.parts {
                                match part {
                                    Part::Text { text } => {
                                        if !text.is_empty() {
                                            println!("{}", text);
                                        }
                                    },
                                    Part::ExecutableCode { executable_code } => {
                                        println!("\n--- Generated Code ({}): ---", executable_code.language);
                                        println!("{}", executable_code.code);
                                        println!("--- End of Generated Code ---\n");
                                    },
                                    Part::CodeExecutionResult { code_execution_result } => {
                                        println!("\n--- Execution Result: {} ---", code_execution_result.outcome);
                                        println!("{}", code_execution_result.output);
                                        println!("--- End of Execution Result ---\n");
                                    },
                                }
                            }
                        } else {
                            eprintln!("No candidates in response");
                        }
                    } else if let Some(prompt_feedback) = gemini_response.prompt_feedback {
                        if let Some(block_reason) = prompt_feedback.block_reason {
                            eprintln!("Request was blocked: {}", block_reason);
                        } else {
                            eprintln!("No candidates received from Gemini API");
                        }
                    } else {
                        eprintln!("No candidates received from Gemini API");
                    }
                },
                Err(e) => {
                    eprintln!("Error communicating with Gemini API: {}", e);
                }
            }
        }
        Commands::CreateCodebase { description, output_dir } => {
            println!("Creating codebase with description: '{}'", description);
            println!("Output directory: '{}'", output_dir);
            
            match create_codebase_with_gemini(description, output_dir, &api_key).await {
                Ok(gemini_response) => {
                    if let Some(candidates) = gemini_response.candidates {
                        if let Some(candidate) = candidates.get(0) {
                            // Find the text part in the response
                            let mut text_content = String::new();
                            for part in &candidate.content.parts {
                                if let Part::Text { text } = part {
                                    text_content.push_str(&text);
                                }
                            }
                            
                            if !text_content.is_empty() {
                                println!("\n--- Creating Files from Gemini Response ---");
                                match create_files_from_response(&text_content, output_dir) {
                                    Ok(created_files) => {
                                        println!("\n--- Codebase Creation Complete ---");
                                        println!("Created {} files in {}", created_files.len(), output_dir);
                                        println!("\nFiles created:");
                                        for file in created_files {
                                            println!("- {}", file);
                                        }
                                    },
                                    Err(e) => {
                                        eprintln!("Error creating files: {}", e);
                                    }
                                }
                            } else {
                                eprintln!("No text content in response");
                            }
                        } else {
                            eprintln!("No candidates in response");
                        }
                    } else if let Some(prompt_feedback) = gemini_response.prompt_feedback {
                        if let Some(block_reason) = prompt_feedback.block_reason {
                            eprintln!("Request was blocked: {}", block_reason);
                        } else {
                            eprintln!("No candidates received from Gemini API");
                        }
                    } else {
                        eprintln!("No candidates received from Gemini API");
                    }
                },
                Err(e) => {
                    eprintln!("Error communicating with Gemini API: {}", e);
                }
            }
        }
    }
    Ok(())
}
