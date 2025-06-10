# 🧠 Advanced Research Agent

A sophisticated AI research assistant built with LangGraph and Gemini 2.0 that provides mathematical reasoning, multi-source research capabilities, and intelligent analysis with confidence scoring.

## ✨ Features

- **🧮 Mathematical Reasoning**: Symbolic mathematics, calculus, statistics, and advanced calculations
- **🌐 Intelligent Web Search**: Credibility-scored search results using Tavily API
- **📚 arXiv Integration**: Academic paper search with relevance analysis
- **🏥 PubMed Access**: Medical literature search with impact scoring
- **🧠 Reasoning Chain**: Step-by-step problem-solving 
- **📊 Confidence Analysis**: Reliability scoring for all results
- **💾 Memory & Context**: Session memory and conversation history
- **🎨 Modern Interface**: Gradio web interface with analytics dashboard

## 🚀 Installation & Setup

You can try AdvA directly via HuggingFace Space here: https://huggingface.co/spaces/huseyincavus/AdvA
Or, if you prefer to set it up locally, follow these steps:

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/huseyincavusbi/AG1.git
cd your/path/to/AG1

# Create virtual environment
python -m venv AG1

# Activate virtual environment
source AG1/bin/activate
```

### 2. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt
```

### 2. Get Google API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Copy the key

### 3. Configure Environment

Create a `.env` file and replace the placeholder with your actual API key:

```bash
# Create .env file
touch .env

# Add your API keys to the .env file
echo "GOOGLE_API_KEY=your_actual_google_api_key_here" > .env
echo "TAVILY_API_KEY=your_actual_tavily_api_key_here" >> .env
```

**Note:** Replace `your_actual_google_api_key_here` and `your_actual_tavily_api_key_here` with your actual API keys from Google AI Studio and Tavily respectively.

**⚠️ Security Warning:** Never hardcode API keys directly in your source code or commit them to version control. Always use environment variables or configuration files that are excluded from git (add `.env` to your `.gitignore` file).

```bash
# Run the Gradio app
python app.py
```

### 5. Access the Interface

Open your browser and go to: `http://localhost:7860`

## 🔧 Agent Architecture

The agent is built using LangGraph with the following components:

1. **State Management**: Tracks conversation history and tool calls
2. **Tool Selection**: Intelligently chooses appropriate tools based on user queries
3. **Tool Execution**: Executes selected tools and processes results
4. **Response Generation**: Uses Gemini 2.0 to generate natural language responses

### Available Tools

1. **Calculator Tool**
   - Evaluates mathematical expressions safely
   - Supports arithmetic, trigonometry, logarithms, etc.
   - Input: Mathematical expression string
   - Output: Calculated result

2. **Web Search Tool**
   - Searches the web using DuckDuckGo
   - Returns summarized results with URLs
   - Input: Search query string
   - Output: Formatted search results

3. **arXiv Search Tool**
   - Searches academic papers on arXiv
   - Returns paper details with abstracts
   - Input: Research topic/keywords
   - Output: Academic paper summaries

4. **PubMed Search Tool**
   - Searches medical literature via PubMed API
   - Returns medical research papers
   - Input: Medical research query
   - Output: Medical paper summaries

## 🎨 Gradio Interface Features

- **Chat Interface**: Conversational UI for natural interaction
- **Example Queries**: Pre-built examples for each tool type
- **Clear History**: Reset conversation anytime

## 🔍 Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   Error: Please set your GOOGLE_API_KEY in the .env file
   ```
   Solution: Configure your Google API key in the `.env` file

2. **Module Not Found**
   ```
   ModuleNotFoundError: No module named 'langgraph'
   ```
   Solution: Install dependencies with `pip install -r requirements.txt`

3. **Port Already in Use**
   ```
   OSError: [Errno 98] Address already in use
   ```
   Solution: Kill the process using port 7860 or change the port

## 🤝 Contributing

Feel free to contribute by:
- Adding new tools
- Improving the interface
- Enhancing error handling

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Google for the Gemini 2.0 API
- LangChain team for the excellent framework
- Gradio team for the UI framework
- Open source community for the various tools and libraries
- Excellent Agents Course for Huggingface
---

**Happy coding! 🚀**
