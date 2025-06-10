import gradio as gr
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
from AdvA import research_agent

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}

.main-header {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}

.tool-info {
    background: #f8f9fa;
    border-left: 4px solid #007bff;
    padding: 10px;
    margin: 10px 0;
    border-radius: 5px;
}

.confidence-high { color: #28a745; font-weight: bold; }
.confidence-medium { color: #ffc107; font-weight: bold; }
.confidence-low { color: #dc3545; font-weight: bold; }

.reasoning-chain {
    background: #e9ecef;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
}
"""

def format_response_with_analysis(result):
    """Format the agent response with detailed analysis display."""
    response = result['response']
    reasoning_chain = result.get('reasoning_chain', [])
    confidence_scores = result.get('confidence_scores', {})
    tools_used = result.get('tools_used', [])
    
    # Format main response
    formatted_response = f"## 🤖 Agent Response\n\n{response}\n\n"
    
    # Add reasoning chain if available
    if reasoning_chain:
        formatted_response += "## 🧠 Reasoning Process\n\n"
        for i, step in enumerate(reasoning_chain, 1):
            formatted_response += f"**Step {i}:** {step}\n\n"
    
    # Add confidence analysis
    if confidence_scores:
        formatted_response += "## 📊 Confidence Analysis\n\n"
        for key, score in confidence_scores.items():
            confidence_level = "High" if score > 0.8 else "Medium" if score > 0.6 else "Low"
            formatted_response += f"- **{key.title()}**: {score:.1%} ({confidence_level})\n"
        formatted_response += "\n"
    
    # Add tools used
    if tools_used:
        formatted_response += "## 🛠️ Tools Utilized\n\n"
        tool_names = [tool.get('name', 'Unknown') for tool in tools_used if isinstance(tool, dict)]
        for tool in tool_names:
            formatted_response += f"- {tool.replace('_', ' ').title()}\n"
        formatted_response += "\n"
    
    return formatted_response

def create_confidence_chart(confidence_scores):
    """Create a confidence visualization chart."""
    if not confidence_scores:
        return None
    
    df = pd.DataFrame(list(confidence_scores.items()), columns=['Metric', 'Confidence'])
    df['Confidence'] = df['Confidence'] * 100  # Convert to percentage
    
    fig = px.bar(
        df, 
        x='Metric', 
        y='Confidence',
        title='Confidence Levels by Analysis Type',
        color='Confidence',
        color_continuous_scale='RdYlGn',
        range_color=[0, 100]
    )
    
    fig.update_layout(
        xaxis_title="Analysis Type",
        yaxis_title="Confidence (%)",
        yaxis_range=[0, 100]
    )
    
    return fig

def chat_with_agent(message, history, analysis_type, max_results):
    """Main chat function with research agent."""
    if not message.strip():
        return history, "", None, "Please enter a message."
    
    try:
        # Prepare context based on analysis type
        context = {
            "analysis_type": analysis_type,
            "max_results": max_results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Get response from research agent
        result = research_agent.run(message, context)
        
        # Format the response
        formatted_response = format_response_with_analysis(result)
        
        # Create confidence chart
        confidence_chart = create_confidence_chart(result.get('confidence_scores', {}))
        
        # Update chat history
        history.append([message, formatted_response])
        
        # Create analysis summary
        analysis_summary = f"""
**Query Analysis:**
- **Processing Time**: {datetime.now().strftime('%H:%M:%S')}
- **Analysis Type**: {analysis_type}
- **Max Results**: {max_results}
- **Tools Used**: {len(result.get('tools_used', []))}
- **Reasoning Steps**: {len(result.get('reasoning_chain', []))}
"""
        
        return history, "", confidence_chart, analysis_summary
        
    except Exception as e:
        error_message = f"❌ **Error occurred**: {str(e)}\n\nPlease try again or rephrase your question."
        history.append([message, error_message])
        return history, "", None, f"**Error**: {str(e)}"

def get_conversation_summary():
    """Get conversation summary from the agent."""
    return research_agent.get_conversation_summary()

def clear_conversation():
    """Clear the conversation history."""
    research_agent.clear_memory()
    return [], None, "Conversation cleared successfully!"

def create_example_queries():
    """Create example queries for different capabilities."""
    examples = {
        "Mathematical Analysis": [
            "Calculate the derivative of x^3 + 2x^2 - 5x + 1 and find its critical points",
            "Perform statistical analysis on this dataset: [23, 45, 67, 89, 12, 34, 56, 78, 90, 21]",
            "Solve the equation x^2 + 4x + 4 = 0 symbolically"
        ],
        "Research & Information": [
            "Search for recent developments in artificial intelligence and analyze source credibility",
            "Find the latest research papers on climate change from arXiv",
            "Search PubMed for recent studies on COVID-19 treatments"
        ],
        "Complex Analysis": [
            "Compare the effectiveness of different machine learning algorithms based on recent papers",
            "Analyze the mathematical foundations of quantum computing",
            "Find and synthesize information about CRISPR gene editing applications"
        ]
    }
    return examples

# Create the Gradio interface
with gr.Blocks(css=custom_css, title="Advanced Research Agent") as demo:
    
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>🧠 Advanced Research Agent</h1>
        <p>AI assistant with mathematical reasoning, multi-source research, and intelligent analysis</p>
    </div>
    """)
    
    # Main interface
    with gr.Row():
        with gr.Column(scale=2):
            # Chat interface
            chatbot = gr.Chatbot(
                height=500,
                label="💬 Conversation",
                bubble_full_width=False,
                show_label=True
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask me anything: math problems, research questions, data analysis...",
                    label="Your Message",
                    lines=2,
                    scale=4
                )
                submit_btn = gr.Button("🚀 Send", variant="primary", scale=1)
            
            # Configuration options
            with gr.Row():
                analysis_type = gr.Dropdown(
                    choices=["general", "mathematical", "research", "fact_check"],
                    value="general",
                    label="Analysis Type",
                    scale=1
                )
                max_results = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Max Results",
                    scale=1
                )
        
        with gr.Column(scale=1):
            # Analysis panel
            gr.HTML("<h3>📊 Analysis Dashboard</h3>")
            
            confidence_plot = gr.Plot(label="Confidence Levels")
            
            analysis_info = gr.Textbox(
                label="📈 Session Analysis",
                lines=8,
                interactive=False
            )
            
            # Control buttons
            with gr.Row():
                summary_btn = gr.Button("📋 Summary", variant="secondary")
                clear_btn = gr.Button("🗑️ Clear", variant="stop")
    
    # Example queries section
    with gr.Accordion("💡 Example Queries", open=False):
        examples = create_example_queries()
        
        for category, queries in examples.items():
            with gr.Accordion(f"📚 {category}", open=False):
                for i, query in enumerate(queries):
                    example_btn = gr.Button(f"Try: {query[:50]}...", size="sm")
                    example_btn.click(
                        lambda q=query: q,
                        outputs=msg
                    )
    
    # Advanced features section
    with gr.Accordion("🔧 Advanced Features", open=False):
        gr.HTML("""
        <div class="tool-info">
            <h4>🧮 Mathematical Capabilities:</h4>
            <ul>
                <li><strong>Symbolic Mathematics</strong>: Solve equations, derivatives, integrals</li>
                <li><strong>Statistical Analysis</strong>: Mean, median, standard deviation, hypothesis testing</li>
                <li><strong>Advanced Calculations</strong>: Support for complex mathematical expressions</li>
            </ul>
        </div>
        
        <div class="tool-info">
            <h4>🔍 Research Tools:</h4>
            <ul>
                <li><strong>Web Search</strong>: Intelligent search with credibility scoring</li>
                <li><strong>arXiv Integration</strong>: Academic paper search with relevance analysis</li>
                <li><strong>PubMed Access</strong>: Medical literature with impact scoring</li>
            </ul>
        </div>
        
        <div class="tool-info">
            <h4>🧠 Intelligence Features:</h4>
            <ul>
                <li><strong>Reasoning Chain</strong>: Step-by-step problem solving</li>
                <li><strong>Source Analysis</strong>: Credibility and relevance scoring</li>
                <li><strong>Memory & Context</strong>: Conversation history and learning</li>
            </ul>
        </div>
        """)
    
    # Event handlers
    def handle_submit(message, history, analysis_type, max_results):
        return chat_with_agent(message, history, analysis_type, max_results)
    
    # Submit button and enter key
    submit_btn.click(
        handle_submit,
        inputs=[msg, chatbot, analysis_type, max_results],
        outputs=[chatbot, msg, confidence_plot, analysis_info]
    )
    
    msg.submit(
        handle_submit,
        inputs=[msg, chatbot, analysis_type, max_results],
        outputs=[chatbot, msg, confidence_plot, analysis_info]
    )
    
    # Summary button
    summary_btn.click(
        get_conversation_summary,
        outputs=analysis_info
    )
    
    # Clear button
    clear_btn.click(
        clear_conversation,
        outputs=[chatbot, confidence_plot, analysis_info]
    )
    
    # Footer
    gr.HTML("""
    <div style="text-align: center; margin-top: 20px; padding: 10px; background: #f8f9fa; border-radius: 5px;">
        <p><strong>Advanced Research Agent</strong> | Powered by Gemini 2.0 Flash | 
        Built with ❤️ using LangGraph, LangChain, and Gradio</p>
        <p><em>Features: Advanced Math • Multi-Source Research • Intelligent Analysis • Memory & Reasoning</em></p>
    </div>
    """)

if __name__ == "__main__":
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        debug=True
    )
