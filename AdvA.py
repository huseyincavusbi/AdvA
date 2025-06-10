import os
import re
import math
import json
import statistics
import operator
import requests
import arxiv
import sympy as sp
import tldextract
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# Agent State definition with memory and context
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    tool_calls: List[Dict[str, Any]]
    memory: Dict[str, Any]
    context: Dict[str, Any]
    reasoning_chain: List[str]
    confidence_scores: Dict[str, float]

# Initialize the LLM with better system prompt
SYSTEM_PROMPT = """You are an advanced AI research assistant with expertise in mathematics, science, and information retrieval. 

Key capabilities:
1. **Mathematical Reasoning**: Solve complex equations, perform statistical analysis, and handle symbolic mathematics
2. **Research Intelligence**: Search and synthesize information from multiple sources (web, arXiv, PubMed)
3. **Critical Thinking**: Evaluate source credibility, cross-reference information, and provide confidence levels
4. **Memory & Context**: Remember previous interactions and build upon them
5. **Adaptive Learning**: Adjust responses based on user expertise level and preferences

Guidelines:
- Always show your reasoning process step by step
- Provide confidence levels for your answers (0-100%)
- Cross-reference multiple sources when possible
- Ask clarifying questions when the request is ambiguous
- Synthesize information rather than just retrieving it
- Consider ethical implications of your responses
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2  # Slightly higher for more creative reasoning
)

# Initialize global API clients for efficiency
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
arxiv_client = arxiv.Client()

@tool
def mathematical_calculator(expression: str, analysis_type: str = "basic") -> Dict[str, Any]:
    """
    Mathematical calculator with symbolic computation, statistics, and calculus analysis.
    
    Args:
        expression: Mathematical expression or data to analyze
        analysis_type: Type of analysis - "basic", "symbolic", "statistical", "differential", "integral"
    """
    try:
        result = {}
        
        if analysis_type == "basic":
            # Basic arithmetic using sympy for security
            expression = expression.replace("^", "**")
            expr = sp.sympify(expression)
            result["value"] = float(expr.evalf())
            result["type"] = "numerical"
            
        elif analysis_type == "symbolic":
            # Symbolic mathematics using SymPy
            x, y, z = sp.symbols('x y z')
            expr = sp.sympify(expression)
            result["expression"] = str(expr)
            result["simplified"] = str(sp.simplify(expr))
            result["expanded"] = str(sp.expand(expr))
            result["factored"] = str(sp.factor(expr))
            result["type"] = "symbolic"
            
        elif analysis_type == "statistical":
            # Statistical analysis
            try:
                # Safer evaluation with restricted environment
                import ast
                data = ast.literal_eval(expression)  # Safer than eval()
            except (ValueError, SyntaxError):
                return {
                    "text_for_llm": "Error: Invalid data format. Please use format like [1,2,3,4,5]",
                    "metadata": {
                        "status": "error"
                    }
                }
            
            if isinstance(data, (list, tuple)) and all(isinstance(x, (int, float)) for x in data):
                if len(data) == 0:
                    return {
                        "text_for_llm": "Error: Cannot perform statistical analysis on empty data",
                        "metadata": {
                            "status": "error"
                        }
                    }
                result["mean"] = statistics.mean(data)
                result["median"] = statistics.median(data)
                try:
                    result["mode"] = statistics.mode(data)
                except statistics.StatisticsError:
                    result["mode"] = "No unique mode"
                result["std_dev"] = statistics.stdev(data) if len(data) > 1 else 0
                result["variance"] = statistics.variance(data) if len(data) > 1 else 0
                result["min"] = min(data)
                result["max"] = max(data)
                result["range"] = max(data) - min(data)
                result["count"] = len(data)
                result["type"] = "statistical"
            else:
                return {
                    "text_for_llm": "Error: Statistical analysis requires a list of numbers like [1,2,3,4,5]",
                    "metadata": {
                        "status": "error"
                    }
                }
                
        elif analysis_type == "differential":
            # Differential calculus
            expr = sp.sympify(expression)
            variables = expr.free_symbols
            
            if len(variables) != 1:
                return {
                    "text_for_llm": "Error: Calculus operations require exactly one variable in the expression.",
                    "metadata": {
                        "status": "error"
                    }
                }
            
            variable = variables.pop()
            result["derivative"] = str(sp.diff(expr, variable))
            result["second_derivative"] = str(sp.diff(expr, variable, 2))
            result["critical_points"] = str(sp.solve(sp.diff(expr, variable), variable))
            result["type"] = "differential"
            
        elif analysis_type == "integral":
            # Integral calculus
            expr = sp.sympify(expression)
            variables = expr.free_symbols
            
            if len(variables) != 1:
                return {
                    "text_for_llm": "Error: Calculus operations require exactly one variable in the expression.",
                    "metadata": {
                        "status": "error"
                    }
                }
            
            variable = variables.pop()
            result["indefinite_integral"] = str(sp.integrate(expr, variable))
            result["type"] = "integral"
        
        return {
            "text_for_llm": json.dumps(result, indent=2),
            "metadata": {
                "status": "success",
                "analysis_type": analysis_type
            }
        }
        
    except Exception as e:
        return {
            "text_for_llm": f"Error in calculation: {str(e)}\nTip: For statistical analysis, use format [1,2,3,4,5]. For symbolic math, use variables like 'x**2 + 2*x + 1'",
            "metadata": {
                "status": "error"
            }
        }

@tool
def web_search(query: str, search_type: str = "general", max_results: int = 5) -> Dict[str, Any]:
    """
    Intelligent web search using Tavily API with source analysis and credibility scoring.
    
    Args:
        query: Search query
        search_type: Type of search - "general", "news", "academic", "fact_check"
        max_results: Maximum number of results to return
    """
    try:
        
        # Configure search parameters based on search type
        search_params = {
            "query": query,
            "max_results": max_results,
            "include_answer": True,
            "include_raw_content": False
        }
        
        # Enhance query and parameters based on search type
        if search_type == "news":
            search_params["query"] = f"recent news {query}"
            search_params["days"] = 7  # Recent news from last 7 days
        elif search_type == "academic":
            search_params["query"] = f"{query} academic research paper study"
        elif search_type == "fact_check":
            search_params["query"] = f"fact check {query}"
        
        # Perform the search
        response = tavily_client.search(**search_params)
        
        if not response.get('results'):
            return {
                "text_for_llm": "No search results found.",
                "metadata": {
                    "average_credibility": 0.0
                }
            }
        
        results = response['results']
        formatted_results = []
        credibility_scores = []
        
        for i, result in enumerate(results, 1):
            # Extract domain and calculate credibility
            url = result.get('url', '')
            domain = url.split('/')[2] if '/' in url else 'unknown'
            credibility = calculate_domain_credibility(domain)
            credibility_scores.append(credibility)
            
            # Get relevance score from Tavily (if available)
            relevance_score = result.get('score', 0.5) * 10  # Convert to 0-10 scale
            
            formatted_results.append(
                f"{i}. **{result.get('title', 'No title')}** (Credibility: {credibility:.1f}/10, Relevance: {relevance_score:.1f}/10)\n"
                f"   URL: {url}\n"
                f"   Summary: {result.get('content', 'No summary available')[:300]}...\n"
                f"   Domain: {domain}\n"
                f"   Published: {result.get('published_date', 'Unknown date')}\n"
            )
        
        # Add Tavily's answer if available
        tavily_answer = response.get('answer', '')
        
        avg_credibility = sum(credibility_scores) / len(credibility_scores) if credibility_scores else 0
        summary = f"**Tavily Search Results:**\n"
        summary += f"Query: {query}\n"
        summary += f"Search type: {search_type}\n"
        summary += f"Average source credibility: {avg_credibility:.1f}/10\n"
        summary += f"Results found: {len(results)}\n"
        
        if tavily_answer:
            summary += f"\n**AI-Generated Answer:**\n{tavily_answer}\n"
        
        summary += "\n**Detailed Results:**\n"
        
        text_for_llm = summary + "\n".join(formatted_results)
        
        return {
            "text_for_llm": text_for_llm,
            "metadata": {
                "average_credibility": avg_credibility / 10.0
            }
        }
        
    except Exception as e:
        return {
            "text_for_llm": f"Error in Tavily web search: {str(e)}\nTip: Make sure TAVILY_API_KEY is set in your environment variables.",
            "metadata": {
                "average_credibility": 0.0
            }
        }

def calculate_domain_credibility(domain: str) -> float:
    """Calculate credibility score for a domain (0-10 scale)."""
    registered_domain = tldextract.extract(domain).registered_domain
    
    high_credibility = {'reuters.com', 'bbc.com', 'nature.com', 'science.org', 'nih.gov', 'who.int', 'nasa.gov'}
    medium_credibility = {'wikipedia.org', 'scholar.google.com', 'researchgate.net', 'cnn.com', 'npr.org'}
    low_credibility = {'reddit.com', 'quora.com', 'yahoo.com'}
    
    if registered_domain in high_credibility:
        return 9.0  # ✅ Fixed: Direct high credibility score
    elif registered_domain in medium_credibility:
        return 7.0  # ✅ Fixed: Direct medium credibility score
    elif registered_domain in low_credibility:
        return 4.0
    elif domain.endswith('.edu') or domain.endswith('.gov'):
        return 9.0
    elif domain.endswith('.org'):
        return 7.0
    else:
        return 5.5  # Default score for unknown domains

@tool
def arxiv_search(query: str, field: str = "all", max_results: int = 5, sort_by: str = "relevance") -> Dict[str, Any]:
    """
    ArXiv search with field filtering and relevance analysis.
    
    Args:
        query: Search query for academic papers
        field: Field to search in - "all", "cs", "physics", "math", "bio", "econ"
        max_results: Maximum number of results
        sort_by: Sort by "relevance", "date", "citations"
    """
    try:
        # Field-specific query enhancement
        field_prefixes = {
            "cs": "cat:cs.*",
            "physics": "cat:physics.*",
            "math": "cat:math.*",
            "bio": "cat:q-bio.*",
            "econ": "cat:econ.*"
        }
        
        if field != "all" and field in field_prefixes:
            query = f"{field_prefixes[field]} AND {query}"
        
        sort_criterion = arxiv.SortCriterion.Relevance
        if sort_by == "date":
            sort_criterion = arxiv.SortCriterion.SubmittedDate
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion
        )
        
        results = list(arxiv_client.results(search))
        
        if not results:
            return {
                "text_for_llm": "No arXiv papers found.",
                "metadata": {
                    "average_relevance": 0.0
                }
            }
        
        formatted_results = []
        relevance_scores = []
        for i, paper in enumerate(results, 1):
            authors = ", ".join([author.name for author in paper.authors[:3]])  # Limit authors
            if len(paper.authors) > 3:
                authors += f" et al. ({len(paper.authors)} total)"
            
            # Calculate relevance score based on query terms
            relevance = calculate_paper_relevance(paper, query)
            relevance_scores.append(relevance)
            
            formatted_results.append(
                f"{i}. **{paper.title}** (Relevance: {relevance:.1f}/10)\n"
                f"   Authors: {authors}\n"
                f"   Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"   Category: {', '.join(paper.categories[:2])}\n"
                f"   URL: {paper.entry_id}\n"
                f"   Abstract: {paper.summary[:250]}...\n"
            )
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        summary = f"**arXiv Search Results for '{query}':**\n"
        summary += f"Field: {field}, Sort: {sort_by}, Results: {len(results)}\n\n"
        
        text_for_llm = summary + "\n".join(formatted_results)
        
        return {
            "text_for_llm": text_for_llm,
            "metadata": {
                "average_relevance": avg_relevance / 10.0
            }
        }
        
    except Exception as e:
        return {
            "text_for_llm": f"Error in arXiv search: {str(e)}",
            "metadata": {
                "average_relevance": 0.0
            }
        }

def calculate_paper_relevance(paper, query: str) -> float:
    """Calculate relevance score for a paper based on query terms."""
    query_terms = query.lower().split()
    title_matches = sum(1 for term in query_terms if term in paper.title.lower())
    abstract_matches = sum(1 for term in query_terms if term in paper.summary.lower())
    
    title_score = (title_matches / len(query_terms)) * 6  # Title matches worth more
    abstract_score = (abstract_matches / len(query_terms)) * 4
    
    return min(10.0, title_score + abstract_score)

@tool
def pubmed_search(query: str, publication_type: str = "all", max_results: int = 5) -> Dict[str, Any]:
    """
    PubMed search with publication type filtering and impact analysis.
    
    Args:
        query: Search query for medical/life science papers
        publication_type: Type of publication - "all", "clinical_trial", "review", "meta_analysis"
        max_results: Maximum number of results
    """
    try:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        # Enhance query based on publication type
        if publication_type == "clinical_trial":
            query += " AND clinical trial[pt]"
        elif publication_type == "review":
            query += " AND review[pt]"
        elif publication_type == "meta_analysis":
            query += " AND meta-analysis[pt]"
        
        # Search for article IDs
        search_url = f"{base_url}esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance"
        }
        
        search_response = requests.get(search_url, params=search_params, timeout=10)
        search_data = search_response.json()
        
        if "esearchresult" not in search_data or not search_data["esearchresult"]["idlist"]:
            return {
                "text_for_llm": "No PubMed articles found.",
                "metadata": {
                    "average_impact": 0.0
                }
            }
        
        ids = search_data["esearchresult"]["idlist"]
        
        # Fetch article details
        fetch_url = f"{base_url}esummary.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json"
        }
        
        fetch_response = requests.get(fetch_url, params=fetch_params, timeout=10)
        fetch_data = fetch_response.json()
        
        formatted_results = []
        impact_scores = []
        
        for i, (pmid, article) in enumerate(fetch_data["result"].items(), 1):
            if pmid == "uids":
                continue
                
            title = article.get("title", "No title")
            authors = ", ".join([author["name"] for author in article.get("authors", [])[:3]])
            if len(article.get("authors", [])) > 3:
                authors += f" et al."
            
            journal = article.get("fulljournalname", "Unknown journal")
            pub_date = article.get("pubdate", "Unknown date")
            pub_types = article.get("pubtypes", [])
            
            # Calculate impact score based on journal and publication type
            impact_score = calculate_publication_impact(journal, pub_types)
            impact_scores.append(impact_score)
            
            formatted_results.append(
                f"{i}. **{title}** (Impact: {impact_score:.1f}/10)\n"
                f"   Authors: {authors}\n"
                f"   Journal: {journal}\n"
                f"   Publication Type: {', '.join(pub_types)}\n"
                f"   Published: {pub_date}\n"
                f"   PMID: {pmid}\n"
                f"   URL: https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n"
            )
        
        avg_impact = sum(impact_scores) / len(impact_scores) if impact_scores else 0
        summary = f"**PubMed Search Results:**\n"
        summary += f"Query: {query}\n"
        summary += f"Publication type: {publication_type}\n"
        summary += f"Average impact score: {avg_impact:.1f}/10\n"
        summary += f"Results found: {len(formatted_results)}\n\n"
        
        text_for_llm = summary + "\n".join(formatted_results)
        
        return {
            "text_for_llm": text_for_llm,
            "metadata": {
                "average_impact": avg_impact / 10.0
            }
        }
        
    except Exception as e:
        return {
            "text_for_llm": f"Error in PubMed search: {str(e)}",
            "metadata": {
                "average_impact": 0.0
            }
        }

def calculate_publication_impact(journal: str, pub_types: List[str]) -> float:
    """Calculate impact score for a publication."""
    high_impact_journals = ['nature', 'science', 'cell', 'nejm', 'lancet', 'jama']
    medium_impact_journals = ['plos', 'bmc', 'frontiers', 'scientific reports']
    
    base_score = 5.0
    
    # Journal impact
    journal_lower = journal.lower()
    if any(hj in journal_lower for hj in high_impact_journals):
        base_score += 3.0
    elif any(mj in journal_lower for mj in medium_impact_journals):
        base_score += 1.5
    
    # Publication type impact
    for pub_type in pub_types:
        if 'meta-analysis' in pub_type.lower():
            base_score += 2.0
        elif 'clinical trial' in pub_type.lower():
            base_score += 1.5
        elif 'review' in pub_type.lower():
            base_score += 1.0
    
    return min(10.0, base_score)

# Create tools list
tools = [
    mathematical_calculator,
    web_search,
    arxiv_search,
    pubmed_search
]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# Create tool node first to avoid circular dependency
tool_node = ToolNode(tools)

# Tool node that captures confidence scores
def tool_node_with_confidence(state: AgentState) -> AgentState:
    """Tool node that extracts confidence scores from tool results."""
    # Execute tools using the standard tool node
    tool_result = tool_node.invoke(state)
    
    # Extract confidence scores from the actual tool execution results
    if "messages" in tool_result and tool_result["messages"]:
        confidence_scores = state.get("confidence_scores", {})
        
        # Process each tool result message to extract confidence scores
        for message in tool_result["messages"]:
            if hasattr(message, 'content'):
                content = message.content
                
                # ✅ LangGraph ToolNode returns string content, need to parse if it was JSON
                if isinstance(content, str):
                    try:
                        # Try to parse as JSON in case tools returned structured data
                        parsed_content = json.loads(content)
                        if isinstance(parsed_content, dict):
                            content = parsed_content
                    except (json.JSONDecodeError, ValueError):
                        # Content is just a regular string, analyze for error patterns
                        if "Error" not in content and ("value" in content or "result" in content):
                            confidence_scores['mathematical'] = 0.95
                        elif "Error" in content:
                            confidence_scores['mathematical'] = 0.1
                        continue
                
                # Check if content is a dictionary (from our structured tools)
                if isinstance(content, dict):
                    metadata = content.get("metadata", {})
                    
                    # Extract confidence scores from metadata
                    if "average_credibility" in metadata:
                        # ✅ Ensure scores are in 0-1 range
                        score = metadata["average_credibility"]
                        confidence_scores['web_search'] = max(0.0, min(1.0, score))
                    elif "average_relevance" in metadata:
                        score = metadata["average_relevance"] 
                        confidence_scores['arxiv_search'] = max(0.0, min(1.0, score))
                    elif "average_impact" in metadata:
                        score = metadata["average_impact"]
                        confidence_scores['pubmed_search'] = max(0.0, min(1.0, score))
                    elif "status" in metadata:
                        # Handle calculator tool results
                        if metadata["status"] == "success":
                            confidence_scores['mathematical'] = 0.95
                        elif metadata["status"] == "error":
                            confidence_scores['mathematical'] = 0.1
                    
                    # Replace dictionary content with clean text for the LLM
                    message.content = content.get("text_for_llm", str(content))
        
        tool_result["confidence_scores"] = confidence_scores
    
    # Preserve other state fields that might not be in tool_result
    tool_result["reasoning_chain"] = state.get("reasoning_chain", [])
    tool_result["memory"] = state.get("memory", {})
    tool_result["context"] = state.get("context", {})
    tool_result["tool_calls"] = state.get("tool_calls", [])
    
    return tool_result

def should_continue(state: AgentState) -> str:
    """Decision logic for conversation flow."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message has tool calls, continue to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Check for infinite loop prevention
    reasoning_chain = state.get("reasoning_chain", [])
    agent_loops = len([step for step in reasoning_chain if "Model response" in step])
    
    # Prevent infinite loops - max 3 iterations (consistent limit)
    if agent_loops >= 3:
        return END
    
    # Only consider looping back if we have confidence scores and they're low
    # AND we haven't exceeded our iteration limit
    confidence_scores = state.get("confidence_scores", {})
    if confidence_scores and agent_loops < 3:  # ✅ Consistent with above limit
        # ✅ Safe division - check for empty dict
        if len(confidence_scores) > 0:
            avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
            
            # Loop back if confidence is low and we have tools available to improve it
            if avg_confidence < 0.5:  # More reasonable threshold
                return "agent"
    
    return END

def call_model(state: AgentState) -> AgentState:
    """Model calling with system prompt and context awareness."""
    messages = state["messages"]
    
    # Create messages with system context
    messages_with_context = []
    
    # Always add the system message first
    messages_with_context.append(SystemMessage(content=SYSTEM_PROMPT))
    
    # Add context from memory if available
    if "memory" in state and state["memory"]:
        context_content = f"Context from previous interactions: {json.dumps(state['memory'], indent=2)}"
        messages_with_context.append(SystemMessage(content=context_content))
    
    # Add all user/assistant messages (skip any existing system messages to avoid duplication)
    for msg in messages:
        if not isinstance(msg, SystemMessage):
            messages_with_context.append(msg)
    
    response = llm_with_tools.invoke(messages_with_context)
    
    # Update reasoning chain
    reasoning_chain = state.get("reasoning_chain", [])
    reasoning_chain.append(f"Model response at {datetime.now().isoformat()}")
    
    # Track tool calls if present (avoid duplication)
    tool_calls = state.get("tool_calls", [])
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tc in response.tool_calls:
            # Handle both dict and object-style tool calls
            if hasattr(tc, 'name'):
                tool_name = tc.name
            elif isinstance(tc, dict):
                tool_name = tc.get('name', 'unknown')
            else:
                tool_name = str(tc)
            
            # Only add if not already present (avoid duplicates)
            if not any(call.get("name") == tool_name for call in tool_calls):
                tool_calls.append({
                    "name": tool_name,
                    "timestamp": datetime.now().isoformat()
                })
    
    # Get existing confidence scores
    confidence_scores = state.get("confidence_scores", {})
    
    return {
        "messages": [response],  # ✅ Keep as single response - LangGraph handles message accumulation
        "reasoning_chain": reasoning_chain,
        "confidence_scores": confidence_scores,
        "tool_calls": tool_calls,
        "memory": state.get("memory", {}),  # Preserve memory
        "context": state.get("context", {})  # Preserve context
    }

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node_with_confidence)

# Set entry point
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "agent": "agent",  # Allow looping back for follow-up
        END: END
    }
)

# Add edge from tools back to agent
workflow.add_edge("tools", "agent")

# Compile the graph
research_graph = workflow.compile()

class ResearchAgent:
    """Research agent with memory, reasoning, and multi-source capabilities."""
    
    def __init__(self):
        self.graph = research_graph
        self.session_memory = {}
        self.interaction_history = []
    
    def run(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Run the research agent with user input and return detailed response."""
        try:
            # Create initial agent state
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "tool_calls": [],
                "memory": self.session_memory,
                "context": context or {},
                "reasoning_chain": [],
                "confidence_scores": {}
            }
            
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            # Get the last assistant message
            last_message = result["messages"][-1]
            
            # Update session memory
            self.session_memory.update({
                "last_query": user_input,
                "timestamp": datetime.now().isoformat(),
                "tools_used": [tc.get("name", "") for tc in result.get("tool_calls", [])]
            })
            
            # Store interaction
            self.interaction_history.append({
                "input": user_input,
                "output": last_message.content,
                "timestamp": datetime.now().isoformat(),
                "reasoning_chain": result.get("reasoning_chain", [])
            })
            
            return {
                "response": last_message.content,
                "reasoning_chain": result.get("reasoning_chain", []),
                "confidence_scores": result.get("confidence_scores", {}),
                "tools_used": result.get("tool_calls", []),
                "session_memory": self.session_memory
            }
            
        except Exception as e:
            error_response = {
                "response": f"I encountered an error: {str(e)}. Let me try a different approach.",
                "reasoning_chain": [f"Error occurred: {str(e)}"],
                "confidence_scores": {"error": 0.0},
                "tools_used": [],
                "session_memory": self.session_memory
            }
            return error_response
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation so far."""
        if not self.interaction_history:
            return "No interactions yet."
        
        summary = f"**Conversation Summary** ({len(self.interaction_history)} interactions)\n\n"
        for i, interaction in enumerate(self.interaction_history[-5:], 1):  # Last 5 interactions
            summary += f"{i}. **Query**: {interaction['input'][:100]}...\n"
            summary += f"   **Response**: {interaction['output'][:100]}...\n"
            summary += f"   **Time**: {interaction['timestamp']}\n\n"
        
        return summary
    
    def clear_memory(self):
        """Clear session memory and interaction history."""
        self.session_memory = {}
        self.interaction_history = []

# Create global research agent instance
research_agent = ResearchAgent()

if __name__ == "__main__":
    # Test the research agent
    test_queries = [
        "Calculate the derivative of x^3 + 2x^2 + 1 and find its critical points",
        "Search for recent advances in quantum computing and analyze the credibility of sources",
        "Find the latest papers on CRISPR gene editing in cancer treatment from both arXiv and PubMed",
        "Analyze the statistical significance of this data: [1.2, 1.5, 1.1, 1.8, 1.3, 1.6, 1.4, 1.7, 1.2, 1.5]"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        result = research_agent.run(query)
        print(f"Response: {result['response']}")
        if result['reasoning_chain']:
            print(f"\nReasoning Chain: {result['reasoning_chain']}")
        print(f"\nConfidence Scores: {result['confidence_scores']}")
        print("-" * 80)
