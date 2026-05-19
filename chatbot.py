"""
============================================================
chatbot.py — Interactive Finance AI Chatbot
============================================================
Handles:
1. Semantic search against Pinecone
2. RAG prompt construction with context + memory
3. Streaming LLM response generation
4. Interactive CLI with commands
============================================================
"""

import logging
import time

from google import genai
from google.genai import types
from pinecone import Pinecone

from config import (
    GOOGLE_API_KEY, PINECONE_API_KEY,
    LLM_MODEL, EMBEDDING_MODEL, EMBEDDING_DIMENSION,
    PINECONE_INDEX_NAME, PINECONE_NAMESPACE,
    DEFAULT_TOP_K, MAX_MEMORY_TURNS,
)
from utils import (
    retry_with_backoff, Colors,
    print_header, print_separator, print_success, print_error,
    print_info, print_score, print_bot_prefix, print_user_prefix,
    format_timestamp,
)

logger = logging.getLogger("rag_finance")


# ════════════════════════════════════════════════════════════
# SAMPLE FINANCE QUESTIONS (for testing)
# ════════════════════════════════════════════════════════════

SAMPLE_QUESTIONS = [
    "What is investment banking and how do IPOs work?",
    "Explain Value at Risk (VaR) in risk management.",
    "How do credit default swaps work?",
    "What are the key metrics in fundamental stock analysis?",
    "How does an ETF differ from a mutual fund?",
    "How do FICO credit scores get calculated?",
    "What machine learning techniques are used in fraud detection?",
    "Explain Modern Portfolio Theory and the efficient frontier.",
    "What is WACC and why is it important in corporate finance?",
    "How does blockchain technology apply to financial services?",
]


# ════════════════════════════════════════════════════════════
# SECTION 1: Semantic Search
# ════════════════════════════════════════════════════════════

@retry_with_backoff(max_retries=3)
def _embed_query(client: genai.Client, query: str) -> list[float]:
    """Embed a single query string using Gemini embedding model (with retry)."""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION,
        ),
    )
    return result.embeddings[0].values


def semantic_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Perform semantic similarity search against Pinecone.
    
    1. Embed the query using Gemini embedding model
    2. Query Pinecone for top-K most similar vectors
    3. Return matches with scores and metadata
    
    Args:
        query: User's search query.
        top_k: Number of top results to retrieve.
    
    Returns:
        List of dicts with keys: score, text, topic, chunk_index, source.
    """
    logger.info("Semantic search: '%s' (top_k=%d)", query[:80], top_k)

    # Initialize clients
    gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    # Step 1: Embed the query
    query_embedding = _embed_query(gemini_client, query)

    # Step 2: Query Pinecone with cosine similarity
    results = index.query(
        namespace=PINECONE_NAMESPACE,
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    # Step 3: Extract and format matches
    matches = []
    for match in results.matches:
        matches.append({
            "score": match.score,
            "text": match.metadata.get("text", ""),
            "topic": match.metadata.get("topic", "Unknown"),
            "chunk_index": match.metadata.get("chunk_index", -1),
            "source": match.metadata.get("source", "unknown"),
        })

    logger.info("Retrieved %d matches (top score: %.4f)",
                len(matches), matches[0]["score"] if matches else 0.0)
    return matches


# ════════════════════════════════════════════════════════════
# SECTION 2: RAG Prompt Construction
# ════════════════════════════════════════════════════════════

def build_prompt(query: str, context_chunks: list[dict], conversation_history: list[dict]) -> str:
    """
    Build a structured RAG prompt combining:
    - System instructions (finance expert persona)
    - Retrieved context chunks
    - Conversation history (memory)
    - Current user question
    
    Args:
        query: Current user question.
        context_chunks: Retrieved chunks from Pinecone.
        conversation_history: List of past {role, content} dicts.
    
    Returns:
        str: Fully constructed prompt for the LLM.
    """
    # System prompt establishing the AI's role and behavior
    system_prompt = (
        "You are an expert Finance AI Assistant with deep knowledge in investment banking, "
        "risk management, financial derivatives, stock market analysis, mutual funds, "
        "credit scoring, fraud detection, wealth management, portfolio optimization, "
        "corporate finance, financial compliance, and FinTech systems.\n\n"
        "INSTRUCTIONS:\n"
        "- Answer questions accurately using ONLY the provided context below.\n"
        "- If the context doesn't contain enough information, say so honestly.\n"
        "- Provide detailed, professional explanations with examples when relevant.\n"
        "- Use financial terminology appropriately.\n"
        "- Reference specific concepts from the context to support your answers.\n"
        "- Keep answers well-structured with clear formatting.\n"
    )

    # Format retrieved context
    context_section = "\n═══ RETRIEVED CONTEXT ═══\n"
    for i, chunk in enumerate(context_chunks, 1):
        context_section += f"\n[Source {i} — {chunk['topic']}]\n{chunk['text']}\n"
    context_section += "\n═══ END CONTEXT ═══\n"

    # Format conversation history (memory)
    history_section = ""
    if conversation_history:
        history_section = "\n═══ CONVERSATION HISTORY ═══\n"
        for turn in conversation_history:
            role = "User" if turn["role"] == "user" else "Assistant"
            # Truncate long history entries to save tokens
            content = turn["content"][:300]
            history_section += f"{role}: {content}\n"
        history_section += "═══ END HISTORY ═══\n"

    # Combine everything
    full_prompt = (
        f"{system_prompt}\n"
        f"{context_section}\n"
        f"{history_section}\n"
        f"Current Question: {query}\n\n"
        f"Please provide a comprehensive, well-structured answer based on the context above."
    )

    return full_prompt


# ════════════════════════════════════════════════════════════
# SECTION 3: Streaming Response Generation (with fallback)
# ════════════════════════════════════════════════════════════

# Ordered list of models to try — if one is overloaded, try the next
FALLBACK_MODELS = [
    LLM_MODEL,           # primary: gemini-2.5-flash-lite
    "gemini-2.5-flash",  # fallback 1
    "gemini-2.0-flash",  # fallback 2
]


def generate_response_stream(prompt: str):
    """
    Generate a streaming response from Gemini LLM.
    Tries multiple models with retry if the primary is unavailable (503).
    
    Args:
        prompt: The complete RAG prompt.
    
    Yields:
        str: Text chunks as they arrive from the model.
    """
    client = genai.Client(api_key=GOOGLE_API_KEY)

    for model_name in FALLBACK_MODELS:
        for attempt in range(2):  # 2 attempts per model
            try:
                logger.info("Trying model=%s (attempt %d)", model_name, attempt + 1)
                response_stream = client.models.generate_content_stream(
                    model=model_name,
                    contents=prompt,
                )

                yielded = False
                for chunk in response_stream:
                    if chunk.text:
                        yielded = True
                        yield chunk.text

                if yielded:
                    return  # success — stop trying other models

            except Exception as e:
                logger.warning("Model %s attempt %d failed: %s", model_name, attempt + 1, str(e))
                time.sleep(1)  # brief pause before retry
                continue

    # All models failed — yield nothing (caller handles fallback)
    logger.error("All LLM models failed after retries.")
    return


def _format_context_answer(query: str, matches: list[dict]) -> str:
    """
    Build a clean formatted answer directly from retrieved context chunks.
    Used as a fallback when the LLM is unavailable.
    """
    lines = []
    lines.append(f"Based on the knowledge base, here is what I found about \"{query}\":\n")
    for i, match in enumerate(matches, 1):
        text = match["text"].strip()
        topic = match["topic"]
        lines.append(f"📌 [{topic}]")
        lines.append(f"   {text}\n")
    lines.append("(⚠️ This is a direct excerpt — the AI summarization model is temporarily unavailable.)")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# SECTION 4: Chat Session Manager
# ════════════════════════════════════════════════════════════

class ChatSession:
    """
    Manages an interactive chat session with conversation memory.
    
    Features:
    - Tracks conversation history (configurable max turns)
    - Displays retrieval scores and source text
    - Supports streaming responses
    - Provides CLI commands (help, clear, history, sample, quit)
    """

    def __init__(self, top_k: int = DEFAULT_TOP_K):
        self.top_k = top_k
        self.conversation_history: list[dict] = []
        self.turn_count = 0
        logger.info("Chat session initialized (top_k=%d, max_memory=%d)", top_k, MAX_MEMORY_TURNS)

    def add_to_history(self, role: str, content: str):
        """Add a turn to conversation history, trimming if needed."""
        self.conversation_history.append({"role": role, "content": content})
        # Keep only the last MAX_MEMORY_TURNS turns
        if len(self.conversation_history) > MAX_MEMORY_TURNS * 2:
            self.conversation_history = self.conversation_history[-(MAX_MEMORY_TURNS * 2):]

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.turn_count = 0
        print_success("Conversation history cleared!")

    def show_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            print_info("No conversation history yet.")
            return

        print_separator("─", 60, Colors.CYAN)
        print(f"  {Colors.BOLD}{Colors.CYAN}📝 Conversation History ({len(self.conversation_history)} messages){Colors.RESET}")
        print_separator("─", 60, Colors.CYAN)

        for turn in self.conversation_history:
            role_icon = "👤" if turn["role"] == "user" else "🤖"
            role_color = Colors.CYAN if turn["role"] == "user" else Colors.GREEN
            preview = turn["content"][:150].replace("\n", " ")
            print(f"  {role_icon} {role_color}{preview}...{Colors.RESET}")

        print_separator("─", 60, Colors.CYAN)

    def show_sample_questions(self):
        """Display sample finance questions for testing."""
        print_separator("─", 60, Colors.YELLOW)
        print(f"  {Colors.BOLD}{Colors.YELLOW}💡 Sample Finance Questions{Colors.RESET}")
        print_separator("─", 60, Colors.YELLOW)

        for i, q in enumerate(SAMPLE_QUESTIONS, 1):
            print(f"  {Colors.DIM}{i:2d}.{Colors.RESET} {q}")

        print_separator("─", 60, Colors.YELLOW)
        print(f"  {Colors.DIM}Copy any question above and paste it as your query.{Colors.RESET}\n")

    def process_query(self, query: str):
        """
        Process a user query through the full RAG pipeline:
        1. Semantic search for relevant chunks
        2. Display retrieval results with scores
        3. Build RAG prompt with context + memory
        4. Stream LLM response
        5. Update conversation history
        """
        self.turn_count += 1

        # ── Step 1: Retrieve relevant chunks ──────────────
        print(f"\n  {Colors.DIM}🔍 Searching knowledge base...{Colors.RESET}")

        try:
            matches = semantic_search(query, self.top_k)
        except Exception as e:
            print_error(f"Retrieval failed: {e}")
            logger.error("Retrieval failed: %s", str(e), exc_info=True)
            return

        if not matches:
            print_info("No relevant documents found. Try rephrasing your question.")
            return

        # ── Step 2: Display retrieval scores ──────────────
        print(f"\n  {Colors.BOLD}📊 Retrieved {len(matches)} relevant chunks:{Colors.RESET}")
        print_separator("─", 60, Colors.DIM)

        for rank, match in enumerate(matches, 1):
            print_score(rank, match["score"], match["text"], match["topic"])

        print_separator("─", 60, Colors.DIM)

        # ── Step 3: Build RAG prompt ──────────────────────
        prompt = build_prompt(query, matches, self.conversation_history)

        # ── Step 4: Stream response ───────────────────────
        print_bot_prefix()

        full_response = ""
        try:
            for text_chunk in generate_response_stream(prompt):
                print(text_chunk, end="", flush=True)
                full_response += text_chunk
        except Exception as e:
            logger.error("Generation failed: %s", str(e), exc_info=True)

        # If LLM produced no output, show formatted context as fallback
        if not full_response.strip():
            full_response = _format_context_answer(query, matches)
            print(full_response)

        print()  # newline after streaming

        # ── Step 5: Update memory ─────────────────────────
        self.add_to_history("user", query)
        self.add_to_history("assistant", full_response)

        # Show timestamp
        print(f"  {Colors.DIM}⏱️  {format_timestamp()} │ Turn #{self.turn_count}{Colors.RESET}")


# ════════════════════════════════════════════════════════════
# SECTION 5: Interactive CLI Chatbot
# ════════════════════════════════════════════════════════════

def show_help():
    """Display available commands."""
    print_separator("─", 60, Colors.BLUE)
    print(f"  {Colors.BOLD}{Colors.BLUE}📋 Available Commands{Colors.RESET}")
    print_separator("─", 60, Colors.BLUE)
    print(f"  {Colors.CYAN}help{Colors.RESET}      — Show this help menu")
    print(f"  {Colors.CYAN}sample{Colors.RESET}    — Show sample finance questions")
    print(f"  {Colors.CYAN}history{Colors.RESET}   — View conversation history")
    print(f"  {Colors.CYAN}clear{Colors.RESET}     — Clear conversation history")
    print(f"  {Colors.CYAN}topk N{Colors.RESET}    — Set top-K retrieval (e.g., topk 3)")
    print(f"  {Colors.CYAN}quit{Colors.RESET}      — Exit the chatbot")
    print_separator("─", 60, Colors.BLUE)


def run_chatbot():
    """
    Launch the interactive finance chatbot CLI.
    Supports natural language queries and special commands.
    """
    print_header("Finance AI Chatbot", "Powered by Gemini + Pinecone RAG")

    print(f"  {Colors.WHITE}Ask any finance question and get AI-powered answers{Colors.RESET}")
    print(f"  {Colors.WHITE}backed by semantic search over a curated knowledge base.{Colors.RESET}")
    print(f"  {Colors.DIM}Type 'help' for commands or 'sample' for example questions.{Colors.RESET}\n")
    print_separator("═", 60, Colors.CYAN)

    session = ChatSession()
    user_prompt = print_user_prefix()

    while True:
        try:
            query = input(user_prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  {Colors.YELLOW}👋 Goodbye!{Colors.RESET}\n")
            break

        if not query:
            continue

        # ── Handle commands ───────────────────────────────
        cmd = query.lower()

        if cmd in ("quit", "exit", "q"):
            print(f"\n  {Colors.YELLOW}👋 Thanks for using Finance AI Chatbot! Goodbye!{Colors.RESET}\n")
            break
        elif cmd == "help":
            show_help()
            continue
        elif cmd == "sample":
            session.show_sample_questions()
            continue
        elif cmd == "history":
            session.show_history()
            continue
        elif cmd == "clear":
            session.clear_history()
            continue
        elif cmd.startswith("topk"):
            try:
                new_k = int(cmd.split()[1])
                session.top_k = max(1, min(new_k, 20))
                print_success(f"Top-K set to {session.top_k}")
            except (IndexError, ValueError):
                print_error("Usage: topk N (e.g., topk 3)")
            continue

        # ── Process finance query ─────────────────────────
        session.process_query(query)


if __name__ == "__main__":
    from utils import setup_logging
    setup_logging()
    from config import validate_env
    validate_env()
    run_chatbot()
