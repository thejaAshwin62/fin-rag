"""
============================================================
utils.py — Shared Utilities
============================================================
Provides logging setup, retry mechanism with exponential
backoff, terminal UI formatting helpers, and text hashing
for vector deduplication.
============================================================
"""

import os
import sys
import time
import hashlib
import logging
import functools
from datetime import datetime

from config import LOGS_DIR, LOG_FILE, MAX_RETRIES, RETRY_BASE_DELAY


# ════════════════════════════════════════════════════════════
# SECTION 1: Logging Setup
# ════════════════════════════════════════════════════════════

def setup_logging() -> logging.Logger:
    """
    Configure application-wide logging with both file and console handlers.
    
    - File handler: logs everything (DEBUG+) to logs/rag_finance.log
    - Console handler: logs warnings and above to terminal
    
    Returns:
        logging.Logger: Configured root logger instance.
    """
    # Create logs directory if it doesn't exist
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Create a named logger for the application
    logger = logging.getLogger("rag_finance")
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # Log format: timestamp | level | module | message
    log_format = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(module)-12s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ── File Handler (DEBUG+) ──────────────────────────────
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # ── Console Handler (WARNING+) ─────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    logger.info("Logging initialized — log file: %s", LOG_FILE)
    return logger


# ════════════════════════════════════════════════════════════
# SECTION 2: Retry Mechanism with Exponential Backoff
# ════════════════════════════════════════════════════════════

def retry_with_backoff(max_retries: int = MAX_RETRIES, base_delay: float = RETRY_BASE_DELAY):
    """
    Decorator that retries a function on failure with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds (doubles each retry).
    
    Usage:
        @retry_with_backoff(max_retries=3)
        def call_api():
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("rag_finance")
            last_exception = None

            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    delay = base_delay * (2 ** (attempt - 1))  # 1s, 2s, 4s, ...
                    logger.warning(
                        "Attempt %d/%d for %s failed: %s — retrying in %.1fs",
                        attempt, max_retries, func.__name__, str(e), delay
                    )
                    if attempt < max_retries:
                        time.sleep(delay)

            logger.error(
                "All %d attempts for %s failed. Last error: %s",
                max_retries, func.__name__, str(last_exception)
            )
            raise last_exception
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════
# SECTION 3: Text Hashing (Vector ID Deduplication)
# ════════════════════════════════════════════════════════════

def hash_text(text: str) -> str:
    """
    Generate a deterministic MD5 hash for a text string.
    Used as Pinecone vector IDs to prevent duplicate insertion.
    
    Args:
        text: Input text to hash.
    
    Returns:
        str: 32-character hexadecimal hash string.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ════════════════════════════════════════════════════════════
# SECTION 4: Terminal UI Formatting
# ════════════════════════════════════════════════════════════

# ANSI color codes for terminal output
class Colors:
    """ANSI escape codes for colorful terminal output."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ITALIC  = "\033[3m"
    
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    
    BG_BLUE   = "\033[44m"
    BG_GREEN  = "\033[42m"
    BG_YELLOW = "\033[43m"


def print_header(title: str, subtitle: str = ""):
    """Print a styled application header with box drawing characters."""
    width = 60
    print(f"\n{Colors.CYAN}{'═' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {'💰 ' + title:^{width - 4}}{Colors.RESET}")
    if subtitle:
        print(f"{Colors.DIM}{Colors.CYAN}  {subtitle:^{width - 4}}{Colors.RESET}")
    print(f"{Colors.CYAN}{'═' * width}{Colors.RESET}\n")


def print_separator(char: str = "─", length: int = 60, color: str = Colors.DIM):
    """Print a horizontal separator line."""
    print(f"{color}{char * length}{Colors.RESET}")


def print_success(message: str):
    """Print a success message in green."""
    print(f"  {Colors.GREEN}✅ {message}{Colors.RESET}")


def print_error(message: str):
    """Print an error message in red."""
    print(f"  {Colors.RED}❌ {message}{Colors.RESET}")


def print_warning(message: str):
    """Print a warning message in yellow."""
    print(f"  {Colors.YELLOW}⚠️  {message}{Colors.RESET}")


def print_info(message: str):
    """Print an info message in blue."""
    print(f"  {Colors.BLUE}ℹ️  {message}{Colors.RESET}")


def print_step(step_num: int, message: str):
    """Print a numbered step in the pipeline."""
    print(f"  {Colors.MAGENTA}[{step_num}]{Colors.RESET} {Colors.WHITE}{message}{Colors.RESET}")


def print_score(rank: int, score: float, text_preview: str, topic: str = ""):
    """Print a retrieval result with score and preview."""
    # Color-code score: green (high), yellow (medium), red (low)
    if score >= 0.8:
        score_color = Colors.GREEN
    elif score >= 0.6:
        score_color = Colors.YELLOW
    else:
        score_color = Colors.RED

    preview = text_preview[:120].replace("\n", " ")
    print(f"  {Colors.DIM}#{rank}{Colors.RESET}  "
          f"{score_color}Score: {score:.4f}{Colors.RESET}  "
          f"{Colors.DIM}│{Colors.RESET} "
          f"{Colors.ITALIC}{preview}...{Colors.RESET}")
    if topic:
        print(f"      {Colors.DIM}📂 Topic: {topic}{Colors.RESET}")


def print_bot_prefix():
    """Print the bot's response prefix."""
    print(f"\n  {Colors.GREEN}{Colors.BOLD}🤖 Finance AI:{Colors.RESET} ", end="", flush=True)


def print_user_prefix():
    """Print the user input prompt."""
    return f"\n  {Colors.CYAN}{Colors.BOLD}👤 You:{Colors.RESET} "


def format_timestamp() -> str:
    """Return a formatted current timestamp string."""
    return datetime.now().strftime("%H:%M:%S")
