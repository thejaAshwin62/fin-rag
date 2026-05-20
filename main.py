"""
============================================================
main.py — RAG Finance AI Chatbot Entry Point
============================================================
Provides a menu-driven interface to:
1. Ingest finance data into ChromaDB
2. Start the interactive chatbot
3. Run the full pipeline (ingest + chat)

Usage:
    python main.py
============================================================
"""

import sys

from config import validate_env
from utils import setup_logging, print_header, print_separator, Colors


def show_menu():
    """Display the main application menu."""
    print(f"  {Colors.WHITE}Select an option:{Colors.RESET}\n")
    print(f"  {Colors.CYAN}[1]{Colors.RESET} 📥  Ingest Data       — Generate dataset, embed, store in ChromaDB")
    print(f"  {Colors.CYAN}[2]{Colors.RESET} 💬  Start Chatbot     — Launch interactive finance Q&A")
    print(f"  {Colors.CYAN}[3]{Colors.RESET} 🚀  Full Pipeline     — Ingest data, then start chatbot")
    print(f"  {Colors.CYAN}[4]{Colors.RESET} 🚪  Exit")
    print()
    print_separator("─", 60, Colors.DIM)


def main():
    """Main entry point — validates environment and runs selected option."""

    # Initialize logging
    logger = setup_logging()
    logger.info("Application started")

    # Show application header
    print_header("RAG Finance AI Chatbot", "Gemini + ChromaDB | Semantic Search | Streaming")

    # Validate API keys before proceeding
    validate_env()

    while True:
        show_menu()

        try:
            choice = input(f"  {Colors.BOLD}Enter choice [1-4]:{Colors.RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  {Colors.YELLOW}👋 Goodbye!{Colors.RESET}\n")
            sys.exit(0)

        print()

        if choice == "1":
            # ── Ingest Data ───────────────────────────────
            logger.info("User selected: Ingest Data")
            from ingest import run_ingestion
            run_ingestion()

        elif choice == "2":
            # ── Start Chatbot ─────────────────────────────
            logger.info("User selected: Start Chatbot")
            from chatbot import run_chatbot
            run_chatbot()

        elif choice == "3":
            # ── Full Pipeline ─────────────────────────────
            logger.info("User selected: Full Pipeline")
            from ingest import run_ingestion
            from chatbot import run_chatbot

            run_ingestion()
            print(f"\n  {Colors.GREEN}Ingestion complete! Starting chatbot...{Colors.RESET}\n")
            run_chatbot()

        elif choice == "4":
            print(f"  {Colors.YELLOW}👋 Thanks for using Finance AI Chatbot! Goodbye!{Colors.RESET}\n")
            logger.info("Application exited by user")
            sys.exit(0)

        else:
            print(f"  {Colors.RED}Invalid choice. Please enter 1, 2, 3, or 4.{Colors.RESET}\n")


if __name__ == "__main__":
    main()
