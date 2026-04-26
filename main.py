"""CLI entry point for the Academic Graph-CoT Tutor."""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.agct_system import AGCTSystem


def print_header() -> None:
    print("\n" + "=" * 70)
    print("ACADEMIC GRAPH-CoT TUTOR (AGCT)")
    print("=" * 70)
    print("General study support with retrieval, Graph-CoT reasoning, and verification")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


def print_instructions() -> None:
    print("\nInstructions:")
    print("  - Ask a question")
    print("  - Type 'help' for commands")
    print("  - Type 'stats' for system information")
    print("  - Type 'exit' or 'quit' to leave")
    print("=" * 70 + "\n")


def handle_special_commands(command: str, system: AGCTSystem):
    command = command.strip().lower()

    if command == "help":
        print("\nAvailable Commands:")
        print("  help   - Show help")
        print("  stats  - Show system statistics")
        print("  clear  - Clear the screen")
        print("  exit   - Exit the program\n")
        return True

    if command == "stats":
        stats = system.retriever.get_index_stats()
        print("\nSystem Statistics:")
        print(f"  Total Documents: {stats['total_documents']}")
        print(f"  Index Type: {stats['index_type']}")
        print(f"  Index Size: {stats['index_size']}")
        print(f"  Embedding Dimension: {stats['embedding_dimension']}")
        print(f"  Graph Nodes: {system.graph_engine.graph.number_of_nodes()}")
        print(f"  Graph Edges: {system.graph_engine.graph.number_of_edges()}\n")
        return True

    if command == "clear":
        os.system("cls" if os.name == "nt" else "clear")
        print_header()
        return True

    if command in {"exit", "quit"}:
        return False

    return None


def display_flashcards(cards) -> None:
    print("\nFLASHCARDS\n")
    for index, card in enumerate(cards, start=1):
        print(f"{index}. Front: {card.get('front', 'No front')}")
        print(f"   Back: {card.get('back', 'No back')}")
        print(f"   Hint: {card.get('hint', 'No hint')}\n")


def run_quiz(quiz_items) -> None:
    print("\nQUIZ\n")
    option_labels = ["A", "B", "C", "D"]

    for index, item in enumerate(quiz_items, start=1):
        print(f"{index}. {item.get('question', 'No question')}")
        options = item.get("options", [])
        for option_index, option in enumerate(options[:4]):
            print(f"   {option_labels[option_index]}. {option}")

        choice = input("   Your choice (A/B/C/D, or skip): ").strip().upper()
        correct_index = int(item.get("correct_index", 0))
        correct_label = option_labels[correct_index] if 0 <= correct_index < 4 else "A"
        correct_text = options[correct_index] if 0 <= correct_index < len(options) else "Unknown"

        print(f"   Correct answer: {correct_label}. {correct_text}")
        print(f"   Explanation: {item.get('explanation', 'No explanation')}")
        if choice and choice != "SKIP":
            print(f"   You selected: {choice}")
        print()


def main() -> None:
    print_header()

    try:
        system = AGCTSystem()
    except Exception as exc:
        print(f"\nFailed to initialize system: {exc}")
        print("Check that the data directory exists, dependencies are installed, and Ollama is running.")
        sys.exit(1)

    print_instructions()
    query_count = 0

    while True:
        try:
            user_input = input("Your question: ").strip()
            if not user_input:
                print("Please enter a question or command.\n")
                continue

            special = handle_special_commands(user_input, system)
            if special is False:
                print(f"\nGoodbye. Answered {query_count} questions.")
                break
            if special is True:
                continue

            print("\nProcessing your question...")
            result = system.process_query(user_input)
            query_count += 1

            if result.get("status") != "success":
                print(f"\n{result.get('error', 'Unknown error')}\n")
                continue

            print("\n" + "=" * 70)
            print("ANSWER\n")
            print(result["explanation"])
            if result.get("mode") == "flashcards" and result.get("flashcards"):
                display_flashcards(result["flashcards"])
            if result.get("mode") == "quiz" and result.get("quiz"):
                run_quiz(result["quiz"])
            print("\n" + "-" * 70)
            print(f"Difficulty: {result['difficulty']}")
            print(f"Learning Path: {' -> '.join(result['graph_path'])}")
            print(f"Verified: {'Yes' if result['verified'] else 'No'}")
            print("=" * 70 + "\n")
        except KeyboardInterrupt:
            print(f"\n\nInterrupted. Answered {query_count} questions.")
            break
        except Exception as exc:
            print(f"\nUnexpected error: {exc}\n")


if __name__ == "__main__":
    main()
