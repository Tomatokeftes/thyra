# msiconvert/tools/check_ontology.py
"""Command-line tool to check ontology terms in imzML files."""

import argparse
import json
import logging
from pathlib import Path

from ..metadata.ontology.cache import ONTOLOGY


def main():
    parser = argparse.ArgumentParser(description="Check ontology terms in imzML files")
    parser.add_argument("input", help="imzML file or directory to check")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    from ..metadata.validator import ImzMLOntologyValidator

    validator = ImzMLOntologyValidator()
    input_path = Path(args.input)

    if input_path.is_file():
        results = validator.validate_file(input_path)

        # Only print summary if not outputting to JSON file
        if not args.output:
            if args.verbose and "summary" in results:
                # In verbose mode, print the summary from results
                print(results["summary"])
            else:
                # In normal mode, generate formatted summary
                print("Ontology Validation Summary")
                print("===========================")
                print("Test summary for file")
                print()

            unknown_terms = results.get("unknown_terms", [])
            if unknown_terms:
                print(f"Found {len(unknown_terms)} unknown terms:")
                for term in unknown_terms[:20]:  # Show first 20 terms
                    print(f"  - {term}")
            else:
                print("No unknown terms encountered.")
            print()
    else:
        results = validator.validate_directory(input_path)
        print(f"Checked {results['files_checked']} files")
        print(f"Found {len(results['all_unknown_terms'])} unique unknown terms")

        if results["all_unknown_terms"]:
            print("\nMost common unknown terms:")
            for term in list(results["all_unknown_terms"])[:20]:
                print(f"  - {term}")
            print()  # Add blank line after terms list

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output}")
        print(ONTOLOGY.report_unknown_terms())
    else:
        # Show global unknown terms (with newline if we already printed summary)
        print(ONTOLOGY.report_unknown_terms())


if __name__ == "__main__":
    main()
