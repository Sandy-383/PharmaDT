"""Command-line verification of the provenance chain.

Usage::

    make verify-chain
    python -m pharmadt.ledger.verify --batch BATCH-0001
    python -m pharmadt.ledger.verify --range 1 500
"""

from __future__ import annotations

import argparse
import time

from pharmadt.ledger.chain import HashChainLedger


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the provenance ledger.")
    parser.add_argument("--batch", help="also print this batch's custody trace")
    parser.add_argument(
        "--range", nargs=2, type=int, metavar=("START", "END"), help="verify a seq range only"
    )
    args = parser.parse_args()

    ledger = HashChainLedger()
    height = ledger.height()
    if height == 0:
        raise SystemExit("The ledger is empty. Run `make sim-anchor` first.")

    start, end = (args.range if args.range else (None, None))

    started = time.perf_counter()
    result = ledger.verify_chain_detailed(start, end)
    elapsed = time.perf_counter() - started

    print(f"Chain height: {height:,} records")
    print(f"Tip:          {ledger.tip()}")
    print(f"Checked:      {result.records_checked:,} records in {elapsed:.2f}s")

    if result.valid:
        print("Result:       VALID -- every hash links and every signature verifies")
    else:
        print(f"Result:       BROKEN at seq {result.broken_at_seq}")
        print(f"Reason:       {result.reason}")

    if args.batch:
        trace = ledger.get_provenance(args.batch)
        print(f"\n{args.batch}: {len(trace)} custody events")
        for record in trace:
            print(
                f"  seq {record['seq']:>6}  day {record['sim_day']:>4}  "
                f"{record['event_type']:<22} "
                f"{(record['from_node'] or '-'):<12} -> {(record['to_node'] or '-'):<12} "
                f"signed by {record['signer_node']}"
            )

    raise SystemExit(0 if result.valid else 1)


if __name__ == "__main__":
    main()
