"""Whole-box fulfilment demo for the 3 / 2 / 1 scenario.

Auction: jia wins 3 units, yi 2, bing 1 (uniform clearing price 300 cents).
Warehouse stock: one 3-piece, one 2-piece and one 1-piece sealed box.

The 3-piece box is acceptable only to yi, but yi is owed only 2 and a box is
never split, so the best joint issue is 3 pieces across yi (2) and bing (1);
jia (3) is left owed.  Individual (non-joint) issuance would let jia take
2+1 for one buyer and lock the others out.  The warehouse then officially
reseals the 3-piece box into 1 + 2 pieces, jia authorises it, and all three
collect their full 6 pieces.

Run from the repository root:

    python3 -m examples.run_delivery
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sealed_auction import (
    AuctionService,
    BoxWarehouse,
    ClearingService,
    Database,
    MarginLedger,
)
from sealed_auction.core import DomainError
from sealed_auction.reconcile import assert_reconciled, reconcile_fulfillment


def show(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def dump(label: str, obj) -> None:
    print(f"\n[{label}]")
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def main() -> int:
    tmp = tempfile.TemporaryDirectory()
    db = Database(Path(tmp.name) / "delivery.sqlite3")
    margin = MarginLedger(db)
    auctions = AuctionService(db, margin)
    clearing = ClearingService(db, margin, auctions)
    boxes = BoxWarehouse(db, auctions)
    counter = {"n": 0}

    def req(tag: str) -> str:
        counter["n"] += 1
        return f"demo-{tag}-{counter['n']:03d}"

    try:
        show("1. old auction settles homogeneous units: jia 3, yi 2, bing 1")
        auctions.create_auction(req("new"), "A", 6)
        for buyer, qty, price in (("jia", 3, 500), ("yi", 2, 400), ("bing", 1, 300)):
            margin.deposit(req("dep"), buyer, 10_000)
            hold = margin.freeze(req("frz"), buyer, "A", qty * price)
            auctions.place_bid(req("bid"), f"bid-{buyer}", "A", buyer, qty, price,
                               hold["hold_id"])
        frozen = auctions.freeze_auction(req("close"), "A")
        settled = clearing.settle(req("settle"), "A")
        for a in frozen["allocations"]:
            print(f"    {a['buyer']:>4}: bid {a['qty']} @ {a['unit_price_cents']} "
                  f"-> allocated {a['allocated_qty']} order {a['order_cents']}")
        print(f"    uniform clearing price = {frozen['clearing_price_cents']} cents")
        print("    charges are final and never re-priced by fulfilment:")
        for buyer in ("jia", "yi", "bing"):
            print(f"      {buyer:>4} charged {margin.balance(buyer)['charged_cents']}")

        show("2. warehouse acceptance: seals, pieces, batch, custody condition")
        boxes.open_batch(req("open"), "B1")
        for seal, qty in (("S3", 3), ("S2", 2), ("S1", 1)):
            r = boxes.receive_box(req("rcv"), "B1", seal, qty, "good", store=True)
            print(f"    received seal {seal} = {qty} pcs -> {r['box_id']} "
                  f"status {r['status']}")
        # Same seal, same content re-arrives: counted once.
        dup = boxes.receive_box(req("rcv"), "B1", "S3", 3, "good", store=True)
        print(f"    S3 re-delivered: duplicate={dup['duplicate']} (counted once)")
        # Same seal, different content: must be corrected, not merged.
        try:
            boxes.receive_box(req("rcv"), "B1", "S2", 5, "good")
        except DomainError as exc:
            print(f"    S2 conflicting re-declaration rejected: {exc}")

        for buyer in ("jia", "yi", "bing"):
            boxes.set_buyer_terms(req("terms"), buyer, ["good"], "2099-01-01")
        # Only yi accepts the 3-piece box; all accept the other two.
        boxes.accept_box(req("acc"), "B1", "yi", "box-S3")
        for buyer in ("jia", "yi", "bing"):
            boxes.accept_box(req("acc"), "B1", buyer, "box-S2")
            boxes.accept_box(req("acc"), "B1", buyer, "box-S1")

        show("3. joint issue: maximise fulfilled pieces, then buyers")
        plan = boxes.plan("B1", "A")
        for r in plan["rights"]:
            print(f"    {r['buyer']:>4}: owed {r['remaining_units']} "
                  f"served={r['served']}")
        print(f"    -> {plan['fulfilled_units']} pieces, "
              f"{plan['fulfilled_buyers']} buyers: {plan['served_buyers']}")
        print("    (yi cannot use the 3-piece box: owed 2 and boxes are whole;")
        print("     jia takes 2+1 would fulfil only one buyer, so it loses)")

        # Contrast: issue to jia first (2+1) — a valid full voucher for jia,
        # but then yi and bing are both starved, i.e. only one buyer served.
        jia_alone = boxes.issue_voucher_for_buyer(req("one"), "B1", "A", "jia")
        print(f"    greedy alternative: jia alone gets {jia_alone['vouchers'][0]['units']} "
              f"units seals={[b['seal_id'] for b in jia_alone['vouchers'][0]['boxes']]}")
        print("    -> only 3 pieces / 1 buyer, so the joint plan (3 pieces /")
        print("       2 buyers) wins; the greedy voucher is withdrawn.")

        show("4. individual issue refuses to hand out a half voucher")
        # With S1+S2 held by jia, yi's full right (2) can only be met by the
        # 3-piece box — and a box is never cut to 2.  The single-buyer path
        # refuses rather than issuing a partial voucher.
        try:
            boxes.issue_voucher_for_buyer(req("one"), "B1", "A", "yi")
        except DomainError as exc:
            print(f"    yi single-voucher refused: {exc}")

        boxes.void_voucher(req("void"), jia_alone["vouchers"][0]["voucher_id"],
                           "prefer joint plan")
        first = boxes.issue_vouchers(req("issue"), "B1", "A")
        for v in first["vouchers"]:
            print(f"    voucher {v['buyer']:>4} units={v['units']} "
                  f"seals={[b['seal_id'] for b in v['boxes']]}")

        show("5. withdraw the partial round; warehouse reseals 3 -> 1 + 2")
        for v in first["vouchers"]:
            boxes.void_voucher(req("void"), v["voucher_id"], "re-plan after reseal")
            print(f"    voided voucher for {v['buyer']}: boxes back in usable stock")
        boxes.receive_box(req("rcv"), "B1", "S3a", 1, "good", store=True)
        boxes.receive_box(req("rcv"), "B1", "S3b", 2, "good", store=True)
        reseal = boxes.reseal(
            req("reseal"), "B1", "box-S3",
            [{"seal_id": "S3a", "qty": 1}, {"seal_id": "S3b", "qty": 2}],
            authorized_by=["jia"],
        )
        print(f"    seal S3 {reseal['old_qty']} -> "
              f"{[(b['seal_id'], b['qty']) for b in reseal['new_boxes']]} "
              f"(pieces conserved: {reseal['total_qty']})")

        show("6. re-open issuance: all three now collect their full quantity")
        second = boxes.issue_vouchers(req("issue"), "B1", "A")
        print(f"    fulfilled {second['fulfilled_units']} pieces, "
              f"{second['fulfilled_buyers']} buyers")
        pick_reqs = []
        for v in second["vouchers"]:
            print(f"    voucher {v['buyer']:>4} units={v['units']} "
                  f"seals={[b['seal_id'] for b in v['boxes']]}")
            rid = req("pick")
            pick_reqs.append((v["buyer"], rid))
            receipt = boxes.pickup(rid, v["voucher_id"], "Window-A")
            print(f"      handed over at {receipt['window_id']}: "
                  f"{[b['seal_id'] for b in receipt['boxes']]}")

        show("7. re-open pickup receipts: recover, don't deliver twice")
        for buyer, rid in pick_reqs:
            rec = boxes.get_pickup(rid)
            print(f"    {buyer:>4} receipt {rec['request_id']} -> "
                  f"{[b['seal_id'] for b in rec['boxes']]}")
        for v in second["vouchers"]:
            snap = boxes.inspect_voucher(v["voucher_id"])
            print(f"    {v['buyer']:>4} voucher {snap['status']} "
                  f"receipt={snap['pickup_receipt'] is not None} "
                  f"broken={len(snap['broken'])}")
        try:
            boxes.pickup(req("pick"), second["vouchers"][0]["voucher_id"], "Window-B")
        except DomainError as exc:
            print(f"    second handover refused: {exc}")

        show("8. who took what, what was owed, and what remains")
        stmt = boxes.delivery_statement("B1", "A")
        for right in stmt["plan"]["rights"]:
            print(f"    {right['buyer']:>4}: awarded {right['awarded_units']} "
                  f"picked {right['picked_units']} owed {right['remaining_units']}")
        print(f"    still owed: {stmt['still_owed']}")

        show("9. independent reconciliation (money and boxes)")
        assert_reconciled(db)
        report = reconcile_fulfillment(db)
        print(f"    fulfillment ok={report['ok']} violations={report['violations']}")
        print(f"    auction charged total={settled['orders']}")
        return 0 if report["ok"] else 1
    finally:
        tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
