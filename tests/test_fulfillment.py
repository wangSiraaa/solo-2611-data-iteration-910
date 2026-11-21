"""Whole-box delivery: warehouse acceptance, vouchers, reseal, pickup.

The auction still settles homogeneous units; these tests exercise the layer
that turns frozen per-buyer quantities into whole sealed boxes, and checks
the specific 3/2/1 boxes/rights scenario from the brief.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sealed_auction import (
    AuctionService,
    BoxWarehouse,
    ClearingService,
    Database,
    MarginLedger,
)
from sealed_auction.core import (
    ContentConflict,
    DomainError,
    IdempotencyConflict,
    InvalidState,
    InventoryMismatch,
    LimitExceeded,
    NotEligible,
)
from sealed_auction.reconcile import assert_reconciled, reconcile_fulfillment

ALL = ("jia", "yi", "bing")


class WHarness:
    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self._tmp.name) / "boxes.sqlite3")
        self.margin = MarginLedger(self.db)
        self.auctions = AuctionService(self.db, self.margin)
        self.clearing = ClearingService(self.db, self.margin, self.auctions)
        self.boxes = BoxWarehouse(self.db, self.auctions)
        self._n = 0

    def req(self, p="r"):
        self._n += 1
        return f"{p}-{self._n:05d}"

    def auction_3_2_1(self, auction_id="auc", units=6):
        """Three buyers win exactly 3, 2, 1 units at unit prices 500/400/300."""
        self.auctions.create_auction(self.req("new"), auction_id, units)
        for buyer, qty, price in (("jia", 3, 500), ("yi", 2, 400), ("bing", 1, 300)):
            self.margin.deposit(self.req("dep"), buyer, 10_000)
            hold = self.margin.freeze(
                self.req("frz"), buyer, auction_id, qty * price
            )
            self.auctions.place_bid(
                self.req("bid"), f"bid-{auction_id}-{buyer}", auction_id, buyer,
                qty, price, hold["hold_id"],
            )
        frozen = self.auctions.freeze_auction(self.req("close"), auction_id)
        self.clearing.settle(self.req("settle"), auction_id)
        return frozen

    def batch(self, batch_id="B1"):
        return self.boxes.open_batch(self.req("open"), batch_id)

    def receive(self, batch, seal, qty, cond="good", store=True):
        return self.boxes.receive_box(
            self.req("rcv"), batch, seal, qty, cond, store=store
        )

    def terms(self, buyer, conds=("good",), deadline="2099-01-01"):
        return self.boxes.set_buyer_terms(
            self.req("terms"), buyer, list(conds), deadline
        )

    def accept(self, batch, buyer, box_id):
        return self.boxes.accept_box(self.req("acc"), batch, buyer, box_id)

    def close(self):
        self._tmp.cleanup()


class Base(unittest.TestCase):
    def setUp(self):
        self.h = WHarness()

    def tearDown(self):
        self.h.close()

    def assertAllReconciled(self):
        assert_reconciled(self.h.db)
        report = reconcile_fulfillment(self.h.db)
        self.assertTrue(report["ok"], report["violations"])


class IntakeTest(Base):
    def test_same_seal_same_content_counts_once(self):
        h = self.h
        h.batch("B")
        a = h.receive("B", "SEAL-1", 3)
        b = h.receive("B", "SEAL-1", 3)
        self.assertFalse(a["duplicate"])
        self.assertTrue(b["duplicate"])
        self.assertEqual(a["box_id"], b["box_id"])
        with h.db.read() as conn:
            n = conn.execute(
                "SELECT received_boxes FROM intake_batches WHERE batch_id='B'"
            ).fetchone()["received_boxes"]
        self.assertEqual(n, 1)
        self.assertAllReconciled()

    def test_same_seal_different_content_requires_correction(self):
        h = self.h
        h.batch("B")
        h.receive("B", "SEAL-1", 3, "good")
        with self.assertRaises(ContentConflict):
            h.receive("B", "SEAL-1", 2, "good")
        with self.assertRaises(ContentConflict):
            h.receive("B", "SEAL-1", 3, "damp")
        # Explicit correction is allowed while the box is untouched.
        fixed = h.boxes.restate_box(h.req("rst"), "SEAL-1", 2, "damp")
        self.assertEqual(fixed["qty"], 2)
        self.assertEqual(fixed["condition"], "damp")
        # Now the corrected content is the one that matches.
        again = h.receive("B", "SEAL-1", 2, "damp")
        self.assertTrue(again["duplicate"])
        self.assertAllReconciled()

    def test_box_limits_and_positive_piece_counts(self):
        h = self.h
        h.boxes.open_batch(h.req("o"), "B", max_boxes=2)
        with self.assertRaises(DomainError):
            h.receive("B", "ZERO", 0)
        h.receive("B", "S1", 1)
        h.receive("B", "S2", 1)
        with self.assertRaises(LimitExceeded):
            h.receive("B", "S3", 1)
        with self.assertRaises(LimitExceeded):
            h.boxes.open_batch(h.req("o"), "B9", max_boxes=25)


class PlanTest(Base):
    def _seed(self):
        h = self.h
        h.auction_3_2_1()
        h.batch("B1")
        h.receive("B1", "S3", 3)
        h.receive("B1", "S2", 2)
        h.receive("B1", "S1", 1)
        h.terms("jia")
        h.terms("yi")
        h.terms("bing")
        # The 3-piece box is acceptable only to yi; others accept the rest.
        h.accept("B1", "yi", "box-S3")
        for buyer in ALL:
            h.accept("B1", buyer, "box-S2")
            h.accept("B1", buyer, "box-S1")

    def test_three_piece_box_cannot_be_split_for_yi(self):
        h = self.h
        self._seed()
        # yi's right is 2, so the 3-piece box is not a valid option for yi,
        # and jia (who needs 3) does not accept it.  It cannot be cut up.
        plan = h.boxes.plan("B1", "auc")
        self.assertEqual(plan["fulfilled_units"], 3)
        self.assertEqual(plan["fulfilled_buyers"], 2)
        self.assertEqual(set(plan["served_buyers"]), {"yi", "bing"})
        self.assertEqual([b["qty"] for b in plan["assignments"]["yi"]], [2])
        self.assertEqual([b["qty"] for b in plan["assignments"]["bing"]], [1])
        self.assertEqual(plan["assignments"]["bing"][0]["seal_id"], "S1")

    def test_maximises_pieces_then_buyers(self):
        h = self.h
        self._seed()
        # Issuing to jia alone (2+1) fulfils only one buyer; the objective
        # prefers 3 pieces across 2 buyers.
        joint = h.boxes.issue_vouchers(h.req("iv"), "B1", "auc")
        self.assertEqual(joint["fulfilled_units"], 3)
        self.assertEqual(joint["fulfilled_buyers"], 2)
        served = {v["buyer"]: [b["seal_id"] for b in v["boxes"]]
                  for v in joint["vouchers"]}
        self.assertEqual(served, {"yi": ["S2"], "bing": ["S1"]})
        self.assertEqual(joint["unserved_buyers"], ["jia"])
        self.assertAllReconciled()

    def test_individual_issuance_refuses_half_voucher(self):
        h = self.h
        self._seed()
        # jia's full right (3) is met by 2+1, so a single voucher is fine.
        jia = h.boxes.issue_voucher_for_buyer(h.req("iv"), "B1", "auc", "jia")
        self.assertEqual(jia["vouchers"][0]["units"], 3)
        self.assertEqual(
            sorted(b["seal_id"] for b in jia["vouchers"][0]["boxes"]), ["S1", "S2"]
        )
        # yi now cannot get a full 2 (S3=3 unusable, S1/S2 taken): refused,
        # never a partial 1-unit voucher.
        with self.assertRaises(NotEligible):
            h.boxes.issue_voucher_for_buyer(h.req("iv"), "B1", "auc", "yi")
        self.assertAllReconciled()

    def test_no_terms_or_deadline_passed_blocks_voucher(self):
        h = self.h
        h.auction_3_2_1()
        h.batch("B1")
        h.receive("B1", "S3", 3)
        h.receive("B1", "S2", 2)
        h.receive("B1", "S1", 1)
        # bing never set terms; yi's deadline has passed; jia accepts no box.
        h.terms("yi", deadline="2000-01-01")
        h.terms("jia")
        plan = h.boxes.plan("B1", "auc")
        reasons = {r["buyer"]: r["reason"] for r in plan["rights"]}
        self.assertEqual(reasons["bing"], "no_terms")
        self.assertEqual(reasons["yi"], "deadline_passed")
        self.assertEqual(reasons["jia"], "no_box_combination")
        self.assertEqual(plan["served_buyers"], [])
        with self.assertRaises(NotEligible):
            h.boxes.issue_voucher_for_buyer(h.req("iv"), "B1", "auc", "yi")
        self.assertAllReconciled()


class ResealTest(Base):
    def _seed_with_seal(self):
        h = self.h
        h.auction_3_2_1()
        h.batch("B1")
        h.receive("B1", "S3", 3)
        h.receive("B1", "S2", 2)
        h.receive("B1", "S1", 1)
        for buyer in ALL:
            h.terms(buyer)
        h.accept("B1", "yi", "box-S3")
        for buyer in ALL:
            h.accept("B1", buyer, "box-S2")
            h.accept("B1", buyer, "box-S1")

    def test_reseal_three_into_two_and_one_lets_all_collect(self):
        h = self.h
        self._seed_with_seal()
        before = h.boxes.plan("B1", "auc")
        self.assertEqual(before["fulfilled_units"], 3)

        # Warehouse receives the two halves, then officially reseals S3.
        h.receive("B1", "S3a", 1)
        h.receive("B1", "S3b", 2)
        reseal = h.boxes.reseal(
            h.req("reseal"), "B1", "box-S3",
            [{"seal_id": "S3a", "qty": 1}, {"seal_id": "S3b", "qty": 2}],
            authorized_by=["jia"],
        )
        self.assertEqual(reseal["total_qty"], 3)
        self.assertEqual(
            sorted(b["seal_id"] for b in reseal["new_boxes"]), ["S3a", "S3b"]
        )

        plan = h.boxes.plan("B1", "auc")
        self.assertEqual(plan["fulfilled_units"], 6)
        self.assertEqual(plan["fulfilled_buyers"], 3)
        # jia authorised the reseal, so it now accepts both halves.
        self.assertIn("box-S3a", {b["box_id"] for b in plan["assignments"]["jia"]})
        self.assertIn("box-S3b", {b["box_id"] for b in plan["assignments"]["jia"]})

        issued = h.boxes.issue_vouchers(h.req("iv"), "B1", "auc")
        self.assertEqual(issued["fulfilled_units"], 6)
        for v in issued["vouchers"]:
            h.boxes.pickup(h.req("pk"), v["voucher_id"], "Window-A")
        # Every buyer collected their full awarded quantity.
        for buyer, expected in (("jia", 3), ("yi", 2), ("bing", 1)):
            right = h.boxes.delivery_right("auc", buyer)
            self.assertEqual(right["picked_units"], expected)
            self.assertEqual(right["remaining_units"], 0)
        self.assertAllReconciled()

    def test_reseal_requires_conservation_and_two_received_boxes(self):
        h = self.h
        self._seed_with_seal()
        h.receive("B1", "S3a", 1)
        h.receive("B1", "S3b", 2)
        with self.assertRaises(DomainError):
            h.boxes.reseal(
                h.req("rs"), "B1", "box-S3",
                [{"seal_id": "S3a", "qty": 1}, {"seal_id": "S3b", "qty": 3}],
                ["jia"],
            )
        with self.assertRaises(DomainError):
            h.boxes.reseal(
                h.req("rs"), "B1", "box-S3",
                [{"seal_id": "S3a", "qty": 1}], ["jia"],
            )
        with self.assertRaises(DomainError):
            # No authorising buyer: the reseal does not change anyone's rights.
            h.boxes.reseal(
                h.req("rs"), "B1", "box-S3",
                [{"seal_id": "S3a", "qty": 1}, {"seal_id": "S3b", "qty": 2}], []
            )
        self.assertAllReconciled()

    def test_reseal_after_pickup_is_forbidden(self):
        h = self.h
        self._seed_with_seal()
        h.receive("B1", "S3a", 1)
        h.receive("B1", "S3b", 2)
        h.boxes.reseal(
            h.req("rs"), "B1", "box-S3",
            [{"seal_id": "S3a", "qty": 1}, {"seal_id": "S3b", "qty": 2}], ["jia"]
        )
        issued = h.boxes.issue_vouchers(h.req("iv"), "B1", "auc")
        # Hand out the voucher that carries the resealed halves.
        jia_v = next(v for v in issued["vouchers"] if v["buyer"] == "jia")
        h.boxes.pickup(h.req("pk"), jia_v["voucher_id"], "W")
        with self.assertRaises(InvalidState):
            h.boxes.reseal(
                h.req("rs"), "B1", "box-S3b",
                [{"seal_id": "S1", "qty": 1}, {"seal_id": "S2", "qty": 1}], ["yi"]
            )
        self.assertAllReconciled()


class PickupTest(Base):
    def _single_buyer(self, units=2):
        h = self.h
        h.auctions.create_auction(h.req("new"), "A", units)
        h.margin.deposit(h.req("dep"), "bob", 10_000)
        hold = h.margin.freeze(h.req("frz"), "bob", "A", units * 100)
        h.auctions.place_bid(h.req("bid"), "b1", "A", "bob", units, 100, hold["hold_id"])
        h.auctions.freeze_auction(h.req("close"), "A")
        h.clearing.settle(h.req("settle"), "A")
        h.batch("B")
        h.receive("B", f"P{units}", units)
        h.terms("bob")
        h.accept("B", "bob", f"box-P{units}")

    def test_pickup_receipt_is_idempotent_and_never_doubled(self):
        h = self.h
        self._single_buyer(2)
        v = h.boxes.issue_vouchers(h.req("iv"), "B", "A")["vouchers"][0]["voucher_id"]
        rid = h.req("pk")
        first = h.boxes.pickup(rid, v, "W1")
        replay = h.boxes.pickup(rid, v, "W1")
        self.assertEqual(first["boxes"], replay["boxes"])
        # A genuinely new request cannot redeem the same voucher again.
        with self.assertRaises(InvalidState):
            h.boxes.pickup(h.req("pk"), v, "W1")
        with self.assertRaises(IdempotencyConflict):
            h.boxes.pickup(rid, v, "W2")
        # The original receipt is recoverable, not replayed as new delivery.
        recovered = h.boxes.get_pickup(rid)
        self.assertEqual(recovered["voucher_id"], v)
        self.assertEqual([b["status"] for b in recovered["boxes"]], ["PICKED"])
        self.assertEqual(h.boxes.delivery_right("A", "bob")["remaining_units"], 0)
        self.assertAllReconciled()

    def test_another_window_withdrawal_makes_confirmation_respect_stock(self):
        h = self.h
        self._single_buyer(2)
        v = h.boxes.issue_vouchers(h.req("iv"), "B", "A")["vouchers"][0]["voucher_id"]
        # The physical box leaves unexpectedly at another window.
        h.boxes.withdraw_at_window(h.req("wd"), "box-P2", "OtherWindow")
        with self.assertRaises(InventoryMismatch):
            h.boxes.pickup(h.req("pk"), v, "W1")
        snapped = h.boxes.inspect_voucher(v)
        self.assertEqual(snapped["status"], "ISSUED")
        self.assertEqual(len(snapped["broken"]), 1)
        with self.assertRaises(InventoryMismatch):
            h.boxes.withdraw_at_window(h.req("wd"), "box-P2", "ThirdWindow")
        self.assertAllReconciled()

    def test_damaged_box_cannot_be_handed_out(self):
        h = self.h
        self._single_buyer(2)
        v = h.boxes.issue_vouchers(h.req("iv"), "B", "A")["vouchers"][0]["voucher_id"]
        h.boxes.damage_box(h.req("dmg"), "box-P2", "crushed")
        with self.assertRaises(InventoryMismatch):
            h.boxes.pickup(h.req("pk"), v, "W1")
        # Void the stale voucher; the right is still owed but unfulfillable
        # until fresh stock arrives.
        h.boxes.void_voucher(h.req("void"), v, "box damaged")
        self.assertEqual(h.boxes.delivery_right("A", "bob")["remaining_units"], 2)
        self.assertAllReconciled()

    def test_void_returns_boxes_and_cannot_void_picked(self):
        h = self.h
        self._single_buyer(2)
        v = h.boxes.issue_vouchers(h.req("iv"), "B", "A")["vouchers"][0]["voucher_id"]
        h.boxes.void_voucher(h.req("void"), v)
        # Box is usable again by a fresh voucher.
        v2 = h.boxes.issue_vouchers(h.req("iv"), "B", "A")["vouchers"][0]["voucher_id"]
        h.boxes.pickup(h.req("pk"), v2, "W")
        with self.assertRaises(InvalidState):
            h.boxes.void_voucher(h.req("void"), v2)
        self.assertAllReconciled()


class MoneyIsolationTest(Base):
    def test_uncollected_buyer_does_not_rewrite_clearing_price(self):
        h = self.h
        frozen = h.auction_3_2_1()
        # Uniform clearing price is the lowest allocated bid price (300).
        self.assertEqual(frozen["clearing_price_cents"], 300)
        h.batch("B1")
        h.receive("B1", "X3", 3)  # only jia can use it, jia needs 3
        for buyer in ALL:
            h.terms(buyer)
        h.accept("B1", "jia", "box-X3")
        plan = h.boxes.plan("B1", "auc")
        # Only jia is served; yi and bing are owed but cannot collect.
        self.assertEqual(plan["served_buyers"], ["jia"])
        # Charges remain the true old settlement at 300/unit, untouched.
        for buyer, qty in (("jia", 3), ("yi", 2), ("bing", 1)):
            self.assertEqual(h.margin.balance(buyer)["charged_cents"], qty * 300)
        # Re-reading the auction still returns the same frozen price.
        reopened = h.auctions.read_result("auc")
        self.assertEqual(reopened["clearing_price_cents"], 300)
        self.assertAllReconciled()


class ManyBoxesTest(Base):
    def test_many_small_boxes_exactly_fill_one_voucher(self):
        h = self.h
        h.auctions.create_auction(h.req("new"), "A", 6)
        h.margin.deposit(h.req("dep"), "bob", 10_000)
        hold = h.margin.freeze(h.req("frz"), "bob", "A", 600)
        h.auctions.place_bid(h.req("bid"), "b1", "A", "bob", 6, 100, hold["hold_id"])
        h.auctions.freeze_auction(h.req("close"), "A")
        h.clearing.settle(h.req("settle"), "A")
        h.batch("B")
        # Eight boxes where the only whole-box combination summing to 6 is
        # 1+1+2+2; the larger boxes are all too big to help.
        for seal, qty in (("a", 1), ("b", 1), ("c", 2), ("d", 2),
                          ("e", 7), ("f", 7), ("g", 8), ("h", 8)):
            h.receive("B", seal, qty)
        h.terms("bob")
        for seal in "abcdefgh":
            h.accept("B", "bob", f"box-{seal}")
        out = h.boxes.issue_vouchers(h.req("iv"), "B", "A")
        self.assertEqual(out["fulfilled_units"], 6)
        v = out["vouchers"][0]
        self.assertEqual(sum(b["qty"] for b in v["boxes"]), 6)
        self.assertEqual(sorted(b["qty"] for b in v["boxes"]), [1, 1, 2, 2])
        h.boxes.pickup(h.req("pk"), v["voucher_id"], "W")
        self.assertAllReconciled()

    def test_one_condition_only_one_box_matches(self):
        h = self.h
        h.auctions.create_auction(h.req("new"), "A", 2)
        h.margin.deposit(h.req("dep"), "bob", 10_000)
        hold = h.margin.freeze(h.req("frz"), "bob", "A", 200)
        h.auctions.place_bid(h.req("bid"), "b1", "A", "bob", 2, 100, hold["hold_id"])
        h.auctions.freeze_auction(h.req("close"), "A")
        h.clearing.settle(h.req("settle"), "A")
        h.batch("B")
        h.receive("B", "warm", 2, "ambient")
        h.receive("B", "cold", 2, "chilled")
        # bob only licenses chilled custody.
        h.terms("bob", conds=("chilled",))
        h.accept("B", "bob", "box-warm")
        h.accept("B", "bob", "box-cold")
        out = h.boxes.issue_vouchers(h.req("iv"), "B", "A")
        self.assertEqual(
            [b["seal_id"] for b in out["vouchers"][0]["boxes"]], ["cold"]
        )
        self.assertAllReconciled()


class IssuanceIdempotencyTest(Base):
    def _seed(self):
        h = self.h
        h.auction_3_2_1()
        h.batch("B1")
        h.receive("B1", "S3", 3)
        h.receive("B1", "S2", 2)
        h.receive("B1", "S1", 1)
        for buyer in ALL:
            h.terms(buyer)
        h.accept("B1", "yi", "box-S3")
        for buyer in ALL:
            h.accept("B1", buyer, "box-S2")
            h.accept("B1", buyer, "box-S1")

    def test_replaying_issue_does_not_duplicate_vouchers(self):
        h = self.h
        self._seed()
        rid = h.req("issue")
        first = h.boxes.issue_vouchers(rid, "B1", "auc")
        again = h.boxes.issue_vouchers(rid, "B1", "auc")
        self.assertEqual(first, again)
        with h.db.read() as conn:
            n = conn.execute(
                "SELECT COUNT(*) n FROM vouchers WHERE auction_id='auc'"
            ).fetchone()["n"]
        self.assertEqual(n, len(first["vouchers"]))
        # A second, distinct request would try to re-use boxes already held
        # by live vouchers, so it should serve nobody new.
        second = h.boxes.issue_vouchers(h.req("issue"), "B1", "auc")
        self.assertEqual(second["served_buyers"], [])
        self.assertAllReconciled()

    def test_same_box_never_issued_to_two_buyers(self):
        h = self.h
        self._seed()
        out = h.boxes.issue_vouchers(h.req("issue"), "B1", "auc")
        with h.db.read() as conn:
            rows = conn.execute(
                "SELECT box_id, COUNT(*) n FROM voucher_boxes GROUP BY box_id"
            ).fetchall()
        self.assertTrue(all(r["n"] == 1 for r in rows))
        all_boxes = [b["box_id"] for v in out["vouchers"] for b in v["boxes"]]
        self.assertEqual(len(all_boxes), len(set(all_boxes)))
        self.assertAllReconciled()


class VoucherCapTest(Base):
    def test_at_most_ten_vouchers_per_batch(self):
        h = self.h
        buyers = [f"b{i}" for i in range(5)]
        h.batch("B")
        # 24 one-piece boxes; each buyer wins 1 unit in each of 3 auctions.
        for i in range(24):
            h.receive("B", f"s{i}", 1)
        for buyer in buyers:
            h.terms(buyer)
            for i in range(24):
                h.accept("B", buyer, f"box-s{i}")
        voucher_count = 0
        for auction_no, auction_buyers in enumerate(
            (buyers, buyers, ["b0"])
        ):
            aid = f"A{auction_no}"
            h.auctions.create_auction(h.req("new"), aid, len(auction_buyers))
            for buyer in auction_buyers:
                h.margin.deposit(h.req("dep"), buyer, 1_000)
                hold = h.margin.freeze(h.req("frz"), buyer, aid, 100)
                h.auctions.place_bid(
                    h.req("bid"), f"bid-{aid}-{buyer}", aid, buyer, 1, 100,
                    hold["hold_id"],
                )
            h.auctions.freeze_auction(h.req("close"), aid)
            h.clearing.settle(h.req("settle"), aid)
            if auction_no < 2:
                out = h.boxes.issue_vouchers(h.req("iv"), "B", aid)
                voucher_count += len(out["vouchers"])
                self.assertEqual(len(out["vouchers"]), len(auction_buyers))
            else:
                # The 11th voucher for this batch is refused.
                with self.assertRaises(LimitExceeded):
                    h.boxes.issue_vouchers(h.req("iv"), "B", aid)
        self.assertEqual(voucher_count, 10)
        self.assertAllReconciled()


if __name__ == "__main__":
    unittest.main(verbosity=2)
