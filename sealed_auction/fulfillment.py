"""Sealed whole-box warehouse, delivery rights and pickup vouchers.

The legacy auction settles homogeneous *units* and freezes each buyer's
awarded quantity at cutoff.  Physical fulfilment, however, is by whole
supplier-sealed sample boxes: a box is never split.  This module layers
warehouse acceptance and delivery documents on top of the untouched auction
result:

  * intake       - each box has a stable seal, a positive integer piece
                   count, an intake batch and a custody condition.  A box
                   re-arriving with the same seal and same content counts
                   once; the same seal with different content is a conflict
                   that must be corrected, never silently merged.
  * inspection   - a stored (verified) box may be accepted by a buyer.  Only
                   stored, not-yet-withdrawn boxes are usable.
  * reseal       - an accepted whole box may be officially replaced by two
                   received boxes whose piece counts are conserved.  The old
                   seal must still be physically present: anything already
                   picked up can never be resealed.
  * fulfilment   - a buyer is served only by an all-or-nothing voucher whose
                   whole-box sizes sum to exactly that buyer's *full* frozen
                   remaining quantity.  Half vouchers are never issued.
  * allocation   - vouchers for a batch are issued jointly.  The chosen plan
                   maximises fulfilled pieces first, then the number of
                   fulfilled buyers; ties are broken arbitrarily but
                   deterministically.
  * pickup       - a pickup receipt must respect live stock: a box reported
                   damaged or already withdrawn at another window cannot be
                   handed over again, and a repeated/unknown receipt is
                   verified against the original document, never replayed as
                   a fresh delivery.

Money is untouched here.  The frozen clearing price is read, never rewritten,
so one buyer failing to collect cannot silently re-price another buyer.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Iterable, Sequence

from .core import (
    ContentConflict,
    DataIntegrityError,
    DomainError,
    InvalidState,
    InventoryMismatch,
    LimitExceeded,
    NotEligible,
    NotFound,
    get_receipt,
    now_utc,
    put_receipt,
    require_int,
    require_nonempty,
)

MAX_BOXES_PER_BATCH = 24
MAX_VOUCHERS_PER_BATCH = 10
# Safety valve for pathological subset enumeration; the documented
# constraints (<=24 boxes, <=10 units of demand per buyer) stay well inside it.

USABLE_BOX_STATES = ("RECEIVED", "STORED")
VOUCHERABLE_BOX_STATES = ("STORED",)


def _today() -> str:
    return date.today().isoformat()


def _as_date(value: str) -> str:
    require_nonempty(value, "date")
    return value[:10]


class BoxWarehouse:
    def __init__(self, db, auctions):
        self.db = db
        self.auctions = auctions

    # ============================================================ intake

    def open_batch(
        self,
        request_id: str,
        batch_id: str,
        max_boxes: int = MAX_BOXES_PER_BATCH,
        received_at: str | None = None,
    ) -> dict[str, Any]:
        require_nonempty(request_id, "request_id")
        require_nonempty(batch_id, "batch_id")
        require_int(max_boxes, "max_boxes", minimum=1)
        if max_boxes > MAX_BOXES_PER_BATCH:
            raise LimitExceeded(f"at most {MAX_BOXES_PER_BATCH} boxes per batch")
        payload = {"batch_id": batch_id, "max_boxes": max_boxes}
        operation = "box.open_batch"
        with self.db.tx() as conn:
            prior = get_receipt(conn, request_id, operation, payload)
            if prior is not None:
                return prior
            if conn.execute(
                "SELECT 1 FROM intake_batches WHERE batch_id=?", (batch_id,)
            ).fetchone():
                raise InvalidState(f"batch {batch_id} already exists")
            at = received_at or now_utc()
            conn.execute(
                "INSERT INTO intake_batches(batch_id, max_boxes, received_at, "
                "received_boxes) VALUES (?, ?, ?, 0)",
                (batch_id, max_boxes, at),
            )
            response = {"request_id": request_id, "batch_id": batch_id,
                        "max_boxes": max_boxes, "received_boxes": 0,
                        "received_at": at}
            put_receipt(conn, request_id, operation, payload, response)
        return response

    def receive_box(
        self,
        request_id: str,
        batch_id: str,
        seal_id: str,
        qty: int,
        condition: str,
        store: bool = False,
    ) -> dict[str, Any]:
        """Accept one supplier box.  Duplicate seals are counted once.

        Same seal + same content is an idempotent duplicate.  Same seal with
        a different batch/quantity/condition is a content conflict; the
        declared receipt must be corrected explicitly (``restate_box``)
        rather than silently overwritten.
        """
        require_nonempty(request_id, "request_id")
        require_nonempty(batch_id, "batch_id")
        require_nonempty(seal_id, "seal_id")
        require_int(qty, "qty", minimum=1)
        require_nonempty(condition, "condition")
        payload = {"batch_id": batch_id, "seal_id": seal_id, "qty": qty,
                   "condition": condition, "store": store}
        operation = "box.receive"
        with self.db.tx() as conn:
            prior = get_receipt(conn, request_id, operation, payload)
            if prior is not None:
                return prior
            batch = conn.execute(
                "SELECT * FROM intake_batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if batch is None:
                raise NotFound(f"batch {batch_id!r} not found")
            existing = conn.execute(
                "SELECT * FROM boxes WHERE seal_id=?", (seal_id,)
            ).fetchone()
            if existing is not None:
                if (existing["batch_id"] != batch_id or existing["qty"] != qty
                        or existing["condition"] != condition):
                    raise ContentConflict(
                        f"seal {seal_id} already received as "
                        f"{existing['qty']} pcs/{existing['condition']} in batch "
                        f"{existing['batch_id']}; new declaration "
                        f"{qty} pcs/{condition} in batch {batch_id} differs - "
                        "correct the receipt explicitly"
                    )
                response = {
                    "request_id": request_id,
                    "box_id": existing["box_id"],
                    "seal_id": seal_id,
                    "batch_id": batch_id,
                    "qty": qty,
                    "condition": condition,
                    "status": existing["status"],
                    "duplicate": True,
                }
                put_receipt(conn, request_id, operation, payload, response)
                return response
            if batch["received_boxes"] >= batch["max_boxes"]:
                raise LimitExceeded(
                    f"batch {batch_id} already holds {batch['max_boxes']} boxes"
                )
            box_id = f"box-{seal_id}"
            at = now_utc()
            status = "STORED" if store else "RECEIVED"
            conn.execute(
                "INSERT INTO boxes(box_id, seal_id, qty, batch_id, condition, "
                "status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (box_id, seal_id, qty, batch_id, condition, status, at, at),
            )
            conn.execute(
                "UPDATE intake_batches SET received_boxes = received_boxes + 1 "
                "WHERE batch_id=?",
                (batch_id,),
            )
            conn.execute(
                "INSERT INTO box_events(box_id, seal_id, request_id, action, "
                "detail_json, at) VALUES (?, ?, ?, ?, ?, ?)",
                (box_id, seal_id, request_id, "RECEIVE",
                 json.dumps({"qty": qty, "condition": condition,
                             "batch_id": batch_id, "stored": store},
                            sort_keys=True), at),
            )
            response = {
                "request_id": request_id,
                "box_id": box_id,
                "seal_id": seal_id,
                "batch_id": batch_id,
                "qty": qty,
                "condition": condition,
                "status": status,
                "duplicate": False,
            }
            put_receipt(conn, request_id, operation, payload, response)
        return response

    def store_box(self, request_id: str, box_id: str) -> dict[str, Any]:
        """Warehouse acceptance: verify and shelve an inspected box."""
        return self._box_transition(request_id, "box.store", box_id, "STORE",
                                     {"RECEIVED"}, "STORED")

    def restate_box(
        self, request_id: str, seal_id: str, qty: int, condition: str
    ) -> dict[str, Any]:
        """Correct a mis-declared receipt (the conflict remedy).

        Allowed only while the box is physically untouched: not yet picked,
        damaged or replaced.
        """
        require_nonempty(request_id, "request_id")
        require_nonempty(seal_id, "seal_id")
        require_int(qty, "qty", minimum=1)
        require_nonempty(condition, "condition")
        payload = {"seal_id": seal_id, "qty": qty, "condition": condition}
        operation = "box.restate"
        with self.db.tx() as conn:
            prior = get_receipt(conn, request_id, operation, payload)
            if prior is not None:
                return prior
            box = conn.execute(
                "SELECT * FROM boxes WHERE seal_id=?", (seal_id,)
            ).fetchone()
            if box is None:
                raise NotFound(f"seal {seal_id!r} not received")
            if box["status"] in ("PICKED", "DAMAGED", "REPLACED"):
                raise InvalidState(
                    f"box {box['box_id']} is {box['status']}; content cannot be "
                    "restated after it left usable stock"
                )
            at = now_utc()
            conn.execute(
                "UPDATE boxes SET qty=?, condition=?, updated_at=? WHERE box_id=?",
                (qty, condition, at, box["box_id"]),
            )
            conn.execute(
                "INSERT INTO box_events(box_id, seal_id, request_id, action, "
                "detail_json, at) VALUES (?, ?, ?, 'RESTATE', ?, ?)",
                (box["box_id"], seal_id, request_id,
                 json.dumps({"old_qty": box["qty"], "old_condition": box["condition"],
                             "qty": qty, "condition": condition},
                            sort_keys=True), at),
            )
            response = {"request_id": request_id, "box_id": box["box_id"],
                        "seal_id": seal_id, "qty": qty, "condition": condition,
                        "status": box["status"]}
            put_receipt(conn, request_id, operation, payload, response)
        return response

    def _box_transition(self, request_id, operation, box_id, action,
                        allowed_from: Sequence[str], to_status: str) -> dict:
        require_nonempty(request_id, "request_id")
        require_nonempty(box_id, "box_id")
        payload = {"box_id": box_id}
        with self.db.tx() as conn:
            prior = get_receipt(conn, request_id, operation, payload)
            if prior is not None:
                return prior
            box = self._box(conn, box_id)
            if box["status"] not in allowed_from:
                raise InvalidState(
                    f"box {box_id} is {box['status']}; expected one of "
                    f"{list(allowed_from)}"
                )
            at = now_utc()
            conn.execute(
                "UPDATE boxes SET status=?, updated_at=? WHERE box_id=?",
                (to_status, at, box_id),
            )
            conn.execute(
                "INSERT INTO box_events(box_id, seal_id, request_id, action, "
                "detail_json, at) VALUES (?, ?, ?, ?, '{}', ?)",
                (box_id, box["seal_id"], request_id, action, at),
            )
            response = {"request_id": request_id, "box_id": box_id,
                        "status": to_status, "seal_id": box["seal_id"],
                        "qty": box["qty"]}
            put_receipt(conn, request_id, operation, payload, response)
        return response

    @staticmethod
    def _box(conn, box_id: str) -> dict:
        row = conn.execute("SELECT * FROM boxes WHERE box_id=?", (box_id,)).fetchone()
        if row is None:
            raise NotFound(f"box {box_id!r} not found")
        return dict(row)

    # ------------------------------------------------------- buyer terms

    def set_buyer_terms(
        self,
        request_id: str,
        buyer: str,
        conditions: Iterable[str],
        pickup_deadline: str,
    ) -> dict[str, Any]:
        """The buyer's own licence: acceptable custody conditions + deadline."""
        require_nonempty(request_id, "request_id")
        require_nonempty(buyer, "buyer")
        require_nonempty(pickup_deadline, "pickup_deadline")
        conds = sorted({c for c in conditions if isinstance(c, str) and c.strip()})
        if not conds:
            raise DomainError("at least one acceptable condition is required")
        payload = {"buyer": buyer, "conditions": conds,
                   "pickup_deadline": pickup_deadline}
        operation = "box.set_buyer_terms"
        with self.db.tx() as conn:
            prior = get_receipt(conn, request_id, operation, payload)
            if prior is not None:
                return prior
            at = now_utc()
            conn.execute(
                "INSERT INTO buyer_terms(buyer, conditions_json, pickup_deadline, "
                "updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(buyer) DO UPDATE SET conditions_json=excluded.conditions_json, "
                "pickup_deadline=excluded.pickup_deadline, updated_at=excluded.updated_at",
                (buyer, json.dumps(conds), pickup_deadline, at),
            )
            response = {"request_id": request_id, "buyer": buyer,
                        "conditions": conds, "pickup_deadline": pickup_deadline}
            put_receipt(conn, request_id, operation, payload, response)
        return response

    def get_buyer_terms(self, buyer: str) -> dict[str, Any] | None:
        with self.db.read() as conn:
            return self._terms_conn(conn, buyer)

    @staticmethod
    def _terms_conn(conn, buyer: str) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM buyer_terms WHERE buyer=?", (buyer,)
        ).fetchone()
        if row is None:
            return None
        return {"buyer": buyer, "conditions": json.loads(row["conditions_json"]),
                "pickup_deadline": row["pickup_deadline"]}

    def accept_box(
        self, request_id: str, batch_id: str, buyer: str, box_id: str
    ) -> dict[str, Any]:
        """Record that ``buyer`` inspected ``box_id`` and accepts it.

        Acceptance is per buyer per box.  A box nobody accepted is usable by
        nobody; a box accepted by one buyer only is that buyer's exclusive
        option even if others would physically tolerate it.
        """
        require_nonempty(request_id, "request_id")
        require_nonempty(batch_id, "batch_id")
        require_nonempty(buyer, "buyer")
        require_nonempty(box_id, "box_id")
        payload = {"batch_id": batch_id, "buyer": buyer, "box_id": box_id}
        operation = "box.accept"
        with self.db.tx() as conn:
            prior = get_receipt(conn, request_id, operation, payload)
            if prior is not None:
                return prior
            box = self._box(conn, box_id)
            if box["batch_id"] != batch_id:
                raise DomainError(
                    f"box {box_id} belongs to batch {box['batch_id']}, not {batch_id}"
                )
            if box["status"] not in VOUCHERABLE_BOX_STATES:
                raise InvalidState(
                    f"box {box_id} is {box['status']}; only verified/stored boxes "
                    "can be accepted"
                )
            acceptance_id = f"acc-{batch_id}-{buyer}-{box_id}"
            conn.execute(
                "INSERT OR IGNORE INTO acceptances(acceptance_id, batch_id, buyer, "
                "box_id, created_at) VALUES (?, ?, ?, ?, ?)",
                (acceptance_id, batch_id, buyer, box_id, now_utc()),
            )
            response = {"request_id": request_id, "acceptance_id": acceptance_id,
                        "batch_id": batch_id, "buyer": buyer, "box_id": box_id,
                        "seal_id": box["seal_id"], "qty": box["qty"]}
            put_receipt(conn, request_id, operation, payload, response)
        return response

    # ------------------------------------------------------------ damage

    def damage_box(self, request_id: str, box_id: str, reason: str = "") -> dict:
        """Warehouse reports a loss.  A damaged box is never deliverable."""
        require_nonempty(request_id, "request_id")
        require_nonempty(box_id, "box_id")
        payload = {"box_id": box_id, "reason": reason}
        operation = "box.damage"
        with self.db.tx() as conn:
            prior = get_receipt(conn, request_id, operation, payload)
            if prior is not None:
                return prior
            box = self._box(conn, box_id)
            if box["status"] == "PICKED":
                raise InvalidState(f"box {box_id} already picked up; cannot damage")
            if box["status"] == "DAMAGED":
                response = {"request_id": request_id, "box_id": box_id,
                            "status": "DAMAGED", "duplicate": True}
                put_receipt(conn, request_id, operation, payload, response)
                return response
            if box["status"] not in USABLE_BOX_STATES:
                raise InvalidState(f"box {box_id} is {box['status']}")
            at = now_utc()
            conn.execute(
                "UPDATE boxes SET status='DAMAGED', updated_at=? WHERE box_id=?",
                (at, box_id),
            )
            conn.execute(
                "INSERT INTO box_events(box_id, seal_id, request_id, action, "
                "detail_json, at) VALUES (?, ?, ?, 'DAMAGE', ?, ?)",
                (box_id, box["seal_id"], request_id,
                 json.dumps({"reason": reason}, sort_keys=True), at),
            )
            response = {"request_id": request_id, "box_id": box_id,
                        "status": "DAMAGED", "duplicate": False}
            put_receipt(conn, request_id, operation, payload, response)
        return response

    # ------------------------------------------------------------ reseal

    def reseal(
        self,
        request_id: str,
        batch_id: str,
        old_box_id: str,
        new_boxes: Sequence[dict[str, Any]],
        authorized_by: Sequence[str],
    ) -> dict[str, Any]:
        """Officially replace one whole acceptable box by two received boxes.

        ``new_boxes`` is ``[{"seal_id": ..., "qty": ...}, ...]``; both must
        already be received and stored in the same batch.  The piece total
        must be conserved exactly.  ``authorized_by`` lists the buyers whose
        acceptance of the replacement is part of accepting the reseal; each
        authoriser automatically accepts every new box.

        The old box must still be physically present: a box already picked up
        can never be resealed.
        """
        require_nonempty(request_id, "request_id")
        require_nonempty(batch_id, "batch_id")
        require_nonempty(old_box_id, "old_box_id")
        if len(new_boxes) != 2:
            raise DomainError("a reseal replaces one box with exactly two boxes")
        parsed = []
        for nb in new_boxes:
            require_nonempty(nb.get("seal_id"), "new seal_id")
            require_int(nb.get("qty"), "new qty", minimum=1)
            parsed.append({"seal_id": nb["seal_id"], "qty": nb["qty"]})
        auth = sorted({b for b in authorized_by if isinstance(b, str) and b.strip()})
        if not auth:
            raise DomainError("a reseal requires at least one authorising buyer")
        payload = {"batch_id": batch_id, "old_box_id": old_box_id,
                   "new_boxes": parsed, "authorized_by": auth}
        operation = "box.reseal"

        with self.db.tx() as conn:
            prior = get_receipt(conn, request_id, operation, payload)
            if prior is not None:
                return prior
            old = self._box(conn, old_box_id)
            if old["batch_id"] != batch_id:
                raise DomainError(
                    f"box {old_box_id} is in batch {old['batch_id']}, not {batch_id}"
                )
            if old["status"] == "PICKED":
                raise InvalidState(
                    f"box {old_box_id} was already picked up; it cannot be resealed"
                )
            if old["status"] not in VOUCHERABLE_BOX_STATES:
                raise InvalidState(
                    f"box {old_box_id} is {old['status']}; only stored boxes can "
                    "be resealed"
                )
            if conn.execute(
                "SELECT 1 FROM withdrawals WHERE box_id=?", (old_box_id,)
            ).fetchone():
                raise InvalidState(
                    f"box {old_box_id} already left the dock; cannot reseal"
                )
            if conn.execute(
                "SELECT 1 FROM voucher_boxes WHERE box_id=?", (old_box_id,)
            ).fetchone():
                raise InvalidState(
                    f"box {old_box_id} is already on a live voucher; void it first"
                )
            new_rows = []
            for nb in parsed:
                row = conn.execute(
                    "SELECT * FROM boxes WHERE seal_id=?", (nb["seal_id"],)
                ).fetchone()
                if row is None:
                    raise NotFound(
                        f"replacement seal {nb['seal_id']!r} was never received"
                    )
                row = dict(row)
                if row["batch_id"] != batch_id:
                    raise DomainError(
                        f"replacement seal {nb['seal_id']} is in another batch"
                    )
                if row["qty"] != nb["qty"]:
                    raise ContentConflict(
                        f"replacement seal {nb['seal_id']} holds {row['qty']} pcs, "
                        f"declared {nb['qty']}"
                    )
                if row["status"] not in VOUCHERABLE_BOX_STATES:
                    raise InvalidState(
                        f"replacement box {row['box_id']} is {row['status']}"
                    )
                if row["box_id"] == old_box_id:
                    raise DomainError("replacement boxes must be distinct from the old box")
                new_rows.append(row)
            if len({r["box_id"] for r in new_rows}) != 2:
                raise DomainError("replacement boxes must be two distinct seals")
            total = sum(r["qty"] for r in new_rows)
            if total != old["qty"]:
                raise DomainError(
                    f"piece count not conserved: old {old['qty']} -> new {total}"
                )

            at = now_utc()
            reseal_id = f"reseal-{old_box_id}-{request_id}"
            conn.execute(
                "UPDATE boxes SET status='REPLACED', superseded_by=?, updated_at=? "
                "WHERE box_id=?",
                (",".join(r["box_id"] for r in new_rows), at, old_box_id),
            )
            conn.execute(
                "INSERT INTO reseals(reseal_id, old_box_id, new_boxes_json, "
                "total_qty, status, created_at) VALUES (?, ?, ?, ?, 'COMPLETE', ?)",
                (reseal_id, old_box_id,
                 json.dumps([r["box_id"] for r in new_rows]), total, at),
            )
            conn.execute(
                "INSERT INTO box_events(box_id, seal_id, request_id, action, "
                "detail_json, at) VALUES (?, ?, ?, 'RESEAL_OUT', ?, ?)",
                (old_box_id, old["seal_id"], request_id,
                 json.dumps({"reseal_id": reseal_id,
                             "new": [r["box_id"] for r in new_rows],
                             "total_qty": total}, sort_keys=True), at),
            )
            for r in new_rows:
                conn.execute(
                    "INSERT INTO box_events(box_id, seal_id, request_id, action, "
                    "detail_json, at) VALUES (?, ?, ?, 'RESEAL_IN', ?, ?)",
                    (r["box_id"], r["seal_id"], request_id,
                     json.dumps({"reseal_id": reseal_id,
                                 "old": old_box_id}, sort_keys=True), at),
                )
            # Authorising buyers accept the replacement boxes.
            for buyer in auth:
                for r in new_rows:
                    conn.execute(
                        "INSERT OR IGNORE INTO acceptances(acceptance_id, batch_id, "
                        "buyer, box_id, created_at) VALUES (?, ?, ?, ?, ?)",
                        (f"acc-{batch_id}-{buyer}-{r['box_id']}", batch_id, buyer,
                         r["box_id"], at),
                    )
            response = {
                "request_id": request_id,
                "reseal_id": reseal_id,
                "batch_id": batch_id,
                "old_box_id": old_box_id,
                "old_seal_id": old["seal_id"],
                "old_qty": old["qty"],
                "new_boxes": [
                    {"box_id": r["box_id"], "seal_id": r["seal_id"], "qty": r["qty"]}
                    for r in new_rows
                ],
                "total_qty": total,
                "authorized_by": auth,
            }
            put_receipt(conn, request_id, operation, payload, response)
        return response

    # ==================================================== delivery rights

    @staticmethod
    def _snapshot_orders(conn, auction_id: str) -> dict[str, int]:
        row = conn.execute(
            "SELECT allocations_json FROM settlements WHERE auction_id=?",
            (auction_id,),
        ).fetchone()
        if row is None:
            raise InvalidState(
                f"auction {auction_id} has no frozen result; freeze it first"
            )
        snap = json.loads(row["allocations_json"])
        awarded: dict[str, int] = {}
        for a in snap["allocations"]:
            if a["allocated_qty"] > 0:
                awarded[a["buyer"]] = awarded.get(a["buyer"], 0) + a["allocated_qty"]
        return awarded

    @staticmethod
    def _voucher_states(conn, auction_id: str) -> dict[str, dict[str, int]]:
        """Per buyer: pieces already picked and pieces on live vouchers."""
        picked: dict[str, int] = {}
        live: dict[str, int] = {}
        for v in conn.execute(
            "SELECT voucher_id, buyer, units, status FROM vouchers WHERE auction_id=?",
            (auction_id,),
        ):
            bucket = picked if v["status"] == "REDEEMED" else (
                live if v["status"] == "ISSUED" else {}
            )
            if bucket is not None:
                bucket[v["buyer"]] = bucket.get(v["buyer"], 0) + v["units"]
        return {"picked": picked, "live": live}

    def delivery_right(self, auction_id: str, buyer: str) -> dict[str, Any]:
        with self.db.read() as conn:
            awarded = self._snapshot_orders(conn, auction_id)
            states = self._voucher_states(conn, auction_id)
            terms = self._terms_conn(conn, buyer)
        total = awarded.get(buyer, 0)
        picked = states["picked"].get(buyer, 0)
        live = states["live"].get(buyer, 0)
        return {
            "auction_id": auction_id,
            "buyer": buyer,
            "awarded_units": total,
            "picked_units": picked,
            "vouchered_units": live,
            "remaining_units": total - picked - live,
            "terms": terms,
        }

    def _usable_boxes(self, conn, batch_id: str) -> list[dict]:
        """Verified boxes still physically present (not necessarily accepted)."""
        rows = conn.execute(
            "SELECT b.* FROM boxes b WHERE b.batch_id=? AND b.status='STORED' "
            "AND NOT EXISTS (SELECT 1 FROM withdrawals w WHERE w.box_id=b.box_id) "
            "AND NOT EXISTS (SELECT 1 FROM voucher_boxes vb WHERE vb.box_id=b.box_id) "
            "ORDER BY b.box_id",
            (batch_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def _buyer_acceptable_boxes(self, conn, batch_id: str, buyer: str,
                                boxes: list[dict]) -> set[str]:
        """Box ids this buyer may use: accepted, present, condition licensed."""
        terms = self._terms_conn(conn, buyer)
        if terms is None:
            return set()
        accepted = {
            r["box_id"]
            for r in conn.execute(
                "SELECT box_id FROM acceptances WHERE batch_id=? AND buyer=?",
                (batch_id, buyer),
            )
        }
        return {
            b["box_id"] for b in boxes
            if b["box_id"] in accepted and b["condition"] in terms["conditions"]
        }

    # ----------------------------------------------------------- planning

    def plan(
        self,
        batch_id: str,
        auction_id: str,
        today: str | None = None,
    ) -> dict[str, Any]:
        """Compute the joint voucher plan for one batch/auction.

        Each buyer gets at most one all-or-nothing voucher covering its whole
        frozen remaining quantity.  Objective: maximise fulfilled pieces,
        then the number of fulfilled buyers.  Boxes are never shared.
        """
        require_nonempty(batch_id, "batch_id")
        require_nonempty(auction_id, "auction_id")
        tday = _as_date(today) if today else _today()
        with self.db.read() as conn:
            batch = conn.execute(
                "SELECT * FROM intake_batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if batch is None:
                raise NotFound(f"batch {batch_id!r} not found")
            awarded = self._snapshot_orders(conn, auction_id)
            states = self._voucher_states(conn, auction_id)
            boxes = self._usable_boxes(conn, batch_id)
            buyers = sorted(awarded)

            remaining: dict[str, int] = {}
            reasons: dict[str, str] = {}
            eligible: list[str] = []
            allowed: dict[str, set[str]] = {}
            for buyer in buyers:
                left = (awarded.get(buyer, 0)
                        - states["picked"].get(buyer, 0)
                        - states["live"].get(buyer, 0))
                remaining[buyer] = left
                if left <= 0:
                    reasons[buyer] = "nothing_left"
                    continue
                terms = self._terms_conn(conn, buyer)
                if terms is None:
                    reasons[buyer] = "no_terms"
                    continue
                if _as_date(terms["pickup_deadline"]) < tday:
                    reasons[buyer] = "deadline_passed"
                    continue
                ok = self._buyer_acceptable_boxes(conn, batch_id, buyer, boxes)
                if not ok:
                    reasons[buyer] = "no_box_combination"
                    continue
                eligible.append(buyer)
                allowed[buyer] = ok

            assignment, per_buyer = _solve_fulfillment(
                eligible, remaining, allowed, boxes
            )
            assigned_boxes: dict[str, list[dict]] = {
                buyer: [{"box_id": b["box_id"], "seal_id": b["seal_id"],
                         "qty": b["qty"], "condition": b["condition"]}
                        for b in per_buyer.get(buyer, [])]
                for buyer in assignment
            }
            served = sorted(assignment)
            fulfilled_units = sum(remaining[b] for b in served)
            for buyer in buyers:
                if buyer in assignment:
                    reasons.pop(buyer, None)

            return {
                "batch_id": batch_id,
                "auction_id": auction_id,
                "today": tday,
                "boxes": [
                    {"box_id": b["box_id"], "seal_id": b["seal_id"],
                     "qty": b["qty"], "condition": b["condition"]}
                    for b in boxes
                ],
                "rights": [
                    {"buyer": b, "awarded_units": awarded.get(b, 0),
                     "picked_units": states["picked"].get(b, 0),
                     "vouchered_units": states["live"].get(b, 0),
                     "remaining_units": remaining[b],
                     "served": b in assignment,
                     "reason": reasons.get(b)}
                    for b in buyers
                ],
                "served_buyers": served,
                "unserved_buyers": [b for b in buyers if b not in assignment],
                "fulfilled_units": fulfilled_units,
                "fulfilled_buyers": len(served),
                "assignments": assigned_boxes,
            }

    # ------------------------------------------------------------ issuing

    def issue_vouchers(
        self,
        request_id: str,
        batch_id: str,
        auction_id: str,
        today: str | None = None,
    ) -> dict[str, Any]:
        """Issue the maximal joint plan as all-or-nothing vouchers."""
        require_nonempty(request_id, "request_id")
        require_nonempty(batch_id, "batch_id")
        require_nonempty(auction_id, "auction_id")
        tday = _as_date(today) if today else _today()
        payload = {"batch_id": batch_id, "auction_id": auction_id, "today": tday}
        operation = "box.issue_vouchers"
        with self.db.tx() as conn:
            prior = get_receipt(conn, request_id, operation, payload)
            if prior is not None:
                return prior
            # Re-plan atomically against live stock inside this transaction.
            planned = self._plan_in_conn(conn, batch_id, auction_id, tday)
            issued = []
            for buyer in planned["served_buyers"]:
                issued.append(self._insert_voucher(
                    conn, batch_id, auction_id, buyer,
                    planned["assignments"][buyer],
                    planned["rights_by_buyer"][buyer], request_id))
            self._check_voucher_cap(conn, batch_id)
            response = {
                "request_id": request_id,
                "batch_id": batch_id,
                "auction_id": auction_id,
                "today": tday,
                "fulfilled_units": planned["fulfilled_units"],
                "fulfilled_buyers": planned["fulfilled_buyers"],
                "served_buyers": planned["served_buyers"],
                "unserved_buyers": planned["unserved_buyers"],
                "vouchers": issued,
                "rights": planned["rights"],
            }
            put_receipt(conn, request_id, operation, payload, response)
        return response

    def issue_voucher_for_buyer(
        self,
        request_id: str,
        batch_id: str,
        auction_id: str,
        buyer: str,
        today: str | None = None,
    ) -> dict[str, Any]:
        """Issue one buyer's voucher independently (the non-joint kind).

        This is deliberately all-or-nothing: if the buyer's full remaining
        quantity cannot be met by whole accepted boxes, no voucher at all is
        issued.  It is never allowed to issue a half voucher that would
        consume a shared box and lock others out.
        """
        require_nonempty(request_id, "request_id")
        require_nonempty(batch_id, "batch_id")
        require_nonempty(auction_id, "auction_id")
        require_nonempty(buyer, "buyer")
        tday = _as_date(today) if today else _today()
        payload = {"batch_id": batch_id, "auction_id": auction_id,
                   "buyer": buyer, "today": tday}
        operation = "box.issue_voucher_for_buyer"
        with self.db.tx() as conn:
            prior = get_receipt(conn, request_id, operation, payload)
            if prior is not None:
                return prior
            awarded = self._snapshot_orders(conn, auction_id)
            states = self._voucher_states(conn, auction_id)
            remaining = (awarded.get(buyer, 0)
                         - states["picked"].get(buyer, 0)
                         - states["live"].get(buyer, 0))
            if remaining <= 0:
                raise NotEligible(f"{buyer} has no remaining delivery right")
            terms = self._terms_conn(conn, buyer)
            if terms is None:
                raise NotEligible(f"{buyer} has no pickup terms")
            if _as_date(terms["pickup_deadline"]) < tday:
                raise NotEligible(f"pickup deadline passed for {buyer}")
            boxes = self._usable_boxes(conn, batch_id)
            chosen = _choose_single_buyer_boxes(
                remaining, self._buyer_acceptable_boxes(conn, batch_id, buyer, boxes),
                boxes,
            )
            if not chosen:
                raise NotEligible(
                    f"{buyer} cannot be served in full ({remaining} units): no "
                    "whole-box combination; refusing to issue a partial voucher"
                )
            voucher = self._insert_voucher(conn, batch_id, auction_id, buyer,
                                           chosen, remaining, request_id)
            self._check_voucher_cap(conn, batch_id)
            response = {
                "request_id": request_id,
                "batch_id": batch_id,
                "auction_id": auction_id,
                "today": tday,
                "fulfilled_units": remaining,
                "fulfilled_buyers": 1,
                "served_buyers": [buyer],
                "unserved_buyers": [],
                "vouchers": [voucher],
            }
            put_receipt(conn, request_id, operation, payload, response)
        return response

    def _insert_voucher(self, conn, batch_id, auction_id, buyer, boxes, units,
                        request_id) -> dict:
        voucher_id = f"vch-{auction_id}-{buyer}-{request_id}"
        at = now_utc()
        conn.execute(
            "INSERT INTO vouchers(voucher_id, batch_id, auction_id, buyer, "
            "box_ids_json, units, status, issued_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'ISSUED', ?)",
            (voucher_id, batch_id, auction_id, buyer,
             json.dumps([b["box_id"] for b in boxes]), units, at),
        )
        for b in boxes:
            conn.execute(
                "INSERT INTO voucher_boxes(voucher_id, box_id) VALUES (?, ?)",
                (voucher_id, b["box_id"]),
            )
        return {
            "voucher_id": voucher_id,
            "buyer": buyer,
            "units": units,
            "boxes": [
                {"box_id": b["box_id"], "seal_id": b["seal_id"],
                 "qty": b["qty"], "condition": b["condition"]}
                for b in boxes
            ],
        }

    @staticmethod
    def _check_voucher_cap(conn, batch_id: str) -> None:
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM vouchers WHERE batch_id=?",
            (batch_id,),
        ).fetchone()["n"]
        if count > MAX_VOUCHERS_PER_BATCH:
            raise LimitExceeded(
                f"at most {MAX_VOUCHERS_PER_BATCH} vouchers per batch"
            )

    def _plan_in_conn(self, conn, batch_id: str, auction_id: str,
                      today: str) -> dict[str, Any]:
        """Same as ``plan`` but reusing the caller's transaction connection."""
        awarded = self._snapshot_orders(conn, auction_id)
        states = self._voucher_states(conn, auction_id)
        boxes = self._usable_boxes(conn, batch_id)
        buyers = sorted(awarded)
        remaining, reasons = {}, {}
        eligible: list[str] = []
        allowed: dict[str, set[str]] = {}
        for buyer in buyers:
            left = (awarded.get(buyer, 0)
                    - states["picked"].get(buyer, 0)
                    - states["live"].get(buyer, 0))
            remaining[buyer] = left
            if left <= 0:
                reasons[buyer] = "nothing_left"
                continue
            terms = self._terms_conn(conn, buyer)
            if terms is None:
                reasons[buyer] = "no_terms"
                continue
            if _as_date(terms["pickup_deadline"]) < today:
                reasons[buyer] = "deadline_passed"
                continue
            ok = self._buyer_acceptable_boxes(conn, batch_id, buyer, boxes)
            if not ok:
                reasons[buyer] = "no_box_combination"
                continue
            eligible.append(buyer)
            allowed[buyer] = ok

        assignment, per_buyer = _solve_fulfillment(
            eligible, remaining, allowed, boxes
        )
        assigned = {
            buyer: [{"box_id": b["box_id"], "seal_id": b["seal_id"],
                     "qty": b["qty"], "condition": b["condition"]}
                    for b in per_buyer.get(buyer, [])]
            for buyer in assignment
        }
        served = sorted(assignment)
        for buyer in buyers:
            if buyer in assignment:
                reasons.pop(buyer, None)
        rights_by_buyer = {b: remaining[b] for b in served}
        return {
            "rights": [
                {"buyer": b, "awarded_units": awarded.get(b, 0),
                 "picked_units": states["picked"].get(b, 0),
                 "vouchered_units": states["live"].get(b, 0),
                 "remaining_units": remaining[b],
                 "served": b in assignment, "reason": reasons.get(b)}
                for b in buyers
            ],
            "rights_by_buyer": rights_by_buyer,
            "served_buyers": served,
            "unserved_buyers": [b for b in buyers if b not in assignment],
            "fulfilled_units": sum(remaining[b] for b in served),
            "fulfilled_buyers": len(served),
            "assignments": assigned,
        }

    # ============================================================ pickup

    def pickup(
        self, request_id: str, voucher_id: str, window_id: str
    ) -> dict[str, Any]:
        """Hand over the boxes on a voucher at one dock window.

        Verifies live stock box by box; any damaged / already-withdrawn /
        replaced box aborts the whole handover (no partial delivery) and
        raises ``InventoryMismatch``.  A repeated request replays its receipt;
        a new request on an already-redeemed voucher is refused, so the same
        right or box cannot be delivered twice.
        """
        require_nonempty(request_id, "request_id")
        require_nonempty(voucher_id, "voucher_id")
        require_nonempty(window_id, "window_id")
        payload = {"voucher_id": voucher_id, "window_id": window_id}
        operation = "box.pickup"
        with self.db.tx() as conn:
            prior = get_receipt(conn, request_id, operation, payload)
            if prior is not None:
                return prior
            voucher = conn.execute(
                "SELECT * FROM vouchers WHERE voucher_id=?", (voucher_id,)
            ).fetchone()
            if voucher is None:
                raise NotFound(f"voucher {voucher_id!r} not found")
            if voucher["status"] == "REDEEMED":
                existing = conn.execute(
                    "SELECT request_id FROM pickup_receipts WHERE voucher_id=?",
                    (voucher_id,),
                ).fetchone()
                raise InvalidState(
                    "voucher already redeemed"
                    + (f" by request {existing['request_id']}" if existing else "")
                )
            if voucher["status"] != "ISSUED":
                raise InvalidState(f"voucher {voucher_id} is {voucher['status']}")
            box_ids = json.loads(voucher["box_ids_json"])
            boxes = []
            for box_id in box_ids:
                box = self._box(conn, box_id)
                if box["status"] != "STORED":
                    raise InventoryMismatch(
                        f"box {box_id} is {box['status']}; cannot hand it over"
                    )
                if conn.execute(
                    "SELECT 1 FROM withdrawals WHERE box_id=?", (box_id,)
                ).fetchone():
                    raise InventoryMismatch(
                        f"box {box_id} already left the dock at another window"
                    )
                boxes.append(box)
            if sum(b["qty"] for b in boxes) != voucher["units"]:
                raise DataIntegrityError(
                    f"voucher {voucher_id} pieces {sum(b['qty'] for b in boxes)} "
                    f"!= {voucher['units']}"
                )
            terms = self._terms_conn(conn, voucher["buyer"])
            if terms is None:
                raise NotEligible(f"buyer {voucher['buyer']} has no pickup terms")
            if _as_date(terms["pickup_deadline"]) < _today():
                raise NotEligible(
                    f"pickup deadline {terms['pickup_deadline']} has passed for "
                    f"{voucher['buyer']}"
                )
            for b in boxes:
                if b["condition"] not in terms["conditions"]:
                    raise NotEligible(
                        f"box {b['box_id']} condition {b['condition']} not accepted "
                        f"by {voucher['buyer']}"
                    )
            at = now_utc()
            for b in boxes:
                conn.execute(
                    "UPDATE boxes SET status='PICKED', updated_at=? WHERE box_id=?",
                    (at, b["box_id"]),
                )
                conn.execute(
                    "INSERT INTO withdrawals(withdrawal_id, box_id, window_id, "
                    "frozen_voucher_id, request_id, at) VALUES (?, ?, ?, ?, ?, ?)",
                    (f"wd-{b['box_id']}", b["box_id"], window_id, voucher_id,
                     f"{request_id}:{b['box_id']}", at),
                )
                conn.execute(
                    "INSERT INTO box_events(box_id, seal_id, request_id, action, "
                    "detail_json, at) VALUES (?, ?, ?, 'PICK', ?, ?)",
                    (b["box_id"], b["seal_id"], request_id,
                     json.dumps({"voucher_id": voucher_id,
                                 "window_id": window_id}, sort_keys=True), at),
                )
            conn.execute(
                "UPDATE vouchers SET status='REDEEMED', redeemed_at=? "
                "WHERE voucher_id=?",
                (at, voucher_id),
            )
            conn.execute(
                "INSERT INTO pickup_receipts(request_id, voucher_id, buyer, "
                "window_id, box_ids_json, at) VALUES (?, ?, ?, ?, ?, ?)",
                (request_id, voucher_id, voucher["buyer"], window_id,
                 json.dumps(box_ids), at),
            )
            response = {
                "request_id": request_id,
                "voucher_id": voucher_id,
                "auction_id": voucher["auction_id"],
                "buyer": voucher["buyer"],
                "window_id": window_id,
                "units": sum(b["qty"] for b in boxes),
                "boxes": [
                    {"box_id": b["box_id"], "seal_id": b["seal_id"],
                     "qty": b["qty"], "condition": b["condition"]}
                    for b in boxes
                ],
                "at": at,
            }
            put_receipt(conn, request_id, operation, payload, response)
        return response

    def withdraw_at_window(
        self,
        request_id: str,
        box_id: str,
        window_id: str,
        voucher_id: str | None = None,
    ) -> dict[str, Any]:
        """Record a physical box leaving at a window.

        With ``voucher_id`` this is an ordinary pickup.  Without it, it models
        an unexpected removal (another window / a recall): the box is gone for
        every future voucher and any confirmation for it must respect live
        stock instead of replaying its paper document.
        """
        if voucher_id is not None:
            return self.pickup(request_id, voucher_id, window_id)
        require_nonempty(request_id, "request_id")
        require_nonempty(box_id, "box_id")
        require_nonempty(window_id, "window_id")
        payload = {"box_id": box_id, "window_id": window_id, "voucher_id": None}
        operation = "box.withdraw_at_window"
        with self.db.tx() as conn:
            prior = get_receipt(conn, request_id, operation, payload)
            if prior is not None:
                return prior
            box = self._box(conn, box_id)
            if box["status"] == "PICKED":
                raise InventoryMismatch(f"box {box_id} already left the dock")
            if conn.execute(
                "SELECT 1 FROM withdrawals WHERE box_id=?", (box_id,)
            ).fetchone():
                raise InventoryMismatch(f"box {box_id} already withdrawn")
            at = now_utc()
            conn.execute(
                "UPDATE boxes SET status='PICKED', updated_at=? WHERE box_id=?",
                (at, box_id),
            )
            conn.execute(
                "INSERT INTO withdrawals(withdrawal_id, box_id, window_id, "
                "frozen_voucher_id, request_id, at) VALUES (?, ?, ?, NULL, ?, ?)",
                (f"wd-{box_id}", box_id, window_id, request_id, at),
            )
            conn.execute(
                "INSERT INTO box_events(box_id, seal_id, request_id, action, "
                "detail_json, at) VALUES (?, ?, ?, 'PICK', ?, ?)",
                (box_id, box["seal_id"], request_id,
                 json.dumps({"voucher_id": None, "window_id": window_id},
                            sort_keys=True), at),
            )
            response = {"request_id": request_id, "box_id": box_id,
                        "window_id": window_id, "units": box["qty"],
                        "voucher_id": None, "at": at}
            put_receipt(conn, request_id, operation, payload, response)
        return response

    # ----------------------------------------------------- receipt checks

    def void_voucher(self, request_id: str, voucher_id: str, reason: str = "") -> dict:
        """Void an uncollected voucher and return its boxes to usable stock.

        The buyer's frozen right is untouched: it simply becomes collectable
        again on a later voucher.  A voucher with any picked box cannot be
        voided (the pieces already left); re-seal the physical stock instead.
        """
        require_nonempty(request_id, "request_id")
        require_nonempty(voucher_id, "voucher_id")
        payload = {"voucher_id": voucher_id, "reason": reason}
        operation = "box.void_voucher"
        with self.db.tx() as conn:
            prior = get_receipt(conn, request_id, operation, payload)
            if prior is not None:
                return prior
            v = conn.execute(
                "SELECT * FROM vouchers WHERE voucher_id=?", (voucher_id,)
            ).fetchone()
            if v is None:
                raise NotFound(f"voucher {voucher_id!r} not found")
            if v["status"] == "VOID":
                response = {"request_id": request_id, "voucher_id": voucher_id,
                            "status": "VOID", "duplicate": True}
                put_receipt(conn, request_id, operation, payload, response)
                return response
            if v["status"] != "ISSUED":
                raise InvalidState(f"voucher {voucher_id} is {v['status']}")
            box_ids = json.loads(v["box_ids_json"])
            for box_id in box_ids:
                box = self._box(conn, box_id)
                if box["status"] in ("PICKED", "REPLACED"):
                    raise InvalidState(
                        f"box {box_id} is {box['status']}; voucher cannot be voided"
                    )
            at = now_utc()
            conn.execute(
                "UPDATE vouchers SET status='VOID' WHERE voucher_id=?", (voucher_id,)
            )
            conn.execute(
                "DELETE FROM voucher_boxes WHERE voucher_id=?", (voucher_id,)
            )
            response = {"request_id": request_id, "voucher_id": voucher_id,
                        "status": "VOID", "released_boxes": box_ids,
                        "buyer": v["buyer"], "duplicate": False}
            put_receipt(conn, request_id, operation, payload, response)
        return response

    def get_pickup(self, request_id: str) -> dict[str, Any] | None:
        """Recover a pickup receipt by request id (unknown/lost response)."""
        require_nonempty(request_id, "request_id")
        with self.db.read() as conn:
            row = conn.execute(
                "SELECT * FROM pickup_receipts WHERE request_id=?",
                (request_id,),
            ).fetchone()
            if row is None:
                return None
            boxes = []
            for box_id in json.loads(row["box_ids_json"]):
                b = conn.execute(
                    "SELECT box_id, seal_id, qty, condition, status FROM boxes "
                    "WHERE box_id=?",
                    (box_id,),
                ).fetchone()
                boxes.append(dict(b) if b else {"box_id": box_id, "status": None})
            return {
                "request_id": row["request_id"],
                "voucher_id": row["voucher_id"],
                "buyer": row["buyer"],
                "window_id": row["window_id"],
                "boxes": boxes,
                "at": row["at"],
            }

    def inspect_voucher(self, voucher_id: str) -> dict[str, Any]:
        """Re-read a voucher against live stock, box by box."""
        require_nonempty(voucher_id, "voucher_id")
        with self.db.read() as conn:
            v = conn.execute(
                "SELECT * FROM vouchers WHERE voucher_id=?", (voucher_id,)
            ).fetchone()
            if v is None:
                raise NotFound(f"voucher {voucher_id!r} not found")
            boxes = []
            for box_id in json.loads(v["box_ids_json"]):
                b = conn.execute(
                    "SELECT box_id, seal_id, qty, condition, status FROM boxes "
                    "WHERE box_id=?",
                    (box_id,),
                ).fetchone()
                withdrawn = conn.execute(
                    "SELECT window_id FROM withdrawals WHERE box_id=?", (box_id,)
                ).fetchone()
                boxes.append({
                    **(dict(b) if b else {"box_id": box_id}),
                    "withdrawn_at": withdrawn["window_id"] if withdrawn else None,
                })
            receipt = conn.execute(
                "SELECT request_id, window_id, at FROM pickup_receipts "
                "WHERE voucher_id=?",
                (voucher_id,),
            ).fetchone()
        broken = [
            b for b in boxes
            if b.get("status") != ("PICKED" if v["status"] == "REDEEMED" else "STORED")
            or (v["status"] != "REDEEMED" and b.get("withdrawn_at"))
        ]
        return {
            "voucher_id": voucher_id,
            "auction_id": v["auction_id"],
            "buyer": v["buyer"],
            "units": v["units"],
            "status": v["status"],
            "boxes": boxes,
            "pickup_receipt": dict(receipt) if receipt else None,
            "broken": broken,
        }

    # -------------------------------------------------------- statements

    def delivery_statement(
        self, batch_id: str, auction_id: str, today: str | None = None
    ) -> dict[str, Any]:
        """Who can collect what now, and which rights remain undelivered."""
        planned = self.plan(batch_id, auction_id, today=today)
        with self.db.read() as conn:
            vouchers = [
                dict(r) for r in conn.execute(
                    "SELECT v.voucher_id, v.buyer, v.units, v.status, "
                    "v.box_ids_json FROM vouchers v WHERE v.auction_id=? "
                    "ORDER BY v.voucher_id",
                    (auction_id,),
                )
            ]
            for v in vouchers:
                v["box_ids"] = json.loads(v.pop("box_ids_json"))
            owed = []
            picked = []
            for right in planned["rights"]:
                if right["remaining_units"] > 0:
                    owed.append(right)
                if right["picked_units"] > 0:
                    picked.append({"buyer": right["buyer"],
                                   "picked_units": right["picked_units"]})
            # Original auction credentials: the frozen allocation per buyer.
            row = conn.execute(
                "SELECT allocations_json FROM settlements WHERE auction_id=?",
                (auction_id,),
            ).fetchone()
            snap = json.loads(row["allocations_json"]) if row else {"allocations": []}
            original = [
                {"buyer": a["buyer"], "bid_id": a["bid_id"],
                 "allocated_qty": a["allocated_qty"],
                 "unit_price_cents": a["unit_price_cents"],
                 "order_cents": a["order_cents"], "hold_id": a["hold_id"]}
                for a in snap["allocations"] if a["allocated_qty"] > 0
            ]
            handed = [
                dict(r) for r in conn.execute(
                    "SELECT p.request_id, p.voucher_id, p.buyer, p.window_id, "
                    "p.box_ids_json, p.at FROM pickup_receipts p "
                    "JOIN vouchers v ON v.voucher_id=p.voucher_id "
                    "WHERE v.auction_id=? ORDER BY p.at",
                    (auction_id,),
                )
            ]
            for h in handed:
                h["box_ids"] = json.loads(h.pop("box_ids_json"))
        return {
            "batch_id": batch_id,
            "auction_id": auction_id,
            "original_credentials": original,
            "plan": planned,
            "vouchers": vouchers,
            "pickup_receipts": handed,
            "still_owed": owed,
            "already_picked": picked,
        }


# ============================================================== helpers


def _choose_single_buyer_boxes(
    target: int, allowed: set[str], boxes: Sequence[dict]
) -> list[dict]:
    """One whole-box subset summing exactly to ``target`` (or ``[]``).

    Prefers the fewest boxes, then the lexicographically smallest box-id
    sequence, for a deterministic document.
    """
    if target <= 0:
        return []
    cands = sorted(
        (b for b in boxes if b["box_id"] in allowed and b["qty"] <= target),
        key=lambda b: b["box_id"],
    )
    best: list[dict] | None = None

    def better(opt: list[dict]) -> bool:
        if best is None:
            return True
        return (len(opt), [b["box_id"] for b in opt]) < (
            len(best), [b["box_id"] for b in best]
        )

    def rec(i: int, remaining: int, acc: list[dict]) -> None:
        nonlocal best
        if remaining == 0:
            if better(acc):
                best = list(acc)
            return
        # Even taking every remaining box cannot reach the target.
        if sum(b["qty"] for b in cands[i:]) < remaining:
            return
        for j in range(i, len(cands)):
            b = cands[j]
            if b["qty"] > remaining:
                continue
            acc.append(b)
            rec(j + 1, remaining - b["qty"], acc)
            acc.pop()

    rec(0, target, [])
    return best or []


def _solve_fulfillment(
    eligible: Sequence[str],
    remaining: dict[str, int],
    allowed: dict[str, set[str]],
    boxes: Sequence[dict],
) -> tuple[list[str], dict[str, list[dict]]]:
    """Optimal whole-box fulfilment for one auction.

    A buyer is either served in full (whole boxes summing exactly to its
    remaining quantity) or not at all.  Therefore pieces delivered when a
    buyer is served equal that buyer's remaining quantity, and the objective
    reduces to choosing a feasible set of served buyers maximising
    ``(sum of their remaining, number of buyers)``.

    Solved by a DP over boxes with state = remaining demand per buyer.  Only
    boxes no larger than some unmet demand can help; total demand per auction
    is bounded by the auction's units, so the state space stays small.

    Returns ``(served_buyers, {buyer: [box, ...]})``.
    """
    buyers = list(eligible)
    n = len(buyers)
    if n == 0:
        return [], {}
    demands = tuple(remaining[b] for b in buyers)
    # Keep any box that some buyer with enough demand is allowed to use; all
    # other boxes can never appear on a valid full voucher.
    usable = [
        b for b in boxes
        if any(b["qty"] <= demands[j] and b["box_id"] in allowed[buyers[j]]
               for j in range(n))
    ]
    usable.sort(key=lambda b: b["box_id"])

    memo: dict[tuple[int, tuple[int, ...]], tuple[int, int, tuple[int, ...]]] = {}

    def rec(i: int, rem: tuple[int, ...]) -> tuple[int, int, tuple[int, ...]]:
        key = (i, rem)
        cached = memo.get(key)
        if cached is not None:
            return cached
        if i == len(usable):
            pieces = sum(demands[j] for j in range(n) if rem[j] == 0)
            count = sum(1 for j in range(n) if rem[j] == 0)
            result = (pieces, count, ())
            memo[key] = result
            return result
        q = usable[i]["qty"]
        # Option A: skip this box.
        best = rec(i + 1, rem)
        best_choice = (-1,) + best[2]
        # Option B: give this box to any buyer it fully fits.
        for j in range(n):
            if 0 < rem[j] and q <= rem[j] and usable[i]["box_id"] in allowed[buyers[j]]:
                new_rem = rem[:j] + (rem[j] - q,) + rem[j + 1:]
                sub = rec(i + 1, new_rem)
                cand_choice = (j,) + sub[2]
                if (sub[0], sub[1]) > (best[0], best[1]) or (
                    (sub[0], sub[1]) == (best[0], best[1])
                    and cand_choice < best_choice
                ):
                    best = (sub[0], sub[1], cand_choice)
                    best_choice = cand_choice
        result = (best[0], best[1], best_choice)
        memo[key] = result
        return result

    _, _, choice = rec(0, demands)
    per_buyer: dict[str, list[dict]] = {b: [] for b in buyers}
    final_rem = list(demands)
    for box, j in zip(usable, choice):
        if j >= 0:
            per_buyer[buyers[j]].append(box)
            final_rem[j] -= box["qty"]
    # Only fully met buyers count; any partial boxes are simply not used.
    served = [
        buyers[j] for j in range(n)
        if per_buyer[buyers[j]] and final_rem[j] == 0
    ]
    return served, {b: per_buyer[b] for b in served}
