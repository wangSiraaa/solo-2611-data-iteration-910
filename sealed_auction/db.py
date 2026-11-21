"""SQLite storage shared by the two services.

The auction service and margin warehouse are separate Python services, but
they may use a shared transaction (explicitly allowed by the problem) so a
bid revision and its collateral adjustment commit atomically.  Durable
intents/receipts are still used at the settlement boundary, where each
margin operation is its own idempotent transaction.
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS auctions (
    auction_id       TEXT PRIMARY KEY,
    units            INTEGER NOT NULL CHECK (units BETWEEN 1 AND 10),
    status           TEXT NOT NULL CHECK (status IN ('OPEN','FROZEN','SETTLED')),
    created_at       TEXT NOT NULL,
    frozen_at        TEXT,
    settled_at       TEXT
);

CREATE TABLE IF NOT EXISTS holds (
    hold_id          TEXT PRIMARY KEY,
    buyer            TEXT NOT NULL,
    auction_id       TEXT NOT NULL,
    amount_cents     INTEGER NOT NULL CHECK (amount_cents >= 0),
    status           TEXT NOT NULL CHECK (status IN ('ACTIVE','RELEASED','CONSUMED')),
    charged_cents    INTEGER NOT NULL DEFAULT 0 CHECK (charged_cents >= 0),
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_holds_buyer ON holds(buyer);
CREATE INDEX IF NOT EXISTS idx_holds_auction ON holds(auction_id);

CREATE TABLE IF NOT EXISTS bids (
    bid_id            TEXT PRIMARY KEY,
    auction_id        TEXT NOT NULL REFERENCES auctions(auction_id),
    buyer             TEXT NOT NULL,
    qty               INTEGER NOT NULL CHECK (qty > 0),
    unit_price_cents  INTEGER NOT NULL CHECK (unit_price_cents > 0),
    revision          INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
    status            TEXT NOT NULL CHECK (status IN ('ACTIVE','CANCELLED')),
    hold_id           TEXT NOT NULL REFERENCES holds(hold_id),
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
-- At most one valid bid per buyer per auction.
CREATE UNIQUE INDEX IF NOT EXISTS ux_bid_active_buyer
    ON bids(auction_id, buyer) WHERE status = 'ACTIVE';
-- A hold can back at most one valid bid at a time.
CREATE UNIQUE INDEX IF NOT EXISTS ux_bid_active_hold
    ON bids(hold_id) WHERE status = 'ACTIVE';
CREATE INDEX IF NOT EXISTS idx_bids_auction ON bids(auction_id);

CREATE TABLE IF NOT EXISTS bid_events (
    seq               INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id        TEXT NOT NULL,
    bid_id            TEXT NOT NULL,
    auction_id        TEXT NOT NULL,
    buyer             TEXT NOT NULL,
    action            TEXT NOT NULL CHECK (action IN ('PLACE','REVISE','CANCEL')),
    qty               INTEGER NOT NULL,
    unit_price_cents  INTEGER NOT NULL,
    revision          INTEGER NOT NULL,
    hold_id           TEXT NOT NULL,
    at                TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bid_events_auction ON bid_events(auction_id);

CREATE TABLE IF NOT EXISTS ledger_events (
    seq               INTEGER PRIMARY KEY AUTOINCREMENT,
    buyer             TEXT NOT NULL,
    kind              TEXT NOT NULL CHECK (kind IN ('DEPOSIT','FREEZE','RELEASE','CHARGE')),
    amount_cents      INTEGER NOT NULL CHECK (amount_cents > 0),
    hold_id           TEXT,
    auction_id        TEXT,
    request_id        TEXT NOT NULL,
    at                TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ledger_buyer ON ledger_events(buyer);
CREATE INDEX IF NOT EXISTS idx_ledger_hold ON ledger_events(hold_id);
CREATE INDEX IF NOT EXISTS idx_ledger_auction ON ledger_events(auction_id);

CREATE TABLE IF NOT EXISTS receipts (
    request_id        TEXT PRIMARY KEY,
    operation         TEXT NOT NULL,
    fingerprint       TEXT NOT NULL,
    response_json     TEXT NOT NULL,
    at                TEXT NOT NULL
);

-- One hold may be charged at most once per auction, independent of retries.
CREATE UNIQUE INDEX IF NOT EXISTS ux_charge_once
    ON ledger_events(auction_id, hold_id) WHERE kind = 'CHARGE';

CREATE TABLE IF NOT EXISTS settlements (
    auction_id             TEXT PRIMARY KEY REFERENCES auctions(auction_id),
    clearing_price_cents   INTEGER,
    allocations_json       TEXT NOT NULL,
    status                 TEXT NOT NULL CHECK (status IN ('FROZEN','CLEARING','CLEARED')),
    created_at             TEXT NOT NULL,
    cleared_at             TEXT
);

-- =====================================================================
-- Sealed whole-box warehouse, delivery right and pickup vouchers.
-- The auction allocates homogeneous *units*; physical fulfilment is by
-- whole sealed sample boxes received from suppliers.  Boxes are never
-- split: a delivery right of N units can only be met by whole boxes
-- whose sizes sum to exactly N.
-- =====================================================================

CREATE TABLE IF NOT EXISTS intake_batches (
    batch_id         TEXT PRIMARY KEY,
    max_boxes        INTEGER NOT NULL CHECK (max_boxes BETWEEN 1 AND 24),
    received_at      TEXT NOT NULL,
    received_boxes   INTEGER NOT NULL DEFAULT 0
);

-- A registered box.  The seal is the stable identity; content and custody
-- condition are declared at delivery.
CREATE TABLE IF NOT EXISTS boxes (
    box_id           TEXT PRIMARY KEY,
    seal_id          TEXT NOT NULL,
    qty              INTEGER NOT NULL CHECK (qty > 0),
    batch_id         TEXT NOT NULL REFERENCES intake_batches(batch_id),
    condition        TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'RECEIVED'
                     CHECK (status IN ('RECEIVED','STORED','PICKED','DAMAGED','REPLACED')),
    superseded_by    TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
-- One seal may be physically present only once; same seal + same content
-- re-arriving is a duplicate, a different content is a correction conflict.
CREATE UNIQUE INDEX IF NOT EXISTS ux_box_seal ON boxes(seal_id);
CREATE INDEX IF NOT EXISTS idx_box_batch ON boxes(batch_id);

CREATE TABLE IF NOT EXISTS box_events (
    seq              INTEGER PRIMARY KEY AUTOINCREMENT,
    box_id           TEXT NOT NULL,
    seal_id          TEXT NOT NULL,
    request_id       TEXT NOT NULL,
    action           TEXT NOT NULL CHECK (action IN (
                        'RECEIVE','RESTATE','STORE','ODD','DAMAGE',
                        'PICK','RESEAL_OUT','RESEAL_IN')),
    detail_json      TEXT NOT NULL,
    at               TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_box_events_box ON box_events(box_id);

-- A buyer's own licensing terms: which custody conditions it will accept and
-- the date by which it must collect.  Both gate every voucher for that buyer.
CREATE TABLE IF NOT EXISTS buyer_terms (
    buyer             TEXT PRIMARY KEY,
    conditions_json   TEXT NOT NULL,
    pickup_deadline   TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

-- A buyer declaring a specific inspected box acceptable.  Several buyers may
-- accept the same box; the allocator picks at most one of them.  Accepting a
-- box is what makes it usable for that buyer ("only seal X is acceptable to
-- buyer Y" is represented by only Y having an acceptance for X).
CREATE TABLE IF NOT EXISTS acceptances (
    acceptance_id    TEXT PRIMARY KEY,
    batch_id         TEXT NOT NULL REFERENCES intake_batches(batch_id),
    buyer            TEXT NOT NULL,
    box_id           TEXT NOT NULL REFERENCES boxes(box_id),
    created_at       TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_accept_buyer_box
    ON acceptances(batch_id, buyer, box_id);
CREATE INDEX IF NOT EXISTS idx_accept_box ON acceptances(box_id);
CREATE INDEX IF NOT EXISTS idx_accept_buyer ON acceptances(buyer);

-- Official re-seal, split of one accepted box into two accepted boxes with
-- the same total pieces.  Both halves must be already received.
CREATE TABLE IF NOT EXISTS reseals (
    reseal_id        TEXT PRIMARY KEY,
    old_box_id       TEXT NOT NULL REFERENCES boxes(box_id),
    new_boxes_json   TEXT NOT NULL,
    total_qty        INTEGER NOT NULL CHECK (total_qty > 0),
    status           TEXT NOT NULL,
    created_at       TEXT NOT NULL
);

-- Physical withdrawal window (a pickup clerk at the dock).  This is NOT the
-- buyer's fulfilment: it records that pieces physically left the warehouse.
-- `frozen_voucher_id` is the voucher that reserved those pieces, if any.
CREATE TABLE IF NOT EXISTS withdrawals (
    withdrawal_id    TEXT PRIMARY KEY,
    box_id           TEXT NOT NULL REFERENCES boxes(box_id),
    window_id        TEXT NOT NULL,
    frozen_voucher_id TEXT,
    request_id       TEXT NOT NULL UNIQUE,
    at               TEXT NOT NULL
);
-- A box physically leaves the dock at most once.
CREATE UNIQUE INDEX IF NOT EXISTS ux_withdraw_box ON withdrawals(box_id);

-- A pickup voucher (提货单) is all-or-nothing: it is only issued when the
-- frozen remaining delivery right of one buyer can be met by whole accepted
-- boxes summing to exactly that quantity.
CREATE TABLE IF NOT EXISTS vouchers (
    voucher_id       TEXT PRIMARY KEY,
    batch_id         TEXT NOT NULL REFERENCES intake_batches(batch_id),
    auction_id       TEXT NOT NULL,
    buyer            TEXT NOT NULL,
    box_ids_json     TEXT NOT NULL,
    units            INTEGER NOT NULL CHECK (units > 0),
    status           TEXT NOT NULL CHECK (status IN ('ISSUED','REDEEMED','VOID')),
    issued_at        TEXT NOT NULL,
    redeemed_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_voucher_buyer ON vouchers(buyer);
CREATE INDEX IF NOT EXISTS idx_voucher_batch ON vouchers(batch_id);

CREATE TABLE IF NOT EXISTS voucher_boxes (
    voucher_id       TEXT NOT NULL REFERENCES vouchers(voucher_id),
    box_id           TEXT NOT NULL REFERENCES boxes(box_id),
    PRIMARY KEY (voucher_id, box_id)
);
-- A box may appear on at most one live voucher, ever.
CREATE UNIQUE INDEX IF NOT EXISTS ux_voucher_box ON voucher_boxes(box_id);

-- Pickup receipts are idempotent per request id; each records exactly which
-- boxes were handed over, in which window, against which frozen voucher.
CREATE TABLE IF NOT EXISTS pickup_receipts (
    request_id       TEXT PRIMARY KEY,
    voucher_id       TEXT NOT NULL REFERENCES vouchers(voucher_id),
    buyer            TEXT NOT NULL,
    window_id        TEXT NOT NULL,
    box_ids_json     TEXT NOT NULL,
    at               TEXT NOT NULL
);
"""


class Database:
    """Thin SQLite wrapper using one connection per transaction/read.

    BEGIN IMMEDIATE serialises writers.  Two concurrent freeze attempts for
    the last available cent therefore cannot both observe the same balance.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _init_schema(self) -> None:
        conn = self.connect()
        try:
            if self.path != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = FULL")
            conn.executescript(SCHEMA)
        finally:
            conn.close()

    @contextlib.contextmanager
    def tx(self):
        conn = self.connect()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    @contextlib.contextmanager
    def read(self):
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()
