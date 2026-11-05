"""Tests for the per-connection entity declaration/binding commit ledger.

The ledger is a *commit* ledger, not an intent window:

* declarations/bindings already emitted to the broker and idents already
  written to ``Connection.declared_entities`` are never rolled back when a
  ledger scope exits;
* an inner scope only appends to its parent when it commits - it can never
  rewrite, reorder or erase the parent's already committed prefix;
* every distinct completion surface (single-step ``exchange_declare``,
  the multi-step queue assembly, the two binding cuts, the
  ``maybe_declare`` cache gate) is recorded as its own commit kind.
"""

from __future__ import annotations

import pytest

from kombu import Connection, Consumer, Exchange, Queue
from kombu.common import maybe_declare
from kombu.declaration import (
    CACHE_COMMIT,
    CACHE_HIT,
    EXCHANGE_DECLARED,
    QUEUE_BOUND,
    QUEUE_DECLARED,
    DeclarationCommit,
    DeclarationLedger,
    record_commit,
)
from t.mocks import Transport


def get_conn():
    conn = Connection(transport=Transport)
    conn.connect()
    return conn


def kinds(commits):
    return [c.kind for c in commits]


class test_DeclarationLedger:

    def test_connection_owns_a_single_ledger(self):
        conn = get_conn()
        assert isinstance(conn._declaration_ledger, DeclarationLedger)
        # every scope opened on the connection shares the same ledger
        scope = conn.declaration_ledger()
        try:
            assert scope._ledger is conn._declaration_ledger
        finally:
            scope.__exit__(None, None, None)
        assert not conn._declaration_ledger.active

    def test_records_are_commits_with_monotonic_sequence(self):
        conn = get_conn()
        chan = conn.channel()
        x = Exchange('foo', 'direct')(chan)
        with conn.declaration_ledger() as scope:
            x.declare()
            x.declare()
            commits = scope.commits
        assert kinds(commits) == [EXCHANGE_DECLARED, EXCHANGE_DECLARED]
        assert commits[0].name == 'foo'
        assert commits[0].seq < commits[1].seq
        assert all(isinstance(c, DeclarationCommit) for c in commits)

    def test_recording_without_an_open_scope_is_a_pin_compatible_noop(self):
        conn = get_conn()
        chan = conn.channel()
        # no ledger scope open: broker side effects happen as before and no
        # state is kept anywhere
        Exchange('foo', 'direct')(chan).declare()
        assert 'exchange_declare' in chan
        assert conn._declaration_ledger.committed == ()
        record_commit(chan, EXCHANGE_DECLARED, name='foo')  # must not raise

    def test_nested_scope_does_not_write_through_parent_prefix(self):
        conn = get_conn()
        chan = conn.channel()
        x = Exchange('foo', 'direct')(chan)
        with conn.declaration_ledger() as outer:
            x.declare()
            prefix = outer.commits
            assert kinds(prefix) == [EXCHANGE_DECLARED]

            with conn.declaration_ledger() as inner:
                x.declare()
                # while the inner scope is active its commits stay on the
                # inner ledger, they do not bleed into the outer scope
                assert kinds(inner.commits) == [EXCHANGE_DECLARED]
                assert outer.commits == prefix

            # inner exit promotes the commits, append only: the already
            # committed prefix survives object-identical and in order
            assert tuple(outer.commits[:len(prefix)]) == tuple(prefix)
            assert kinds(outer.commits) == [
                EXCHANGE_DECLARED, EXCHANGE_DECLARED,
            ]
            assert outer.commits[-1].seq > outer.commits[0].seq

    def test_sibling_scopes_and_sequential_roots_do_not_share_commits(self):
        conn = get_conn()
        chan = conn.channel()
        x = Exchange('foo', 'direct')(chan)
        with conn.declaration_ledger() as outer:
            x.declare()
            with conn.declaration_ledger() as first_inner:
                x.declare()
            # the exited inner scope keeps its own commit record
            assert kinds(first_inner.commits) == [EXCHANGE_DECLARED]
            with conn.declaration_ledger() as second_inner:
                # a fresh inner scope does not inherit the sibling commits
                assert second_inner.commits == ()
                x.declare()
            assert kinds(outer.commits) == [
                EXCHANGE_DECLARED, EXCHANGE_DECLARED, EXCHANGE_DECLARED,
            ]
        # a brand new root scope starts empty even though the ledger keeps
        # the previously committed history
        assert conn._declaration_ledger.committed
        with conn.declaration_ledger() as second_root:
            assert second_root.commits == ()

    def test_inner_exit_never_undoes_broker_or_cache_side_effects(self):
        conn = get_conn()
        chan = conn.channel()
        q = Queue('ledq', Exchange('ledex', 'direct'), 'rk')
        with conn.declaration_ledger():
            with conn.declaration_ledger():
                assert maybe_declare(q, chan) is True
        # the cache write committed by maybe_declare is not popped
        assert hash(q) in conn.declared_entities
        # the broker commands stay emitted; the ledger never issues
        # compensating delete/unbind commands
        assert 'exchange_declare' in chan
        assert 'queue_declare' in chan
        assert 'queue_bind' in chan
        for undo in ('exchange_delete', 'queue_delete',
                     'queue_unbind', 'exchange_unbind'):
            assert undo not in chan

    def test_exception_in_inner_scope_keeps_committed_facts(self):
        conn = get_conn()
        chan = conn.channel()
        x = Exchange('foo', 'direct')(chan)
        with conn.declaration_ledger() as outer:
            x.declare()
            with pytest.raises(RuntimeError):
                with conn.declaration_ledger() as inner:
                    x.declare()
                    raise RuntimeError('boom')
            # the exchange_declare that completed before the error is a
            # committed fact, promoted to the parent - it is not rolled back
            assert kinds(inner.commits) == [EXCHANGE_DECLARED]
            assert kinds(outer.commits) == [
                EXCHANGE_DECLARED, EXCHANGE_DECLARED,
            ]
        assert 'exchange_delete' not in chan

    def test_two_channels_of_one_connection_share_the_ledger(self):
        conn = get_conn()
        chan1 = conn.channel()
        chan2 = conn.channel()
        assert chan1 is not chan2
        with conn.declaration_ledger() as scope:
            Exchange('foo', 'direct')(chan1).declare()
            Exchange('bar', 'direct')(chan2).declare()
        names = [c.name for c in scope.commits]
        assert names == ['foo', 'bar']

    def test_ledger_is_reset_when_the_connection_closes(self):
        conn = get_conn()
        scope = conn.declaration_ledger()
        scope.__enter__()
        assert conn._declaration_ledger.active
        conn.close()
        assert not conn._declaration_ledger.active
        scope.__exit__(None, None, None)


class test_SameConnectionTimeline:
    """Cross-subsystem causal ordering on ONE connection."""

    def test_producer_cache_gate_then_consumer_direct_redeclare(self):
        conn = Connection('memory://')
        producer = conn.Producer(serializer='json')
        x = Exchange('ordex', 'direct')
        q = Queue('ordq', exchange=x, routing_key='ordrk')

        with conn.declaration_ledger() as outer:
            producer.publish(
                {'n': 1}, exchange=x, routing_key='ordrk',
                declare=[q],
            )
            first = tuple(outer.commits)
            assert kinds(first) == [
                EXCHANGE_DECLARED, QUEUE_DECLARED, QUEUE_BOUND, CACHE_COMMIT,
            ]

            with conn.declaration_ledger() as inner:
                # second publish: cache gate short-circuits the whole
                # multi-step assembly to a single cache hit
                producer.publish(
                    {'n': 2}, exchange=x, routing_key='ordrk',
                    declare=[q],
                )
                # Consumer.declare bypasses maybe_declare and performs the
                # direct multi-step assembly again
                consumer = Consumer(conn, queues=[q], auto_declare=False)
                consumer.declare()
                assert kinds(inner.commits) == [
                    CACHE_HIT,
                    EXCHANGE_DECLARED, QUEUE_DECLARED, QUEUE_BOUND,
                ]
                # inner activity must not touch the outer prefix
                assert tuple(outer.commits) == first

            assert tuple(outer.commits[:len(first)]) == first
            assert kinds(outer.commits)[len(first):] == [
                CACHE_HIT,
                EXCHANGE_DECLARED, QUEUE_DECLARED, QUEUE_BOUND,
            ]

        # declared_entities write survives ledger exit
        assert hash(q) in conn.declared_entities

        # broker side effects are real, not intent-window state: the message
        # is still consumable after every ledger scope has exited
        received = []
        consumer = Consumer(
            conn, queues=[q], callbacks=[lambda body, msg: received.append(body)],
            no_ack=True, auto_declare=False,
        )
        with consumer:
            conn.drain_events(timeout=1)
        assert received == [{'n': 1}]
        conn.release()
