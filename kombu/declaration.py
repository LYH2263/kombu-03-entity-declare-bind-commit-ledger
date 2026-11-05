"""Per-connection commit ledger for entity declarations and bindings.

This module is a *commit* ledger, not a transaction:

* every entry (a :class:`DeclarationCommit`) is recorded **after** the
  corresponding broker command has been emitted, i.e. once the declaration
  or binding is an observable broker fact;
* leaving a ledger scope therefore never undoes anything - it never calls
  ``exchange_delete``/``queue_delete``/``queue_unbind`` and it never removes
  idents from ``Connection.declared_entities``;
* nested scopes are isolated while open: commits recorded inside an inner
  scope stay on that scope.  When the inner scope exits its commits are
  promoted to the parent scope by appending them, never by rewriting,
  reordering or erasing the parent's already committed prefix.

The ledger binds the three distinct declaration surfaces together on a
single connection:

* the single-step :meth:`kombu.Exchange.declare`,
* the multi-step :meth:`kombu.Queue.declare` assembly (exchange declare,
  queue declare, queue bind, plus one declare/bind cut per extra binding),
* the cache gate in :func:`kombu.common.maybe_declare` (cache hit versus a
  fresh declaration followed by a cache commit).

When no scope is open, :func:`record_commit` is a no-op, so the behaviour of
every declaration entry point is unchanged unless a host explicitly opens a
ledger.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

__all__ = (
    'DeclarationCommit', 'DeclarationLedger', 'LedgerScope',
    'record_commit', 'ledger_for_channel',
    'EXCHANGE_DECLARED', 'QUEUE_DECLARED',
    'QUEUE_BOUND', 'EXCHANGE_BOUND',
    'CACHE_HIT', 'CACHE_COMMIT',
)

#: An ``exchange.declare`` broker command was emitted.
EXCHANGE_DECLARED = 'exchange_declare'

#: A ``queue.declare`` broker command was emitted.
QUEUE_DECLARED = 'queue_declare'

#: A ``queue.bind`` broker command was emitted.
QUEUE_BOUND = 'queue_bind'

#: An ``exchange.bind`` broker command was emitted.
EXCHANGE_BOUND = 'exchange_bind'

#: ``maybe_declare`` found the entity ident in declared_entities and
#: skipped declaring it.
CACHE_HIT = 'cache_hit'

#: ``maybe_declare`` added the entity ident to declared_entities.
CACHE_COMMIT = 'cache_commit'


class DeclarationCommit(NamedTuple):
    """One committed declaration/binding step.

    Attributes
    ----------
        seq: Per-connection monotonic sequence number, used to prove the
            commit ordering across nested scopes.
        kind: One of the ``*_DECLARED``/``*_BOUND``/``CACHE_*`` constants.
        entity: The entity (Exchange/Queue) the commit belongs to.
        name: Name of the declared exchange/queue, or of the bound
            destination entity.
        exchange: Source exchange name for bind commits.
        routing_key: Binding key for bind commits.
        arguments: Binding/declaration arguments.
        ident: declared_entities hash for cache commits/hits.
    """

    seq: int
    kind: str
    entity: Any = None
    name: str = ''
    exchange: str = ''
    routing_key: str = ''
    arguments: Any = None
    ident: Optional[int] = None


class LedgerScope:
    """A single open commit boundary on a :class:`DeclarationLedger`."""

    def __init__(self, ledger: 'DeclarationLedger'):
        self._ledger = ledger
        self._commits: list[DeclarationCommit] = []

    @property
    def commits(self) -> tuple[DeclarationCommit, ...]:
        """Commits attributed to this scope, in causal order."""
        return tuple(self._commits)

    def _append(self, commit: DeclarationCommit) -> None:
        self._commits.append(commit)

    def _adopt(self, child: 'LedgerScope') -> None:
        # Promotion is append-only: this scope's already committed prefix
        # is never touched; the child's commits follow it in order.
        self._commits.extend(child._commits)

    def __enter__(self) -> 'LedgerScope':
        return self._ledger._enter(self)

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return self._ledger._exit(self)


class DeclarationLedger:
    """Connection-scoped registry of declaration/binding commit steps.

    Owned by a :class:`~kombu.Connection`; hosts open commit boundaries
    with :meth:`Connection.declaration_ledger`::

        with connection.declaration_ledger() as ledger:
            maybe_declare(queue, channel)
            ...
        ledger.commits

    The ledger never rolls broker state or ``declared_entities`` back:
    commits represent facts already emitted to the broker.
    """

    def __init__(self, client: Any = None):
        self.client = client
        self._scopes: list[LedgerScope] = []
        self._history: list[DeclarationCommit] = []
        self._seq = 0

    @property
    def active(self) -> bool:
        return bool(self._scopes)

    @property
    def committed(self) -> tuple[DeclarationCommit, ...]:
        """Commits promoted out of every closed root scope."""
        return tuple(self._history)

    def open(self) -> LedgerScope:
        """Open a new commit boundary (use as a context manager)."""
        return LedgerScope(self)

    def reset(self) -> None:
        """Drop open scopes and committed history.

        Used only when the connection lifecycle itself ends (close /
        URL switch / collect).  A plain (re)establishment does not reset
        the ledger: the host owns the scope and every recorded commit is a
        historical fact.  Broker side effects themselves are never undone.
        """
        self._scopes.clear()
        self._history.clear()

    def commit(self, kind: str, entity: Any = None, name: str = '',
               exchange: str = '', routing_key: str = '',
               arguments: Any = None, ident: Optional[int] = None) -> None:
        """Attribute a commit of ``kind`` to the innermost open scope.

        No-op unless a scope is open.
        """
        if self._scopes:
            self._seq += 1
            self._scopes[-1]._append(DeclarationCommit(
                seq=self._seq,
                kind=kind,
                entity=entity,
                name=name,
                exchange=exchange,
                routing_key=routing_key,
                arguments=arguments,
                ident=ident,
            ))

    def _enter(self, scope: LedgerScope) -> LedgerScope:
        self._scopes.append(scope)
        return scope

    def _exit(self, scope: LedgerScope) -> bool:
        # Promote even when the body raised: every recorded commit is a
        # broker fact that has already happened and cannot be popped back.
        if self._scopes and self._scopes[-1] is scope:
            self._scopes.pop()
        if self._scopes:
            self._scopes[-1]._adopt(scope)
        else:
            self._history.extend(scope._commits)
        # Never swallow exceptions: this is not a rollback context.
        return False


def ledger_for_channel(channel: Any) -> Optional['DeclarationLedger']:
    """Resolve the connection ledger reachable from ``channel``.

    Returns :const:`None` for channels that are not bound to a kombu
    :class:`~kombu.Connection` (including test doubles).
    """
    connection = getattr(channel, 'connection', None)
    if connection is None:
        return None
    client = getattr(connection, 'client', None)
    if client is None:
        return None
    ledger = getattr(client, '_declaration_ledger', None)
    if isinstance(ledger, DeclarationLedger):
        return ledger
    return None


def record_commit(channel: Any, kind: str, entity: Any = None, name: str = '',
                  exchange: str = '', routing_key: str = '',
                  arguments: Any = None, ident: Optional[int] = None) -> None:
    """Record ``kind`` on the ledger owning ``channel``'s connection.

    No-op unless a ledger scope is currently open.
    """
    ledger = ledger_for_channel(channel)
    if ledger is not None:
        ledger.commit(
            kind, entity=entity, name=name, exchange=exchange,
            routing_key=routing_key, arguments=arguments, ident=ident,
        )
