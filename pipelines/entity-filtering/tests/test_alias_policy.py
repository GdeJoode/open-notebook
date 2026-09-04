"""One alias policy, obeyed by both stages that could write one (PC.2).

The finding: `KGResolver` auto-registered an alias on a fuzzy match while concept
alignment, in the same pass, refused to register anything on the stated grounds
that merging identities must be a deliberate act (D-N4-9). One decision, two
opposite answers, and no test could tell.
"""

from __future__ import annotations

import inspect

from entity_filtering.config import KGResolutionConfig
from entity_filtering.resolution.kg_resolver import KGResolver


def test_no_stage_writes_an_alias_without_being_asked() -> None:
    """Both the config and the direct constructor default to OFF.

    Two defaults rather than one, because a caller constructing `KGResolver`
    directly bypasses `KGResolutionConfig` entirely — which is how one flag ends
    up meaning two things.
    """
    assert KGResolutionConfig().register_aliases is False
    signature = inspect.signature(KGResolver.__init__)
    assert signature.parameters["register_aliases"].default is False


async def test_a_fuzzy_match_registers_nothing_by_default() -> None:
    """The default resolver never reaches the repository's alias writer.

    Asserted against the real collaborator seam rather than the flag: a default
    that is read but not obeyed is the failure this test exists to catch.
    """
    calls: list[dict] = []

    class _Repo:
        async def register_alias(self, **kwargs):
            calls.append(kwargs)
            return True

    resolver = KGResolver(entity_repo=_Repo())
    await resolver._maybe_register_alias(
        canonical_entity_id="entity:x",
        alias_text="Gemeente Leudal",
        match_type="fuzzy",
        similarity_score=0.91,
    )
    assert calls == [], "an alias was written under the default policy"

    # And the opt-in still works, so the policy is a default and not a removal.
    resolver_optin = KGResolver(entity_repo=_Repo(), register_aliases=True)
    resolver_optin._repo = resolver._repo
    await resolver_optin._maybe_register_alias(
        canonical_entity_id="entity:x",
        alias_text="Gemeente Leudal",
        match_type="fuzzy",
        similarity_score=0.91,
    )
    assert len(calls) == 1
    assert calls[0]["method"] == "kg_resolver"


async def test_the_alias_counter_moves_when_an_alias_is_written() -> None:
    """`aliases_registered` was logged but never incremented.

    So the INFO line printed `0` for every run, including runs that had written
    aliases — a counter that cannot move, which reads as evidence that nothing
    happened. Asserted in both directions: it counts a successful write and does
    not count a failed one, because a counter that increments unconditionally is
    the same lie with the sign flipped.
    """

    class _Repo:
        def __init__(self, ok: bool) -> None:
            self.ok = ok

        async def register_alias(self, **kwargs):
            return self.ok

    for ok, expected in ((True, 1), (False, 0)):
        resolver = KGResolver(entity_repo=_Repo(ok), register_aliases=True)
        report = {"aliases_registered": 0}
        await resolver._maybe_register_alias(
            canonical_entity_id="entity:x",
            alias_text="Gemeente Leudal",
            match_type="fuzzy",
            similarity_score=0.91,
            report=report,
        )
        assert report["aliases_registered"] == expected, f"success={ok}"
