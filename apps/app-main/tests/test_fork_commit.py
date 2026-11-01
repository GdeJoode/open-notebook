"""``fork_commit()`` geeft de commit waaruit het image is gebouwd (F2-10, K2).

In de container zet de build ``SIGNAAL_FORK_COMMIT``; de HEAD van de host
(``SIGNAAL_FLOWS_COMMIT``, de gekoppelde flows) telt alleen voor de flows.
"""

from app_main.services import flow_register


def _vers(monkeypatch, **env):
    for naam in ("SIGNAAL_FORK_COMMIT", "SIGNAAL_FLOWS_COMMIT"):
        monkeypatch.delenv(naam, raising=False)
    for naam, waarde in env.items():
        monkeypatch.setenv(naam, waarde)
    flow_register.fork_commit.cache_clear()


def test_commit_uit_de_build(monkeypatch) -> None:
    _vers(monkeypatch, SIGNAAL_FORK_COMMIT="A" * 40, SIGNAAL_FLOWS_COMMIT="b" * 40)
    assert flow_register.fork_commit() == "a" * 40
    flow_register.fork_commit.cache_clear()


def test_flowcommit_telt_niet_voor_de_fork(monkeypatch) -> None:
    _vers(monkeypatch, SIGNAAL_FLOWS_COMMIT="b" * 40)
    assert flow_register.fork_commit() != "b" * 40
    _vers(monkeypatch, SIGNAAL_FORK_COMMIT="onbekend", SIGNAAL_FLOWS_COMMIT="b" * 40)
    assert flow_register.fork_commit() != "b" * 40
    flow_register.fork_commit.cache_clear()
