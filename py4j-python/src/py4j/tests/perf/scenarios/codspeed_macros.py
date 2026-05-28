"""Pytest-codspeed adapter for macro scenarios.

Lives alongside ``scenarios/micro.py`` and is excluded from default
pytest discovery (see ``conftest.py:collect_ignore``). Only invoked
explicitly: by the CodSpeed CI workflow (``.github/workflows/codspeed.yml``)
which passes this path to pytest with ``--codspeed``. The perf framework's
own harness (``python -m py4j.tests.perf``) has its own macro runner and
doesn't use this file.

Each parametrized entry becomes a separately tracked CodSpeed benchmark
on the dashboard, so regressions show up per-scenario rather than as one
opaque aggregate.

Scenario coverage rationale: macros covering the perf characteristics
most prone to regression on the py4j socket / call path. Two test
functions live here:

``test_macro_scenario`` — scenarios that share a long-running JVM
provided by the ``macro_gateway`` fixture. Each scenario reuses the
same gateway across rounds (setup once, measure many).

* ``X1-1``  — single-thread concurrent_1_thread (10k sequential calls).
              Baseline round-trip latency floor.
* ``X2-10k`` — iterate_javalist_10k. The canonical "N round-trips"
              anti-pattern; targets of the JavaIterator / bulk-fetch
              optimization area.
* ``X4``    — callback_sort_100_items. Java->Python callback hot path;
              the most complex code path in py4j.
* ``X6``    — pool_saturation_50_threads. Connection pool behavior
              under high concurrency, tail-latency sensitive.
* ``X7-16k`` / ``X8-16k`` — bytes recv/send 16k.
* ``XA-3``  — attribute walk depth (Python __getattr__ cache canary).
* ``XB``    — gateway reconnect against running JVM (per-connection
              setup cost).
* ``XC``    — callback infra overhead unused (delta vs X1-1).

``test_cold_start_scenario`` — scenarios that own their full JVM
lifecycle inside ``measure()``. Cannot share the fixture's JVM
because port 25333 is already in use; each round must spawn and
shut down its own subprocess.

* ``XD``    — full_cold_start_subprocess_first_call.

Adding more scenarios later is one parametrize entry per scenario.
"""

import pytest

from py4j.tests.perf.jvm import (
    JvmNotBuiltError,
    JvmStartupError,
    fresh_jvm,
)
from py4j.tests.perf.scenarios.macro import (
    X1_1Thread,
    X2_10k,
    X4_Callbacks,
    X6_PoolSaturation,
    X7_16k,
    X8_16k,
    XA_AttributeWalk3,
    XB_GatewayReconnect,
    XC_CallbackInfraOverheadUnused,
    XD_FullColdStart,
)


# X7-16k exercises the bytes-decoding recv path (decode_bytearray);
# X8-16k exercises the bytes-encoding send path. Together they cover
# both halves of the byte-codec / Nagle-sensitive bandwidth surface
# in CodSpeed CI. Without either, byte-codec optimizations are
# invisible to the per-PR dashboard — every prior macro returns
# int / list / void / callback and the byte path was a measurement
# blind spot.
#
# XA-XC add cold-start / attribute-walk / connection-setup / callback-
# infra measurement surfaces that share the long-running fixture JVM.
# Each maps 1:1 to a planned cold-start improvement PR (issue #557):
#   XA — attribute caches (Python __getattr__ memoization)
#   XB — Java per-connection command-prototype cache + background warm
#   XC — lazy CallbackClient (delta vs X1-1 = current overhead)
# XD lives in _COLD_START_SCENARIOS below — owns its JVM lifecycle.
# Without these, a per-PR CodSpeed dashboard would not see the wins.
_MACRO_SCENARIOS = [
    X1_1Thread, X2_10k, X4_Callbacks, X6_PoolSaturation,
    X7_16k, X8_16k,
    XA_AttributeWalk3, XB_GatewayReconnect,
    XC_CallbackInfraOverheadUnused,
]

# Scenarios whose measure() spawns its own JVM subprocess per round.
# They must NOT share the fixture's JVM — the fixture's JVM is already
# bound to the default port 25333, so a second spawn would race on
# bind. Each cold-start scenario is self-contained: spawn, connect,
# call, shut down.
_COLD_START_SCENARIOS = [
    XD_FullColdStart,
]


@pytest.fixture(scope="function")
def macro_gateway(request):
    """Fresh JVM per macro scenario, with the callback server enabled
    when the scenario class declares ``enable_callbacks = True``.

    Yields ``(gateway, scenario_cls)`` so the test function can both
    drive the JVM and re-instantiate the scenario for setup + measure.
    """
    scenario_cls = request.param
    enable_callbacks = getattr(scenario_cls, "enable_callbacks", False)
    try:
        with fresh_jvm(enable_callbacks=enable_callbacks) as gw:
            yield gw, scenario_cls
    except JvmNotBuiltError as e:
        pytest.skip(str(e))
    except JvmStartupError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize(
    "macro_gateway", _MACRO_SCENARIOS, indirect=True,
    ids=[cls.id for cls in _MACRO_SCENARIOS],
)
def test_macro_scenario(benchmark, macro_gateway):
    """Run one macro scenario through the benchmark fixture.

    pytest-codspeed (and pytest-benchmark) both honor the ``benchmark``
    fixture and handle iteration scaling automatically. Each
    ``measure()`` call is an expensive macro operation (thousands of
    calls or large-list iteration) so per-iteration measurement is
    appropriate.
    """
    gateway, scenario_cls = macro_gateway
    scenario = scenario_cls()
    if hasattr(scenario, "setup"):
        scenario.setup(gateway)
    benchmark(scenario.measure, gateway)


@pytest.mark.parametrize(
    "scenario_cls", _COLD_START_SCENARIOS,
    ids=[cls.id for cls in _COLD_START_SCENARIOS],
)
def test_cold_start_scenario(benchmark, scenario_cls):
    """Run a cold-start scenario that owns its JVM lifecycle.

    No shared fixture JVM — each ``measure()`` invocation spawns a
    fresh subprocess, runs one full cold-start cycle, and tears it
    down. Slow per-iteration (~300 ms on M-series + JDK 21, multiple
    seconds on slower hosts); CodSpeed adapts iteration count
    accordingly.

    Skips cleanly if the Java side hasn't been built — checked via the
    same ``verify_classpath`` path used by ``fresh_jvm``.
    """
    from py4j.tests.perf.jvm import verify_classpath
    try:
        verify_classpath()
    except JvmNotBuiltError as e:
        pytest.skip(str(e))
    scenario = scenario_cls()
    # Cold-start scenarios have no shared state; setup() is not
    # expected. measure() ignores the gateway arg (it owns its own).
    benchmark(scenario.measure, None)
