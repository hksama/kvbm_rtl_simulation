"""Queueing delay model."""

from __future__ import annotations

from model.types import QueueingMetrics


def compute_queueing(
    arrival_rate: float,
    service_rate: float,
    service_time_us: float,
) -> QueueingMetrics:
    """
    M/M/1-style queueing approximation.

    arrival_rate: requests or ops per second
    service_rate: max sustainable service rate per second
    service_time_us: base service time in microseconds (without queue)
    """
    if service_rate <= 0:
        return QueueingMetrics(
            utilization=1.0,
            avg_latency_us=service_time_us * 100,
            p95_latency_us=service_time_us * 200,
            p99_latency_us=service_time_us * 500,
            saturated=True,
        )

    if arrival_rate >= service_rate:
        return QueueingMetrics(
            utilization=1.0,
            avg_latency_us=service_time_us * 100,
            p95_latency_us=service_time_us * 200,
            p99_latency_us=service_time_us * 500,
            saturated=True,
        )

    utilization = arrival_rate / service_rate

    # M/M/1 average wait = service_time / (1 - rho)
    avg_latency_us = service_time_us / (1.0 - utilization)

    # Approximate tail latencies using exponential assumption
    p95_latency_us = avg_latency_us * (1.0 + 2.0 * utilization)
    p99_latency_us = avg_latency_us * (1.0 + 4.6 * utilization)

    return QueueingMetrics(
        utilization=utilization,
        avg_latency_us=avg_latency_us,
        p95_latency_us=p95_latency_us,
        p99_latency_us=p99_latency_us,
        saturated=False,
    )


def queue_delay_us(queueing: QueueingMetrics, service_time_us: float) -> float:
    """Queue component of latency (wait beyond service time)."""
    return max(0.0, queueing.avg_latency_us - service_time_us)
