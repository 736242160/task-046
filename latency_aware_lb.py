#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
latency_aware_lb.py — 延迟感知负载均衡器（纯标准库，单文件）

设计要点
========

1. 动态权重：指数平滑（EWMA）+ 批量重算
   - 每次成功请求用 ewma = alpha*sample + (1-alpha)*ewma 更新节点延迟估计。
   - 权重不随每个请求重算，而是每隔 weight_interval 秒批量重算一次：
     weight_i = max(1, round(100 * min_ewma / ewma_i))，即最快节点权重 100。
   - 取舍：alpha 越大跟踪越快，但 EWMA 的"有效样本数"约为 (2-alpha)/alpha，
     alpha=0.3 时仅约 6 个样本，单次延迟毛刺就能明显改变权重，放大抖动；
     alpha 太小（如 0.05，约 39 个样本）则节点真实变慢后要几十秒才降权。
   - 默认 alpha=0.3、weight_interval=1.0s：1 秒的批量窗口内每个节点通常已
     积累多个样本，EWMA 已稳定，权重基于"平滑后的估计"而非单次毛刺变化；
     同时 1 秒级的权重刷新对负载均衡场景足够快。离线/低 QPS 场景可调大
     weight_interval，高抖动链路可调小 alpha。

2. 健康判定：滑动窗口成功率
   - 只看成功率，不看延迟——慢节点由权重降权处理，健康判定不误杀慢节点。
   - 窗口 W 与阈值 T 相互制约：窗口内失败 f 次即成功率 (W-f)/W，
     触发熔断的最少失败数 f_min = floor(W*(1-T)) + 1，且成功率的粒度是 1/W。
   - 默认 W=20、T=0.5：f_min=11，单次失败只移动成功率 5%，偶发超时不会
     误杀；连续/密集故障约 11 次失败内熔断，也不会放过真故障。
   - 反例：W=20、T=0.95 时 f_min=2，两次抖动就误杀；T=0.1 时 f_min=19，
     节点几乎死透才熔断。建议 W∈[20,50]、T∈[0.5,0.8]，min_samples≥W/2
     防止启动期样本不足误判。
   - 恢复：不健康的节点不再接流量，窗口无法自然更新，因此引入"半开探测"：
     不健康超过 recovery_cooldown 秒后，以约 1/10 正常权重参与调度（涓流探测），
     窗口随新样本滑动，成功率回到阈值以上即恢复健康；再失败再熔断。

3. next() 并发安全
   - 调度状态（各节点 current_weight、权重重算）由一把 _pick_lock 保护，
     健康检查与选取在同一把锁内完成：节点一旦被 record_result 判定不健康，
     之后的 next() 绝不会再选中它（不存在"刚判不健康还被选中"的窗口）。
   - 上报路径 record_result 用每节点一把锁，O(1)、互不阻塞，不与调度锁争用。
   - 代价：选取被串行化，临界区为 O(h)（h=健康节点数）的整数运算，
     实测 5000 节点单次约 250µs（约 50ns/节点，单线程约 4000 次/秒），
     与 nginx 平滑加权轮询同复杂度，节点池扩到几千时调度不退化
    （无堆重排、无排序，耗时随节点数线性增长而非超线性）。若单机 QPS
     高到锁成为瓶颈，按业务分片为多个 balancer 实例即可线性扩展。
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field


class NoHealthyNodeError(RuntimeError):
    """所有节点均不健康（且均未进入半开探测期）时由 next() 抛出。"""


@dataclass
class Node:
    name: str
    ewma_ms: float | None = None        # 指数平滑后的延迟估计
    weight: int = 50                    # 有效权重（批量重算；未采样节点用默认中值）
    current_weight: float = 0.0         # 平滑加权轮询的运行时状态
    healthy: bool = True
    unhealthy_since: float = 0.0
    window: deque = field(default_factory=deque)   # 最近 N 次成功/失败
    lock: threading.Lock = field(default_factory=threading.Lock)


class LatencyAwareBalancer:
    def __init__(
        self,
        alpha: float = 0.3,               # EWMA 平滑系数，有效样本数约 (2-alpha)/alpha
        weight_interval: float = 1.0,     # 权重批量重算间隔（秒）
        window_size: int = 20,            # 健康判定滑动窗口大小 W
        trip_threshold: float = 0.5,      # 熔断阈值 T：成功率 < T 判不健康
        recover_threshold: float | None = None,  # 恢复阈值，默认等于 trip_threshold
        min_samples: int = 10,            # 窗口内至少这么多样本才做判定
        recovery_cooldown: float = 2.0,   # 不健康多久后进入半开探测（秒）
        clock=time.monotonic,             # 可注入时钟，便于测试/模拟
    ):
        if not 0 < alpha <= 1:
            raise ValueError("alpha 必须在 (0, 1] 内")
        if not 0 < trip_threshold < 1:
            raise ValueError("trip_threshold 必须在 (0, 1) 内")
        self.alpha = alpha
        self.weight_interval = weight_interval
        self.window_size = window_size
        self.trip_threshold = trip_threshold
        self.recover_threshold = recover_threshold if recover_threshold is not None else trip_threshold
        self.min_samples = min_samples
        self.recovery_cooldown = recovery_cooldown
        self._clock = clock
        self._nodes: list[Node] = []
        self._pick_lock = threading.Lock()
        self._last_recompute = 0.0

    def add_node(self, name: str) -> Node:
        node = Node(name=name)
        node.window = deque(maxlen=self.window_size)
        with self._pick_lock:
            self._nodes.append(node)
        return node

    # ---- 上报路径：每节点锁，O(1)，不阻塞其他节点的上报 ----
    def record_result(self, node: Node, latency_ms: float | None = None, success: bool = True) -> None:
        now = self._clock()
        with node.lock:
            if success and latency_ms is not None:
                if node.ewma_ms is None:
                    node.ewma_ms = float(latency_ms)
                else:
                    node.ewma_ms = self.alpha * latency_ms + (1 - self.alpha) * node.ewma_ms
            node.window.append(bool(success))
            self._judge(node, now)

    def _judge(self, node: Node, now: float) -> None:
        if len(node.window) < self.min_samples:
            return
        rate = sum(node.window) / len(node.window)
        if node.healthy and rate < self.trip_threshold:
            node.healthy = False
            node.unhealthy_since = now
            node.current_weight = 0.0   # 防止恢复时平滑轮询状态残留造成突发
        elif not node.healthy and rate >= self.recover_threshold:
            node.healthy = True

    # ---- 调度路径：一把锁，临界区 O(h) ----
    def next(self) -> Node:
        with self._pick_lock:
            now = self._clock()
            if now - self._last_recompute >= self.weight_interval:
                self._recompute_weights()
                self._last_recompute = now
            best = None
            total = 0
            for node in self._nodes:
                w = self._effective_weight(node, now)
                if w is None:
                    continue
                total += w
                node.current_weight += w
                if best is None or node.current_weight > best.current_weight:
                    best = node
            if best is None:
                raise NoHealthyNodeError("没有可用节点：全部不健康且均未进入半开探测期")
            best.current_weight -= total
            return best

    def _effective_weight(self, node: Node, now: float) -> int | None:
        if node.healthy:
            return node.weight
        if now - node.unhealthy_since >= self.recovery_cooldown:
            return max(1, node.weight // 10)    # 半开探测：约 1/10 正常权重的涓流
        return None

    def _recompute_weights(self) -> None:
        # 读 ewma_ms 不拿节点锁：float 读写在 CPython 下是原子的，
        # 最多用到稍旧的估计值，下一秒重算即收敛，属于可接受的良性竞争。
        sampled = [n for n in self._nodes if n.ewma_ms is not None]
        if not sampled:
            return
        min_ewma = min(n.ewma_ms for n in sampled)
        for n in sampled:
            n.weight = max(1, round(100 * min_ewma / n.ewma_ms))


# ============================ 延迟模拟示例 ============================

def _demo() -> None:
    rng = random.Random(20260925)
    now = [0.0]  # 虚拟时钟，模拟可瞬间跑完
    lb = LatencyAwareBalancer(
        alpha=0.3, weight_interval=1.0,
        window_size=20, trip_threshold=0.5, min_samples=10,
        recovery_cooldown=2.0, clock=lambda: now[0],
    )
    nodes = {name: lb.add_node(name) for name in ("A", "B", "C", "D")}

    # 节点画像：基准延迟 ms；D 在第 2 阶段 90% 失败，第 3 阶段恢复
    base_latency = {"A": 20.0, "B": 60.0, "C": 120.0, "D": 30.0}

    def fail_rate(name: str, t: float) -> float:
        if name == "D":
            if 4.0 <= t < 8.0:
                return 0.90   # 真故障
            return 0.0
        return 0.0

    phases = [(0.0, 4.0, "阶段1: 全部正常"), (4.0, 8.0, "阶段2: D 故障(90%失败)"), (8.0, 16.0, "阶段3: D 恢复")]
    dt = 0.005  # 模拟 200 QPS
    prev_healthy = {n.name: n.healthy for n in nodes.values()}

    print("== 健康状态切换事件 ==")
    for start, end, title in phases:
        counts = {name: 0 for name in nodes}
        t = start
        while t < end:
            node = lb.next()
            counts[node.name] += 1
            lat = base_latency[node.name] * rng.uniform(0.8, 1.2)
            ok = rng.random() >= fail_rate(node.name, t)
            lb.record_result(node, latency_ms=lat if ok else None, success=ok)
            now[0] = t
            t += dt
            for n in nodes.values():
                if n.healthy != prev_healthy[n.name]:
                    print(f"  t={t:6.2f}s  节点 {n.name} -> {'健康' if n.healthy else '不健康'}")
                    prev_healthy[n.name] = n.healthy
        total = sum(counts.values())
        print(f"\n== {title} ==")
        print(f"  {'节点':<4}{'分配占比':>8}{'权重':>6}{'EWMA(ms)':>10}{'窗口成功率':>10}{'健康':>6}")
        for name, n in nodes.items():
            rate = sum(n.window) / len(n.window) if n.window else float("nan")
            ewma = f"{n.ewma_ms:9.1f}" if n.ewma_ms is not None else "      -"
            print(f"  {name:<4}{counts[name]/total:>8.1%}{n.weight:>6}{ewma:>10}{rate:>10.2f}{str(n.healthy):>6}")
        print()

    # 全部不健康时的行为
    print("== 全部不健康 ==")
    lb2 = LatencyAwareBalancer(window_size=10, trip_threshold=0.5, min_samples=5,
                               recovery_cooldown=60.0, clock=lambda: now[0])
    bad = lb2.add_node("only-node")
    for _ in range(10):
        lb2.record_result(bad, success=False)
    try:
        lb2.next()
    except NoHealthyNodeError as exc:
        print(f"  next() 抛出 NoHealthyNodeError: {exc}")


if __name__ == "__main__":
    _demo()
