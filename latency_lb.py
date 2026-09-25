"""
latency_lb.py — 延迟感知负载均衡器（纯标准库，单文件）

核心机制
========
1. 权重：每个节点用 EWMA（指数加权移动平均）平滑近期响应延迟，
   权重 ∝ 1/EWMA延迟，再映射到一张固定规模的"槽位表"。
2. 调度：加权轮询 = 槽位表 + 游标，next() 为 O(1)，节点扩到几千个不退化。
3. 健康：滑动窗口成功率判定，与延迟完全解耦——慢节点只被降权，不会被误杀。
4. 恢复：不健康节点冷却后以少量"探测槽位"半开回归，窗口清空重新采样。

======================================================================
决策一：权重更新频率 vs 平滑系数 alpha
======================================================================
- alpha 决定 EWMA 的记忆长度（时间常数 ≈ 1/alpha 个样本）：
    alpha 大（如 0.5）→ 只记最近 ~2 个样本，权重跟着延迟抖动来回跳，
                        槽位表频繁重建，分配出现"振荡"；
    alpha 小（如 0.05）→ 记最近 ~20 个样本，平滑但节点真实变化要很久才反映。
  默认 alpha=0.2（记忆 ~5 个样本）：对单次毛刺不敏感，节点持续变化时
  约 5 次采样即可收敛到新水平，是响应性与稳定性的常见折中。
- update_every 决定多少次 report 后重建一次槽位表：
  重建是 O(表长+节点数)，摊销到每次请求可忽略；而且权重的任何变化
  若小于 1/表长 本来就不可见，比 EWMA 收敛速度更频繁地重建纯属浪费。
  默认 update_every=64（远大于 1/alpha=5 的收敛周期）。
  调优建议：流量大、节点状态变化快 → 降到 16~32；节点多/求稳 → 128+。
  alpha 不要超过 0.5，否则权重在追噪声而不是追趋势。

======================================================================
决策二：next() 的并发安全
======================================================================
方案：热路径无锁。
- 槽位表是不可变 tuple，重建时整体替换引用（CPython 下引用赋值原子），
  读侧永远看到一张完整一致的表，无需锁。
- 游标放在 threading.local() 里，每线程独立推进，next() 全程无锁。
- 兜底：next() 取出节点后复查 node.healthy（普通属性读，GIL 下原子），
  防止"刚被判不健康、但某线程还拿着旧表"的竞态把请求发过去。
- report() 用每节点一把的细粒度锁，临界区只有几次算术和一次 deque 操作，
  不同节点之间零竞争；表重建由独立的 sched_lock 保护，每 update_every
  次 report 才发生一次。
代价（明确取舍）：
- 每线程各自计数 → 全局序列不是严格的平滑交错，但每个线程的请求流各自
  均衡，聚合分布在统计上仍收敛到权重；换来的是热路径零锁竞争。
- 原子性依赖 CPython 的 GIL（引用赋值/属性读原子）。这是本方案的前提，
  也是"无锁"能成立的原因；复查 healthy 兜底了剩余的窄竞态窗口。
- 权重/健康变化最多延迟 update_every 次请求才反映到表中（健康状态变化
  会立即触发重建，只有权重变化是批量延迟的）。

======================================================================
健康判定：窗口大小 W 与阈值 τ 怎么配
======================================================================
原则：健康只看成功率，不看延迟。慢节点流量自然被权重降下去，
不需要"杀"；只有持续失败才摘出。这是"不误杀慢节点"的根本设计。

配置方法（二项分布近似）：观测成功率的波动标准差 ≈ sqrt(p(1-p)/W)。
- 不误杀健康节点（真实成功率 p0）：要求 τ ≤ p0 - 3*sqrt(p0(1-p0)/W)
- 不放过故障节点（真实成功率 p1）：要求 τ ≥ p1 + 3*sqrt(p1(1-p1)/W)
例：健康节点 p0=0.99、故障节点 p1=0.5、W=50：
  健康侧波动 3σ ≈ 0.04 → τ 上限 ≈ 0.95；故障侧 3σ ≈ 0.21 → τ 下限 ≈ 0.71。
  取 τ=0.8，两侧都有数倍 σ 的裕量，误杀和漏判概率都极低。
- W 越大判定越稳但越慢：故障要在该节点上积累约 W*(1-τ) 量级的失败才越界。
  W=50 配合 min_samples=20：既防止启动时"1 次失败=0% 成功率"的误判，
  又能在持续故障时几十次请求内摘出。
- min_samples 必须明显小于 W（建议 W 的 1/3~1/2），否则窗口还没滑满
  旧样本就主导判定，恢复会变慢。

复杂度
======
next() O(1)；report() 均摊 O(1)；表重建 O(表长+节点数)，
每 update_every 次请求最多一次；内存 O(表长+节点数)。几千节点无压力。
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque


class NoHealthyNodeError(RuntimeError):
    """所有节点都不健康（且没有到冷却期的探测节点）时由 next() 抛出。"""


class Node:
    __slots__ = (
        "name", "ewma", "window", "success_count",
        "healthy", "probing", "unhealthy_since", "lock",
    )

    def __init__(self, name: str, window_size: int):
        self.name = name
        self.ewma: float | None = None      # EWMA 延迟（秒），None 表示尚无样本
        self.window: deque[bool] = deque(maxlen=window_size)
        self.success_count = 0
        self.healthy = True
        self.probing = False                # 是否处于半开探测状态
        self.unhealthy_since = 0.0
        self.lock = threading.Lock()

    def __repr__(self) -> str:
        return f"Node({self.name})"


class LatencyAwareBalancer:
    def __init__(
        self,
        nodes,
        *,
        alpha: float = 0.2,            # EWMA 平滑系数，见模块 docstring 决策一
        window_size: int = 50,         # 健康判定滑动窗口 W
        min_success_rate: float = 0.8, # 健康阈值 τ
        min_samples: int = 20,         # 窗口内至少这么多样本才做判定
        update_every: int = 64,        # 多少次 report 重建一次槽位表
        table_slots: int | None = None,  # 槽位表规模，默认 max(1024, 4*节点数)
        recovery_cooldown: float = 5.0,  # 不健康后多少秒给探测流量
        clock=time.monotonic,
    ):
        if not (0 < alpha <= 1):
            raise ValueError("alpha 必须在 (0, 1]")
        if not (0 < min_success_rate < 1):
            raise ValueError("min_success_rate 必须在 (0, 1)")
        if not (0 < min_samples <= window_size):
            raise ValueError("min_samples 必须在 (0, window_size]")
        names = list(nodes)
        if not names:
            raise ValueError("节点列表不能为空")
        self.alpha = alpha
        self.min_success_rate = min_success_rate
        self.min_samples = min_samples
        self.update_every = update_every
        self.table_slots = table_slots or max(1024, 4 * len(names))
        self.probe_slots = max(1, self.table_slots // 64)
        self.recovery_cooldown = recovery_cooldown
        self._clock = clock
        self._nodes = [Node(n, window_size) for n in names]
        self._by_name = {nd.name: nd for nd in self._nodes}
        self._sched_lock = threading.Lock()   # 只保护计数器与表重建
        self._since_update = 0
        self._local = threading.local()       # 每线程游标
        self._table: tuple[Node, ...] = ()
        with self._sched_lock:
            self._rebuild_locked()

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    @property
    def nodes(self) -> list[Node]:
        return list(self._nodes)

    def node(self, name: str) -> Node:
        return self._by_name[name]

    def next(self) -> Node:
        """取一个节点。全部不健康时抛 NoHealthyNodeError。无锁热路径。"""
        table = self._table
        if not table:
            # 可能是全部不健康后冷却期已到但还没人触发重建，补一次重建再判
            with self._sched_lock:
                if not self._table:
                    self._rebuild_locked()
            table = self._table
            if not table:
                raise NoHealthyNodeError("没有可用节点：全部节点都不健康")
        cursor = getattr(self._local, "cursor", None)
        if cursor is None:
            cursor = random.randrange(len(table))  # 各线程错开起点
        size = len(table)
        now = None
        for _ in range(size):
            node = table[cursor % size]
            cursor += 1
            if node.healthy:
                self._local.cursor = cursor
                return node
            # 表里的探测节点：健康标志仍是 False，但冷却期已过允许服务
            if now is None:
                now = self._clock()
            if now - node.unhealthy_since >= self.recovery_cooldown:
                self._local.cursor = cursor
                return node
        self._local.cursor = cursor
        raise NoHealthyNodeError("没有可用节点：全部节点都不健康")

    def report(self, node: Node, latency: float | None, success: bool = True) -> None:
        """上报一次请求结果。latency 单位秒；失败时延迟不计入 EWMA。"""
        rebuild = False
        with node.lock:
            if success and latency is not None and latency > 0:
                if node.ewma is None:
                    node.ewma = latency
                else:
                    node.ewma = self.alpha * latency + (1 - self.alpha) * node.ewma
            if len(node.window) == node.window.maxlen and node.window[0]:
                node.success_count -= 1  # 滑出窗口的旧样本
            node.window.append(success)
            if success:
                node.success_count += 1
            if len(node.window) >= self.min_samples:
                rate = node.success_count / len(node.window)
                now = self._clock()
                if node.healthy and rate < self.min_success_rate:
                    node.healthy = False
                    node.probing = False
                    node.unhealthy_since = now
                    rebuild = True
                elif not node.healthy:
                    if rate >= self.min_success_rate:
                        node.healthy = True
                        node.probing = False
                        rebuild = True
                    else:
                        node.unhealthy_since = now  # 探测仍失败，重新冷却
        with self._sched_lock:
            self._since_update += 1
            if rebuild or self._since_update >= self.update_every:
                self._rebuild_locked()
                self._since_update = 0

    def stats(self) -> list[dict]:
        """当前状态快照（调试用）：每节点的健康、EWMA、槽位数。"""
        table = self._table
        slots = {nd.name: 0 for nd in self._nodes}
        for nd in table:
            slots[nd.name] += 1
        return [
            {
                "name": nd.name,
                "healthy": nd.healthy,
                "probing": nd.probing,
                "ewma": nd.ewma,
                "slots": slots[nd.name],
                "success_rate": (
                    node_rate if (node_rate := self._rate(nd)) is not None else None
                ),
            }
            for nd in self._nodes
        ]

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    @staticmethod
    def _rate(node: Node) -> float | None:
        if not node.window:
            return None
        return node.success_count / len(node.window)

    def _rebuild_locked(self) -> None:
        """按当前权重重建槽位表并原子替换。调用方须持有 _sched_lock。"""
        now = self._clock()
        known = [nd.ewma for nd in self._nodes if nd.ewma is not None]
        fallback = sum(known) / len(known) if known else 1.0
        entries: list[list] = []  # [node, slots]
        weights = []
        for nd in self._nodes:
            if nd.healthy:
                lat = nd.ewma if nd.ewma is not None else fallback
                weights.append((nd, 1.0 / lat))
            elif now - nd.unhealthy_since >= self.recovery_cooldown:
                if not nd.probing:
                    with nd.lock:  # 半开：清空旧窗口重新采样
                        nd.window.clear()
                        nd.success_count = 0
                    nd.probing = True
                entries.append([nd, self.probe_slots])
        total_w = sum(w for _, w in weights)
        for nd, w in weights:
            entries.append([nd, max(1, round(self.table_slots * w / total_w))])
        total = sum(s for _, s in entries)
        if total == 0:
            self._table = ()
            return
        # 均匀交错放置槽位，保证轮询序列平滑
        table: list[Node | None] = [None] * total
        for i, (nd, slots) in enumerate(entries):
            start = (i * total) // len(entries)
            step = total / slots
            for k in range(slots):
                idx = int(start + (k + 0.5) * step) % total
                while table[idx] is not None:
                    idx = (idx + 1) % total
                table[idx] = nd
        self._table = tuple(table)  # 引用整体替换，读侧无锁可见


# ======================================================================
# 延迟模拟示例：直接运行 python3 latency_lb.py 验证分配与健康切换
# ======================================================================
if __name__ == "__main__":
    from collections import Counter

    random.seed(7)
    lb = LatencyAwareBalancer(
        ["fast-5ms", "mid-20ms", "slow-80ms", "flaky-15ms"],
        alpha=0.2,
        window_size=50,
        min_success_rate=0.8,
        min_samples=20,
        update_every=64,
        recovery_cooldown=0.3,  # 演示用短冷却；生产建议 5s+
    )
    base_lat = {"fast-5ms": 5.0, "mid-20ms": 20.0, "slow-80ms": 80.0, "flaky-15ms": 15.0}
    fail_rate = {name: 0.0 for name in base_lat}

    def run_phase(title: str, n: int) -> None:
        counts: Counter[str] = Counter()
        for _ in range(n):
            nd = lb.next()
            counts[nd.name] += 1
            ok = random.random() >= fail_rate[nd.name]
            lat = max(0.5, random.gauss(base_lat[nd.name], base_lat[nd.name] * 0.2))
            lb.report(nd, lat / 1000.0, ok)
        print(f"\n== {title} ==")
        for st in lb.stats():
            ewma = f"{st['ewma'] * 1000:6.1f}ms" if st["ewma"] else "   --  "
            state = "健康" if st["healthy"] else ("探测中" if st["probing"] else "不健康")
            print(
                f"  {st['name']:10s} ewma={ewma} {state:4s} "
                f"槽位={st['slots']:4d} 实分得={counts[st['name']]:5d} "
                f"({counts[st['name']] / n * 100:5.1f}%)"
            )

    # 阶段1：全部健康，分配应正比于 1/延迟 ≈ 60.8% / 15.2% / 3.8% / 20.3%
    run_phase("阶段1：全部健康，按 1/延迟 加权", 4000)

    # 阶段2：flaky 90% 失败，应被判定不健康并摘出
    fail_rate["flaky-15ms"] = 0.9
    run_phase("阶段2：flaky 90% 失败，应被摘出", 4000)

    # 阶段3：flaky 恢复，冷却后经探测重新上线
    fail_rate["flaky-15ms"] = 0.0
    time.sleep(0.35)
    run_phase("阶段3：flaky 恢复，探测后重新上线", 6000)

    # 阶段4：全部故障，next() 必须明确报错
    for name in fail_rate:
        fail_rate[name] = 1.0
    print("\n== 阶段4：全部故障 ==")
    try:
        for _ in range(4000):
            nd = lb.next()
            lb.report(nd, 0.01, False)
        print("  问题：没有抛出 NoHealthyNodeError！")
    except NoHealthyNodeError as exc:
        print(f"  next() 正确抛出 NoHealthyNodeError: {exc}")
