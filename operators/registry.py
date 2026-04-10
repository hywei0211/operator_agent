"""
算子仓库 - 持久化存储已验证的算子
避免重复生成，支持跨 GPU 的算子复用

v2: SQLite 存储，支持版本历史、并发安全、索引查询
"""
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

REGISTRY_DB_PATH = os.path.join(os.path.dirname(__file__), "../.operator_registry.db")
REGISTRY_JSON_PATH = os.path.join(os.path.dirname(__file__), "../.operator_registry.json")


@dataclass
class OperatorEntry:
    """仓库中的单条算子记录"""
    operator_name: str
    gpu_model: str
    backend: str                        # cuda / hip / ascendc / triton
    source_code: str
    header_code: str = ""
    build_flags: list = field(default_factory=list)
    launch_config: dict = field(default_factory=dict)

    # 验证结果
    correctness_passed: bool = False
    max_relative_error: float = 1.0
    bandwidth_utilization: float = 0.0
    verified_shapes: list = field(default_factory=list)

    # 元数据
    created_at: float = field(default_factory=time.time)
    iteration_count: int = 1
    optimizations_applied: list = field(default_factory=list)
    tags: list = field(default_factory=list)

    # 版本（SQLite 新增）
    version: int = 1
    prompt_version: str = ""  # 记录生成时使用的 prompt 版本
    verification_level: str = "none"  # none/static/llm_review/cpu_math/compiled/hw_verified/benchmarked

    @property
    def registry_key(self) -> str:
        return f"{self.operator_name}::{self.gpu_model}"

    def is_production_ready(self) -> bool:
        return self.correctness_passed and self.bandwidth_utilization >= 0.5


class OperatorRegistry:
    """
    算子仓库（SQLite 后端）

    核心功能：
    1. 存储已验证算子，避免重复生成
    2. 支持按 (算子名 + GPU型号) 快速查找
    3. 支持相似GPU之间的算子迁移提示
    4. 版本历史追踪
    5. 并发安全（WAL 模式 + 线程本地连接）
    """

    def __init__(self, registry_path: str = REGISTRY_DB_PATH):
        self.db_path = registry_path
        self._local = threading.local()
        self._init_db()
        self._migrate_json_if_needed()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS operators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operator_name TEXT NOT NULL,
                gpu_model TEXT NOT NULL,
                backend TEXT NOT NULL,
                source_code TEXT NOT NULL,
                header_code TEXT DEFAULT '',
                build_flags TEXT DEFAULT '[]',
                launch_config TEXT DEFAULT '{}',
                correctness_passed INTEGER DEFAULT 0,
                max_relative_error REAL DEFAULT 1.0,
                bandwidth_utilization REAL DEFAULT 0.0,
                verified_shapes TEXT DEFAULT '[]',
                created_at REAL NOT NULL,
                iteration_count INTEGER DEFAULT 1,
                optimizations_applied TEXT DEFAULT '[]',
                tags TEXT DEFAULT '[]',
                version INTEGER DEFAULT 1,
                is_current INTEGER DEFAULT 1,
                prompt_version TEXT DEFAULT '',
                verification_level TEXT DEFAULT 'none'
            );
            CREATE INDEX IF NOT EXISTS idx_op_gpu
                ON operators(operator_name, gpu_model);
            CREATE INDEX IF NOT EXISTS idx_current
                ON operators(operator_name, gpu_model, is_current);
            CREATE INDEX IF NOT EXISTS idx_backend
                ON operators(backend);
        """)
        # 兼容旧表：如果 verification_level 列不存在则加上
        try:
            conn.execute("SELECT verification_level FROM operators LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE operators ADD COLUMN verification_level TEXT DEFAULT 'none'")
        conn.commit()

    def _migrate_json_if_needed(self):
        """一次性从 .operator_registry.json 迁移到 SQLite"""
        json_path = self.db_path.replace('.db', '.json')
        if not os.path.exists(json_path):
            json_path = REGISTRY_JSON_PATH
        if not os.path.exists(json_path):
            return

        conn = self._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM operators").fetchone()[0]
        if count > 0:
            return  # 已有数据，不重复迁移

        try:
            with open(json_path) as f:
                raw = json.load(f)
            migrated = 0
            for key, val in raw.items():
                entry = OperatorEntry(**{k: v for k, v in val.items() if k != 'version'})
                self._insert_entry(entry)
                migrated += 1

            logger.info(f"[Registry] Migrated {migrated} entries from JSON to SQLite")
            # 备份旧文件
            backup = json_path + ".bak"
            os.rename(json_path, backup)
            logger.info(f"[Registry] JSON backup saved to {backup}")
        except Exception as e:
            logger.warning(f"[Registry] JSON migration failed: {e}")

    # ── 公开 API（与旧版完全兼容） ─────────────────────────

    def register(self, entry: OperatorEntry):
        """注册一个新算子（或更新已有的）"""
        existing = self.lookup(entry.operator_name, entry.gpu_model)
        if existing and existing.bandwidth_utilization > entry.bandwidth_utilization:
            logger.info(f"[Registry] Skipping {entry.registry_key}: existing version is better")
            return

        conn = self._get_conn()
        if existing:
            # 将旧版本标记为非当前
            conn.execute(
                "UPDATE operators SET is_current=0 WHERE operator_name=? AND gpu_model=? AND is_current=1",
                (entry.operator_name, entry.gpu_model)
            )
            entry.version = existing.version + 1

        self._insert_entry(entry)
        logger.info(f"[Registry] Registered: {entry.registry_key} v{entry.version} "
                    f"(BW={entry.bandwidth_utilization:.1%}, correct={entry.correctness_passed})")

    def lookup(self, operator_name: str, gpu_model: str) -> Optional[OperatorEntry]:
        """精确查找算子（当前版本）"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM operators WHERE operator_name=? AND gpu_model=? AND is_current=1",
            (operator_name, gpu_model)
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def find_similar(self, operator_name: str, gpu_model: str) -> Optional[OperatorEntry]:
        """
        模糊查找 - 寻找同算子在架构相近GPU上的实现
        用于加速新GPU的算子生成（迁移学习）
        """
        from knowledge_base.hardware_specs.gpu_database import get_gpu_spec
        target_spec = get_gpu_spec(gpu_model)

        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM operators WHERE operator_name=? AND is_current=1 AND correctness_passed=1",
            (operator_name,)
        ).fetchall()

        best: Optional[OperatorEntry] = None
        best_score = -1.0

        for row in rows:
            entry = self._row_to_entry(row)
            src_spec = get_gpu_spec(entry.gpu_model)
            if src_spec is None:
                continue
            score = 0.0
            if target_spec and src_spec.vendor == target_spec.vendor:
                score += 0.6
            if target_spec:
                fp16_ratio = min(src_spec.compute.fp16_tflops, target_spec.compute.fp16_tflops) / \
                             max(src_spec.compute.fp16_tflops, target_spec.compute.fp16_tflops)
                score += 0.4 * fp16_ratio
            if score > best_score:
                best_score = score
                best = entry

        if best and best_score > 0.5:
            logger.info(f"[Registry] Found similar operator for {gpu_model}: "
                        f"{best.gpu_model} (score={best_score:.2f})")
        return best if best_score > 0.5 else None

    def list_operators(self, gpu_model: str = None) -> list[OperatorEntry]:
        conn = self._get_conn()
        if gpu_model:
            rows = conn.execute(
                "SELECT * FROM operators WHERE gpu_model=? AND is_current=1", (gpu_model,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM operators WHERE is_current=1"
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def stats(self) -> dict:
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM operators WHERE is_current=1").fetchone()[0]
        prod_ready = conn.execute(
            "SELECT COUNT(*) FROM operators WHERE is_current=1 AND correctness_passed=1 AND bandwidth_utilization>=0.5"
        ).fetchone()[0]

        by_backend = {}
        for row in conn.execute(
            "SELECT backend, COUNT(*) as cnt FROM operators WHERE is_current=1 GROUP BY backend"
        ).fetchall():
            by_backend[row["backend"]] = row["cnt"]

        by_gpu = {}
        for row in conn.execute(
            "SELECT gpu_model, COUNT(*) as cnt FROM operators WHERE is_current=1 GROUP BY gpu_model"
        ).fetchall():
            by_gpu[row["gpu_model"]] = row["cnt"]

        return {
            "total": total,
            "production_ready": prod_ready,
            "by_backend": by_backend,
            "by_gpu": by_gpu,
        }

    # ── 新增 API ──────────────────────────────────────────

    def get_version_history(self, operator_name: str, gpu_model: str) -> list[OperatorEntry]:
        """返回算子的所有历史版本（按版本号排序）"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM operators WHERE operator_name=? AND gpu_model=? ORDER BY version",
            (operator_name, gpu_model)
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    # ── 内部方法 ──────────────────────────────────────────

    def _insert_entry(self, entry: OperatorEntry):
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO operators (
                operator_name, gpu_model, backend, source_code, header_code,
                build_flags, launch_config, correctness_passed, max_relative_error,
                bandwidth_utilization, verified_shapes, created_at, iteration_count,
                optimizations_applied, tags, version, is_current, prompt_version,
                verification_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
        """, (
            entry.operator_name, entry.gpu_model, entry.backend,
            entry.source_code, entry.header_code,
            json.dumps(entry.build_flags), json.dumps(entry.launch_config),
            int(entry.correctness_passed), entry.max_relative_error,
            entry.bandwidth_utilization, json.dumps(entry.verified_shapes),
            entry.created_at, entry.iteration_count,
            json.dumps(entry.optimizations_applied), json.dumps(entry.tags),
            entry.version, entry.prompt_version, entry.verification_level,
        ))
        conn.commit()

    def _row_to_entry(self, row: sqlite3.Row) -> OperatorEntry:
        keys = row.keys()
        return OperatorEntry(
            operator_name=row["operator_name"],
            gpu_model=row["gpu_model"],
            backend=row["backend"],
            source_code=row["source_code"],
            header_code=row["header_code"] or "",
            build_flags=json.loads(row["build_flags"]) if row["build_flags"] else [],
            launch_config=json.loads(row["launch_config"]) if row["launch_config"] else {},
            correctness_passed=bool(row["correctness_passed"]),
            max_relative_error=row["max_relative_error"],
            bandwidth_utilization=row["bandwidth_utilization"],
            verified_shapes=json.loads(row["verified_shapes"]) if row["verified_shapes"] else [],
            created_at=row["created_at"],
            iteration_count=row["iteration_count"],
            optimizations_applied=json.loads(row["optimizations_applied"]) if row["optimizations_applied"] else [],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            version=row["version"],
            prompt_version=row["prompt_version"] if "prompt_version" in keys else "",
            verification_level=row["verification_level"] if "verification_level" in keys else "none",
        )


# 全局单例
_registry: Optional[OperatorRegistry] = None

def get_registry() -> OperatorRegistry:
    global _registry
    if _registry is None:
        _registry = OperatorRegistry()
    return _registry
