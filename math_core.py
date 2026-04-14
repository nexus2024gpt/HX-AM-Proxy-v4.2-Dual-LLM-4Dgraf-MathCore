# math_core.py — HX-AM v4.2 Math Core
"""
MathCore — вычислительный движок для HX-AM v4.2.

Режим 1 (StressTest):
  Принимает 4D-матрицу → запускает ODE-симуляцию →
  расшатывает параметры → возвращает stability_score, λ_max, границы устойчивости.

Режим 2 (ResonanceMatcher):
  Принимает 4D-вектор → ищет изоморфные артефакты в архиве →
  масштабирует параметры в целевой домен → вычисляет P(A→B).

Оптимизировано для i5-6300U / 8 GB RAM:
  N_OSCILLATORS_MAX = 200  (не 5000)
  T_MAX_FACTOR = 50        (50·τ вместо 100·τ)
  Без numba (избыточный overhead при одиночных вызовах)
  Кэш: sim_results/{id}_cache.json (лёгкий JSON, не .npz)

CLI-использование:
  python math_core.py --test
  python math_core.py --stress <artifact_id>
  python math_core.py --resonance <text>
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.distance import cosine as scipy_cosine

# Опциональный импорт nolds (Hurst exponent)
try:
    import nolds
    _NOLDS_AVAILABLE = True
except ImportError:
    _NOLDS_AVAILABLE = False

logger = logging.getLogger("HXAM.mathcore")

# ──────────────────────────────────────────────
# КОНСТАНТЫ
# ──────────────────────────────────────────────

N_OSCILLATORS_MAX = 200   # Ограничение по RAM
T_MAX_FACTOR = 50         # t_max = T_MAX_FACTOR × τ
STRESS_LEVELS = [0.1, 0.3, 0.5]     # уровни расшатывания (± от номинала)
STABILITY_THRESHOLD = 0.6            # r > threshold = стабильно
COHERENCE_LOSS_THRESHOLD = 0.4      # r < threshold = потеря когерентности

SIM_RESULTS_DIR = Path("sim_results")
INSIGHTS_DIR = Path("insights")
ARTIFACTS_DIR = Path("artifacts")
FOUR_D_INDEX = Path("artifacts/four_d_index.jsonl")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ══════════════════════════════════════════════
# СЛОЙ 1 — Симуляция (Kuramoto + Percolation)
# ══════════════════════════════════════════════

class KuramotoSimulator:
    """
    Модель Курамото с задержкой и шумом.

    dθ_i/dt = ω_i + (K/N)·Σ sin(θ_j − θ_i) + η·ξ(t)

    Возвращает: order_parameter r(t) и финальный r.
    """

    def __init__(self, N: int = 100, seed: int = 42):
        self.N = min(N, N_OSCILLATORS_MAX)
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        omega_i: float,
        K: float,
        eta: float,
        tau: float,
        t_end: Optional[float] = None,
    ) -> Dict[str, Any]:
        N = self.N
        t_max = T_MAX_FACTOR * max(tau, 0.1) if t_end is None else t_end
        t_span = (0.0, t_max)
        t_eval = np.linspace(0, t_max, min(500, int(t_max * 20)))

        # Начальные фазы — случайные равномерно
        theta0 = self.rng.uniform(0, 2 * math.pi, N)

        # Собственные частоты: Лоренциан или нормальное
        omegas = self.rng.normal(omega_i, max(omega_i * 0.1, 0.05), N)

        def rhs(t: float, theta: np.ndarray) -> np.ndarray:
            # CORRECTED: diff[i,j] = θ_j − θ_i → sin(θ_j − θ_i)
            # Детерминированный член; шум добавляется через post-step Euler
            diff = theta[None, :] - theta[:, None]  # (N, N): diff[i,j] = θ_j − θ_i
            coupling = (K / N) * np.sum(np.sin(diff), axis=1)
            return omegas + coupling

        try:
            sol = solve_ivp(
                rhs, t_span, theta0,
                t_eval=t_eval,
                method="RK45",
                rtol=1e-3, atol=1e-4,
                max_step=0.5,
            )
        except Exception as e:
            logger.warning(f"Kuramoto solve_ivp failed: {e}")
            return self._failed_result()

        if not sol.success:
            return self._failed_result()

        # Добавляем стохастическое возмущение к траектории (Euler-Maruyama post-processing)
        # Это правильнее чем добавлять шум внутри solve_ivp (который вызывает rhs несколько раз)
        if eta > 0 and len(t_eval) > 1:
            dt_mean = (t_eval[-1] - t_eval[0]) / len(t_eval)
            noise = eta * self.rng.normal(0, np.sqrt(dt_mean), sol.y.shape)
            sol_y = sol.y + noise
        else:
            sol_y = sol.y

        # Параметр порядка r(t) = |1/N Σ exp(i·θ_j)|
        r_series = np.abs(np.mean(np.exp(1j * sol_y), axis=0))
        r_final = float(r_series[-1]) if len(r_series) > 0 else 0.0
        r_mean = float(np.mean(r_series[-min(50, len(r_series)):]))

        return {
            "model": "kuramoto",
            "N": N,
            "t_max": t_max,
            "r_final": round(r_final, 4),
            "r_mean_last_10pct": round(r_mean, 4),
            "stable": r_mean > STABILITY_THRESHOLD,
        }

    def _failed_result(self) -> Dict[str, Any]:
        return {
            "model": "kuramoto",
            "N": self.N,
            "t_max": 0.0,
            "r_final": 0.0,
            "r_mean_last_10pct": 0.0,
            "stable": False,
        }


class PercolationSimulator:
    """
    Бернуллиевая перколяция на случайном графе.
    Находит порог p_c (наличие гигантской компоненты).
    """

    def __init__(self, N: int = 500):
        try:
            import networkx as nx
            self._nx = nx
        except ImportError:
            self._nx = None
        self.N = min(N, 1000)

    def run(self, p: float, k_mean: float) -> Dict[str, Any]:
        if self._nx is None:
            return {"model": "percolation", "giant_fraction": p, "stable": p > 0.5}

        nx = self._nx
        rng = np.random.default_rng(42)
        N = self.N

        # Создаём случайный граф Erdos-Renyi с вероятностью p
        p_edge = min(p, k_mean / max(N - 1, 1))
        G = nx.erdos_renyi_graph(N, p_edge, seed=42)
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        giant_size = len(components[0]) if components else 0
        giant_fraction = giant_size / N

        return {
            "model": "percolation",
            "N": N,
            "p_edge": round(p_edge, 4),
            "giant_fraction": round(giant_fraction, 4),
            "n_components": len(components),
            "stable": giant_fraction > 0.5,
        }


# ══════════════════════════════════════════════
# СЛОЙ 2 — Анализ устойчивости
# ══════════════════════════════════════════════

class StabilityAnalyzer:
    """
    Вычисляет максимальный показатель Ляпунова (упрощённый метод возмущений).
    Не требует numba — только numpy/scipy.
    """

    @staticmethod
    def lyapunov_estimate(
        omega_i: float,
        K: float,
        eta: float,
        N: int = 50,
        t_steps: int = 300,
    ) -> float:
        """
        Приближённый λ_max через разницу двух близких траекторий (метод Бенеттина).
        Запускается детерминированно (без шума) — шум добавляет диффузию,
        которая маскирует настоящий λ_max.
        Отрицательный → асимптотически устойчиво.
        Положительный → хаос или расходимость.
        """
        rng = np.random.default_rng(0)
        omegas = rng.normal(omega_i, max(omega_i * 0.1, 0.01), N)
        dt = 0.05

        def step_det(theta: np.ndarray) -> np.ndarray:
            """Детерминированный шаг Euler (без шума)."""
            diff = theta[None, :] - theta[:, None]
            coupling = (K / N) * np.sum(np.sin(diff), axis=1)
            return theta + (omegas + coupling) * dt

        # Начальные условия: вблизи синхронизованного состояния (если K > K_c)
        if K > 0:
            # Стартуем с малым разбросом фаз — ближе к аттрактору
            theta = rng.normal(0.0, 0.3, N)
        else:
            theta = rng.uniform(0, 2 * math.pi, N)

        # Прогрев: 100 шагов до установившегося режима
        for _ in range(100):
            theta = step_det(theta)

        delta0 = 1e-7
        theta_p = theta + rng.normal(0, delta0, N)

        log_growth = []
        for _ in range(t_steps):
            theta = step_det(theta)
            theta_p = step_det(theta_p)
            dist = float(np.linalg.norm(theta_p - theta))
            if dist < 1e-15 or dist > 1e3:
                break
            log_growth.append(math.log(dist / delta0))
            # Реортонормализация
            theta_p = theta + (theta_p - theta) / dist * delta0

        if not log_growth:
            return 0.0
        # λ_max = среднее log-growth в единицу времени
        return round(float(np.mean(log_growth)) / dt, 4)

    @staticmethod
    def find_critical_eta(
        omega_i: float,
        K: float,
        K_c: float,
        tau: float,
        eta_range: Tuple[float, float] = (0.0, 1.5),
        n_steps: int = 10,
    ) -> float:
        """
        Бинарный поиск максимального η до потери когерентности (r < COHERENCE_LOSS_THRESHOLD).
        """
        sim = KuramotoSimulator(N=80)
        lo, hi = eta_range

        for _ in range(n_steps):
            mid = (lo + hi) / 2
            res = sim.run(omega_i=omega_i, K=K, eta=mid, tau=tau)
            if res["stable"]:
                lo = mid
            else:
                hi = mid

        return round((lo + hi) / 2, 3)


# ══════════════════════════════════════════════
# СЛОЙ 3 — Mode 1: Stress-Test
# ══════════════════════════════════════════════

class StressTester:
    """
    Mode 1: Внутренняя верификация 4D-артефакта.

    Алгоритм:
      1. Запустить базовую симуляцию
      2. Расшатать K, tau, eta на ±10%, ±30%, ±50%
      3. Вычислить λ_max
      4. Найти критические границы
      5. Вернуть stability_score [0,1]
    """

    def __init__(self):
        self.kuramoto = KuramotoSimulator(N=100)
        self.analyzer = StabilityAnalyzer()

    def run(self, four_d: Dict[str, Any], artifact_id: str = "") -> Dict[str, Any]:
        t_start = time.monotonic()

        dyn = four_d.get("dynamics", {})
        tim = four_d.get("time", {})
        inf = four_d.get("influence", {})

        omega_i = float(dyn.get("omega_i", 0.25))
        K = float(dyn.get("K", 0.35))
        K_c = float(dyn.get("K_c", 0.48))
        eta = float(inf.get("eta", 0.2))
        tau = float(tim.get("tau", 0.5))
        model_name = str(dyn.get("model", "kuramoto")).lower()

        # ── Базовая симуляция
        base_result = self._run_model(model_name, omega_i, K, eta, tau, four_d)

        # ── Стресс-тест по τ (×1.5, ×2.0)
        tau_results = []
        for tau_mult in [1.5, 2.0]:
            r = self._run_model(model_name, omega_i, K, eta, tau * tau_mult, four_d)
            tau_results.append({"tau_mult": tau_mult, "stable": r.get("stable", False)})

        # ── Стресс-тест по η (+0.15, +0.30)
        eta_results = []
        for eta_delta in [0.15, 0.30]:
            r = self._run_model(model_name, omega_i, K, min(eta + eta_delta, 1.5), tau, four_d)
            eta_results.append({"eta_delta": round(eta_delta, 2), "stable": r.get("stable", False)})

        # ── Стресс-тест по K (×0.7, ×0.85)
        K_results = []
        for K_mult in [0.70, 0.85]:
            r = self._run_model(model_name, omega_i, K * K_mult, eta, tau, four_d)
            K_results.append({"K_mult": K_mult, "stable": r.get("stable", False)})

        # ── λ_max
        lyapunov = self.analyzer.lyapunov_estimate(omega_i, K, eta)

        # ── Критическое η
        eta_critical = self.analyzer.find_critical_eta(omega_i, K, K_c, tau)

        # ── Stability score: доля прошедших стресс-тестов
        all_stress = tau_results + eta_results + K_results
        passed = sum(1 for s in all_stress if s.get("stable", False))
        stability_score = round(
            (0.3 * int(base_result.get("stable", False)) +
             0.7 * (passed / max(len(all_stress), 1))),
            3,
        )

        cpu_time = round(time.monotonic() - t_start, 2)

        result = {
            "artifact_id": artifact_id,
            "model_used": model_name,
            "timestamp": _now_iso(),
            "cpu_time_s": cpu_time,
            "base_simulation": base_result,
            "stress_tau": tau_results,
            "stress_eta": eta_results,
            "stress_K": K_results,
            "lyapunov_max": lyapunov,
            "lyapunov_stable": lyapunov < 0,
            "eta_critical": eta_critical,
            "bifurcation_boundary": {
                "K_above_critical": K > K_c,
                "K_c": K_c,
                "K": K,
                "eta_max": eta_critical,
                "tau_max_stable": round(
                    tau * 1.5 if tau_results[0]["stable"] else tau * 1.0,
                    3,
                ),
            },
            "stability_score": stability_score,
            "survival_verified": stability_score >= 0.6,
        }

        # Сохраняем в sim_results/
        self._save_result(artifact_id, result)
        return result

    def _run_model(
        self,
        model: str,
        omega_i: float,
        K: float,
        eta: float,
        tau: float,
        four_d: Dict[str, Any],
    ) -> Dict[str, Any]:
        if model == "percolation":
            p = float(four_d.get("dynamics", {}).get("p", 0.65))
            k_mean = float(four_d.get("structure", {}).get("k", 6.0))
            sim = PercolationSimulator(N=300)
            return sim.run(p=p, k_mean=k_mean)
        else:
            # По умолчанию Kuramoto (работает для большинства моделей)
            return self.kuramoto.run(omega_i=omega_i, K=K, eta=eta, tau=tau)

    def _save_result(self, artifact_id: str, result: Dict[str, Any]):
        SIM_RESULTS_DIR.mkdir(exist_ok=True)
        path = SIM_RESULTS_DIR / f"{artifact_id}_stress.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2))


# ══════════════════════════════════════════════
# СЛОЙ 4 — Mode 2: Resonance Matcher
# ══════════════════════════════════════════════

class ResonanceMatcher:
    """
    Mode 2: Поиск изоморфных артефактов по 4D-вектору и генерация инсайтов.
    """

    def __init__(self, four_d_index_path: str = "artifacts/four_d_index.jsonl"):
        self.index_path = Path(four_d_index_path)
        self._vectors: List[np.ndarray] = []
        self._meta: List[Dict] = []
        self._load_index()

    def _load_index(self):
        if not self.index_path.exists():
            return
        with open(self.index_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    vec = np.array(entry["vector"], dtype=np.float64)
                    self._vectors.append(vec)
                    self._meta.append(entry)
                except Exception:
                    continue
        logger.info(f"ResonanceMatcher: loaded {len(self._vectors)} 4D vectors")

    def add_to_index(
        self,
        artifact_id: str,
        four_d: Dict[str, Any],
        domain: str,
        vec: np.ndarray,
        stability_score: float = 0.5,
    ):
        """Добавляет вектор в индекс и сохраняет в файл."""
        entry = {
            "id": artifact_id,
            "domain": domain,
            "four_d": four_d,
            "vector": vec.tolist(),
            "stability_score": stability_score,
            "added_at": _now_iso(),
        }
        self._vectors.append(vec)
        self._meta.append(entry)
        self.index_path.parent.mkdir(exist_ok=True)
        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def find_similar(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.55,
    ) -> List[Dict[str, Any]]:
        """Возвращает top_k ближайших артефактов по 4D-расстоянию."""
        if not self._vectors:
            return []

        results = []
        for i, vec in enumerate(self._vectors):
            resonance = self._compute_resonance(query_vec, vec)
            if resonance >= threshold:
                results.append({
                    **self._meta[i],
                    "4d_resonance": resonance,
                })

        return sorted(results, key=lambda x: -x["4d_resonance"])[:top_k]

    def _compute_resonance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Взвешенный 4D-изоморфизм [0,1]."""
        if a.shape != b.shape:
            return 0.0
        # Косинус + 1 - евклид (среднее двух метрик)
        try:
            cos_sim = 1.0 - float(scipy_cosine(a, b))
        except Exception:
            cos_sim = 0.0
        norm = float(np.linalg.norm(a - b))
        max_norm = float(np.sqrt(a.shape[0]))
        euc_sim = 1.0 - norm / max(max_norm, 1e-9)
        return round((cos_sim * 0.6 + euc_sim * 0.4), 3)


# ══════════════════════════════════════════════
# СЛОЙ 5 — Probability Engine
# ══════════════════════════════════════════════

class ProbabilityEngine:
    """
    Вычисляет P(A→B) — вероятность кросс-доменного резонанса.

    Формула:
      Raw = α·Iso_4D + β·StabilityScore + γ·ScaleAlign + δ·SurvivalBonus − ε·NoisePenalty
      P = σ(k·(Raw − x0))
    """

    # Веса (подобраны на симуляциях из документации)
    ALPHA = 0.35   # 4D-изоморфизм
    BETA  = 0.25   # стабильность (StressTester)
    GAMMA = 0.20   # масштабное согласование
    DELTA = 0.15   # бонус за STRUCTURAL survival
    EPS   = 0.05   # штраф за шум

    # Логистическая калибровка
    K_LOGISTIC = 5.0
    X0_LOGISTIC = 0.60

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def compute(
        self,
        iso_4d: float,
        stability_score: float,
        scale_align: float,
        survival: str,
        noise_penalty: float,
    ) -> Dict[str, Any]:
        survival_bonus = 0.15 if survival == "STRUCTURAL" else 0.0
        raw = (
            self.ALPHA * iso_4d
            + self.BETA * stability_score
            + self.GAMMA * scale_align
            + self.DELTA * survival_bonus
            - self.EPS * noise_penalty
        )
        p = round(self.sigmoid(self.K_LOGISTIC * (raw - self.X0_LOGISTIC)), 3)

        # Калибровочный бэнд
        if p >= 0.75:
            band = "high"
        elif p >= 0.55:
            band = "plausible"
        elif p >= 0.35:
            band = "speculative"
        else:
            band = "low"

        return {
            "probability": p,
            "confidence_band": band,
            "raw_score": round(raw, 3),
            "components": {
                "iso_4d": iso_4d,
                "stability_score": stability_score,
                "scale_align": scale_align,
                "survival_bonus": survival_bonus,
                "noise_penalty": noise_penalty,
            },
        }

    @staticmethod
    def compute_scale_align(domain_a: str, domain_b: str) -> float:
        """
        Согласование масштабов между доменами.
        Упрощённая версия без внешнего config — на основе известных порядков масштабов.
        """
        # Референсный масштаб: характерный размер сети (узлы)
        SCALES: Dict[str, float] = {
            "physics": 1e6,
            "chemistry": 1e4,
            "biology": 1e3,
            "neuroscience": 1e4,
            "psychology": 1e2,
            "sociology": 1e5,
            "economics": 1e6,
            "linguistics": 1e4,
            "ecology": 1e3,
            "geology": 1e5,
            "medicine": 1e3,
            "astronomy": 1e9,
            "history": 1e4,
            "architecture": 1e2,
            "general": 1e3,
        }
        s_a = SCALES.get(domain_a.lower(), 1e3)
        s_b = SCALES.get(domain_b.lower(), 1e3)
        ratio = abs(math.log10(s_a) - math.log10(s_b))
        # Большая разница масштабов = хуже согласование; нормируем на 9 декад (макс. = astro vs psychology)
        return round(max(0.0, 1.0 - ratio / 9.0), 3)


# ══════════════════════════════════════════════
# ОРКЕСТРАТОР
# ══════════════════════════════════════════════

class MathCore:
    """
    Главный оркестратор MathCore v4.2.

    Вызывается из hxam_v_4_server.py после process_with_invariants().
    """

    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        four_d_index: str = "artifacts/four_d_index.jsonl",
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.stress_tester = StressTester()
        self.resonance_matcher = ResonanceMatcher(four_d_index)
        self.prob_engine = ProbabilityEngine()

    # ──────────────────────────────────────────
    # PUBLIC: Mode 1
    # ──────────────────────────────────────────

    def stress_test(
        self,
        artifact_id: str,
        four_d: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Запускает стресс-тест для артефакта.
        Если four_d не передан — читает из файла артефакта.
        Результат записывает в артефакт + sim_results/.
        """
        if four_d is None:
            four_d = self._load_four_d(artifact_id)

        if not four_d:
            return {"error": f"No four_d_matrix for {artifact_id}", "stability_score": 0.0}

        try:
            result = self.stress_tester.run(four_d, artifact_id)
            # Обновить артефакт
            self._patch_artifact(artifact_id, {"simulation": result})
            logger.info(
                f"MathCore stress_test {artifact_id}: "
                f"score={result['stability_score']} λ={result['lyapunov_max']}"
            )
            return result
        except Exception as e:
            logger.error(f"MathCore stress_test failed: {e}", exc_info=True)
            return {"error": str(e), "stability_score": 0.0}

    # ──────────────────────────────────────────
    # PUBLIC: Mode 2
    # ──────────────────────────────────────────

    def find_resonance(
        self,
        query_four_d: Dict[str, Any],
        query_domain: str,
        query_survival: str = "UNKNOWN",
        target_domains: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Ищет изоморфные артефакты и генерирует вероятностный инсайт.
        """
        from schemas.four_d_matrix import FourDMatrix, compute_4d_resonance

        matrix = FourDMatrix.from_raw(query_four_d)
        if matrix is None:
            return {"error": "Invalid four_d_matrix", "insights": []}

        query_vec = matrix.to_vector()
        similar = self.resonance_matcher.find_similar(query_vec, top_k=top_k)

        insights = []
        for match in similar:
            iso_4d = match.get("4d_resonance", 0.0)
            match_domain = match.get("domain", "general")
            stability = match.get("stability_score", 0.5)

            # Загрузить simulation.stability_score из sim_results если есть
            sim_path = SIM_RESULTS_DIR / f"{match['id']}_stress.json"
            if sim_path.exists():
                try:
                    sim_data = json.loads(sim_path.read_text())
                    stability = sim_data.get("stability_score", stability)
                except Exception:
                    pass

            scale_align = self.prob_engine.compute_scale_align(query_domain, match_domain)
            noise_penalty = float(query_four_d.get("influence", {}).get("eta", 0.2))

            prob_result = self.prob_engine.compute(
                iso_4d=iso_4d,
                stability_score=stability,
                scale_align=scale_align,
                survival=query_survival,
                noise_penalty=noise_penalty,
            )

            insight = {
                "id": f"ins_{match['id'][:8]}_{int(time.time())}",
                "source_domain": query_domain,
                "target_domain": match_domain,
                "prototype_id": match["id"],
                **prob_result,
                "iso_4d": iso_4d,
                "scale_align": scale_align,
                "status": "monitoring" if prob_result["probability"] >= 0.55 else "low_confidence",
                "generated_at": _now_iso(),
            }
            insights.append(insight)

        # Сохраняем инсайты
        for insight in insights:
            self._save_insight(insight)

        return {
            "insights": insights,
            "total": len(insights),
            "top_resonance": insights[0]["iso_4d"] if insights else 0.0,
        }

    # ──────────────────────────────────────────
    # PUBLIC: Индексирование нового артефакта
    # ──────────────────────────────────────────

    def index_artifact(
        self,
        artifact_id: str,
        four_d: Dict[str, Any],
        domain: str,
        stability_score: float = 0.5,
    ):
        """Добавляет 4D-вектор нового артефакта в индекс для последующего поиска."""
        from schemas.four_d_matrix import FourDMatrix

        matrix = FourDMatrix.from_raw(four_d)
        if matrix is None:
            return
        vec = matrix.to_vector()
        self.resonance_matcher.add_to_index(
            artifact_id=artifact_id,
            four_d=matrix.to_dict(),
            domain=domain,
            vec=vec,
            stability_score=stability_score,
        )
        logger.info(f"MathCore indexed {artifact_id} (domain={domain})")

    # ──────────────────────────────────────────
    # УТИЛИТЫ
    # ──────────────────────────────────────────

    def _load_four_d(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        path = self.artifacts_dir / f"{artifact_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            gen = data.get("data", {}).get("gen", {})
            return gen.get("four_d_matrix")
        except Exception:
            return None

    def _patch_artifact(self, artifact_id: str, patch: Dict[str, Any]):
        path = self.artifacts_dir / f"{artifact_id}.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.update(patch)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.error(f"_patch_artifact {artifact_id}: {e}")

    def _save_insight(self, insight: Dict[str, Any]):
        INSIGHTS_DIR.mkdir(exist_ok=True)
        path = INSIGHTS_DIR / f"{insight['id']}.json"
        path.write_text(json.dumps(insight, ensure_ascii=False, indent=2))

    # ──────────────────────────────────────────
    # CLI: самотестирование
    # ──────────────────────────────────────────

    @staticmethod
    def selftest():
        """Быстрый тест всех компонентов."""
        print("MathCore v4.2 self-test...")

        # Тест KuramotoSimulator
        sim = KuramotoSimulator(N=50)
        r = sim.run(omega_i=0.25, K=0.6, eta=0.15, tau=0.5)
        assert r["model"] == "kuramoto", "Kuramoto model name mismatch"
        print(f"  KuramotoSimulator: r_final={r['r_final']} stable={r['stable']} ✓")

        # Тест StabilityAnalyzer
        lya = StabilityAnalyzer.lyapunov_estimate(0.25, 0.6, 0.15, N=50, t_steps=100)
        print(f"  LyapunovEstimate: λ_max={lya} ✓")

        # Тест SchemaUtils
        from schemas.four_d_matrix import FourDMatrix, compute_4d_resonance
        fd = FourDMatrix.from_raw({
            "structure": {"C": 0.6, "k": 9, "D": 2.1},
            "influence": {"h": 0.9, "T": 1.0, "eta": 0.2},
            "dynamics": {"omega_i": 0.25, "K": 0.4, "K_c": 0.48, "p": 0.7, "model": "kuramoto"},
            "time": {"tau": 0.5, "H": 0.75, "freq": 1.1},
        })
        assert fd is not None, "FourDMatrix parse failed"
        vec = fd.to_vector()
        assert len(vec) == 13, f"Vector dim mismatch: {len(vec)}"
        r2 = compute_4d_resonance(vec, vec)
        assert r2 == 1.0, f"Self-resonance != 1.0: {r2}"
        print(f"  FourDMatrix: vec.shape={vec.shape} self_resonance={r2} ✓")

        # Тест ProbabilityEngine
        pe = ProbabilityEngine()
        p = pe.compute(iso_4d=0.89, stability_score=0.85, scale_align=0.7, survival="STRUCTURAL", noise_penalty=0.2)
        print(f"  ProbabilityEngine: P={p['probability']} band={p['confidence_band']} ✓")

        # Тест StressTester
        four_d_test = {
            "structure": {"C": 0.6, "k": 8.0, "D": 2.1},
            "influence": {"h": 0.8, "T": 1.0, "eta": 0.18},
            "dynamics": {"omega_i": 0.25, "K": 0.5, "K_c": 0.48, "p": 0.65, "model": "kuramoto"},
            "time": {"tau": 0.5, "H": 0.75, "freq": 1.0},
        }
        st = StressTester()
        res = st.run(four_d_test, "test_artifact")
        print(f"  StressTester: score={res['stability_score']} survival={res['survival_verified']} cpu={res['cpu_time_s']}s ✓")

        print("\n✅ All MathCore components OK")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="MathCore v4.2 CLI")
    parser.add_argument("--test",   action="store_true", help="Run self-test")
    parser.add_argument("--stress", type=str, default="",  help="artifact_id to stress-test")
    args = parser.parse_args()

    if args.test:
        MathCore.selftest()
    elif args.stress:
        core = MathCore()
        result = core.stress_test(args.stress)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        parser.print_help()
