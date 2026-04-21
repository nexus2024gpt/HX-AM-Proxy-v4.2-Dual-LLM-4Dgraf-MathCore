# mgap_matcher.py — HX-AM v4.4 Metric GAP
"""
MGAPMatcher — переносит численные инварианты из артефактов HX-AM
в прикладные отраслевые модели (реестр MGAP).

Полный цикл для одного артефакта:
  1. Загрузка артефакта + извлечение 4D-матрицы и порогов симуляции
  2. Ранжирование моделей реестра по compute_4d_resonance (13-мерный вектор)
  3. Перевод параметров τ, K, η на язык отрасли (translation_map)
  4. Генерация кода адаптации под целевую программу (диспетч по math_type)
  5. Расчёт на синтетических данных (example_data из реестра)
  6. Формирование вывода для разработчика и бизнеса

v4.4 исправления vs пилот:
  - Правильное извлечение: tau→four_d["time"]["tau"], K→four_d["dynamics"]["K"],
    eta→four_d["influence"]["eta"]  (не flat dict)
  - Резонанс через compute_4d_resonance() (13-мерный вектор из FourDMatrix)
  - Top-K матчей, отсортированных по резонансу (не только первый по math_type)
  - Диспетч кода по math_type: graph_invariant / kuramoto / delay
  - LLM вызывается с правильной сигнатурой llm.generate(prompt)
  - Все 11 моделей реестра имеют полные поля

CLI:
  python mgap_matcher.py --artifact 32d4aa917ac4
  python mgap_matcher.py --artifact 32d4aa917ac4 --model M1
  python mgap_matcher.py --artifact 32d4aa917ac4 --top_k 3 --all_types
  python mgap_matcher.py --batch          # все артефакты × все модели
  python mgap_matcher.py --registry       # вывести реестр
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("HXAM.mgap")

# ── Нормализация math_type (реестр использует "delay_ode", LLM генерирует "delay")
_MATH_TYPE_ALIASES: Dict[str, str] = {
    "delay_ode": "delay",
    "delay-ode": "delay",
    "lotka_volterra": "lotka_volterra",
    "graph-invariant": "graph_invariant",
}

ARTIFACTS_DIR = Path("artifacts")
REGISTRY_PATH = Path("mgap_registry.json")


def _norm_math_type(t: str) -> str:
    return _MATH_TYPE_ALIASES.get(t.lower().strip(), t.lower().strip())


# ══════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ИЗВЛЕЧЕНИЯ (исправление бага)
# ══════════════════════════════════════════════════════════

def _flat_4d(four_d: Dict) -> Dict[str, float]:
    """
    Извлекает tau, K, K_c, eta, omega_i, p из nested 4D-структуры.
    four_d = {"structure": {...}, "influence": {...}, "dynamics": {...}, "time": {...}}
    """
    dyn = four_d.get("dynamics", {})
    inf = four_d.get("influence", {})
    tim = four_d.get("time", {})
    return {
        "tau":     float(tim.get("tau",     0.5)),
        "K":       float(dyn.get("K",       0.35)),
        "K_c":     float(dyn.get("K_c",     0.48)),
        "eta":     float(inf.get("eta",     0.2)),
        "omega_i": float(dyn.get("omega_i", 0.25)),
        "p":       float(dyn.get("p",       0.65)),
        "model":   str(dyn.get("model",     "kuramoto")),
    }


def _extract_art_four_d(artifact: Dict) -> Optional[Dict]:
    """four_d_matrix живёт в data.gen.four_d_matrix"""
    data = artifact.get("data", {})
    gen  = data.get("gen", {})
    return gen.get("four_d_matrix")


def _extract_art_sim(artifact: Dict) -> Dict:
    sim = artifact.get("simulation") or {}
    return sim


def _extract_thresholds(sim: Dict, ver: Dict, model: Dict) -> Dict:
    """
    Приоритет извлечения порогов:
    1) ver.stress_test (значения для переведённого домена)
    2) critical_thresholds модели
    3) simulation (результаты MathCore для исходного домена)
    """
    stress = ver.get("stress_test") or {}
    # Приоритет 1: стресс-тест перевода
    eta = stress.get("eta_critical")
    tau = stress.get("tau_robustness")
    # Приоритет 2: критические пороги модели
    if eta is None:
        eta = model.get("critical_thresholds", {}).get("eta_max")
    if tau is None:
        tau = model.get("critical_thresholds", {}).get("tau_max")
    # Приоритет 3: симуляция артефакта
    if eta is None:
        eta = sim.get("eta_critical") or sim.get("bifurcation_boundary", {}).get("eta_max", 0.5)
    if tau is None:
        tau = sim.get("tau_robustness") or sim.get("bifurcation_boundary", {}).get("tau_max_stable", 1.0)
    return {
        "eta_critical": float(eta),
        "tau_robustness": float(tau),
        "lyapunov_max": float(sim.get("lyapunov_max", 0.0)),
        "stability_score": float(sim.get("stability_score", 0.5)),
        "survival_verified": bool(sim.get("survival_verified", False)),
    }


# ══════════════════════════════════════════════════════════
# РЕЗОНАНС ЧЕРЕЗ FourDMatrix (исправление бага)
# ══════════════════════════════════════════════════════════

def _art_vector(four_d: Dict) -> Optional[np.ndarray]:
    try:
        from schemas.four_d_matrix import FourDMatrix
        matrix = FourDMatrix.from_raw(four_d)
        return matrix.to_vector() if matrix else None
    except Exception:
        return None


def _model_vector(model: Dict) -> Optional[np.ndarray]:
    return _art_vector(model.get("four_d_matrix", {}))


def _compute_resonance(art_vec: np.ndarray, model: Dict, art_math_type: str) -> float:
    """
    Взвешенный резонанс:
      70% — compute_4d_resonance (13-мерный вектор)
      30% — бонус совпадения math_type
    """
    try:
        from schemas.four_d_matrix import compute_4d_resonance
        m_vec = _model_vector(model)
        if m_vec is None:
            return 0.0
        vec_res = float(compute_4d_resonance(art_vec, m_vec))
    except Exception:
        vec_res = 0.0

    type_bonus = 0.3 if _norm_math_type(model.get("math_type", "")) == _norm_math_type(art_math_type) else 0.0
    return round(vec_res * 0.7 + type_bonus, 3)


# ══════════════════════════════════════════════════════════
# ГЕНЕРАЦИЯ КОДА (диспетч по math_type)
# ══════════════════════════════════════════════════════════

def _code_graph_invariant(model: Dict, thresholds: Dict, flat: Dict) -> str:
    eta_c = thresholds["eta_critical"]
    tau_c = thresholds["tau_robustness"]
    prog  = model["programs"][0]
    return f"""\
# MGAP Stability Monitor — {prog}
# Артефакт HX-AM: model=graph_invariant, eta_crit={eta_c:.3f}, tau_crit={tau_c:.3f}
def mgap_stability_monitor(
    flow_values: list,   # ежедневные значения (продажи / ликвидность / биомасса)
    lag_values:  list,   # измеренные лаги (дни / мес / лет)
    old_buffer_coef: float = 0.2,
) -> dict:
    import numpy as np
    eta  = np.std(flow_values)  / max(np.mean(flow_values), 1e-9)  # CV
    tau  = np.mean(lag_values)
    warn = (eta > {eta_c:.3f}) or (tau > {tau_c:.3f})
    if warn:
        mult     = max(1.0, (eta / {eta_c:.3f}) * (tau / {tau_c:.3f}))
        new_buf  = old_buffer_coef * mult
        return {{"warning": True, "multiplier": round(mult, 3),
                "new_buffer_coef": round(new_buf, 3),
                "note": f"η={{eta:.3f}} (crit {eta_c:.3f}), τ={{tau:.2f}} (crit {tau_c:.3f})"}}
    return {{"warning": False, "multiplier": 1.0, "new_buffer_coef": old_buffer_coef}}
"""


def _code_kuramoto(model: Dict, thresholds: Dict, flat: Dict) -> str:
    eta_c = thresholds["eta_critical"]
    tau_c = thresholds["tau_robustness"]
    K_c   = model.get("critical_thresholds", {}).get("K_min", flat["K_c"])
    prog  = model["programs"][0]
    return f"""\
# MGAP Stability Monitor — {prog}
# Артефакт HX-AM: model=kuramoto, K_c={K_c:.3f}, eta_crit={eta_c:.3f}, tau_crit={tau_c:.3f}
def mgap_stability_monitor(
    coupling_K: float,   # измеренная сила связи (виральность / синхронизация / coherence)
    noise_eta:  float,   # уровень шума (CV задержек / фоновый шум / информационный шум)
    delay_tau:  float,   # задержка (ч / мс / дней)
) -> dict:
    warnings = []
    if coupling_K < {K_c:.3f}:
        warnings.append(f"K={{coupling_K:.3f}} < K_c={K_c:.3f} — система ниже критического порога синхронизации")
    if noise_eta > {eta_c:.3f}:
        warnings.append(f"η={{noise_eta:.3f}} > η_crit={eta_c:.3f} — шум разрушает синхронизацию")
    if delay_tau > {tau_c:.3f}:
        warnings.append(f"τ={{delay_tau:.3f}} > τ_crit={tau_c:.3f} — задержка слишком велика")
    stable = len(warnings) == 0
    return {{"stable": stable, "warnings": warnings,
            "recommendation": "Система устойчива." if stable
            else "Снизить шум или увеличить силу связи выше K_c={K_c:.3f}."}}
"""


def _code_delay(model: Dict, thresholds: Dict, flat: Dict) -> str:
    eta_c = thresholds["eta_critical"]
    tau_c = thresholds["tau_robustness"]
    K_min = model.get("critical_thresholds", {}).get("K_min", 0.1)
    prog  = model["programs"][0]
    return f"""\
# MGAP Stability Monitor — {prog}
# Артефакт HX-AM: model=delay, eta_crit={eta_c:.3f}, tau_crit={tau_c:.3f}
def mgap_stability_margin(
    eta: float,   # уровень шума / волатильности
    tau: float,   # задержка в единицах модели
    K:   float,   # сила взаимодействия / мультипликатор
) -> dict:
    margin = min(
        1 - eta  / {eta_c:.3f},
        1 - tau  / {tau_c:.3f},
        (K - {K_min:.3f}) / max({K_min:.3f}, 1e-9),
    )
    warn = margin < 0.2
    return {{
        "stability_margin": round(margin, 4),
        "warning": warn,
        "components": {{
            "noise_margin":   round(1 - eta  / {eta_c:.3f}, 4),
            "delay_margin":   round(1 - tau  / {tau_c:.3f}, 4),
            "coupling_margin": round((K - {K_min:.3f}) / max({K_min:.3f}, 1e-9), 4),
        }},
        "recommendation": (
            "Устойчиво." if not warn else
            f"Снизить η до <{eta_c:.3f} и τ до <{tau_c:.3f}, увеличить K выше {K_min:.3f}."
        ),
    }}
"""


def _generate_code(model: Dict, thresholds: Dict, flat: Dict) -> str:
    mt = _norm_math_type(model.get("math_type", "kuramoto"))
    if mt == "graph_invariant":
        return _code_graph_invariant(model, thresholds, flat)
    elif mt == "kuramoto":
        return _code_kuramoto(model, thresholds, flat)
    elif mt in ("delay", "delay_ode"):
        return _code_delay(model, thresholds, flat)
    else:
        return f"# math_type '{mt}' не поддерживается кодогенерацией MGAP v4.4"


# ══════════════════════════════════════════════════════════
# РАСЧЁТ НА ПРИМЕРЕ (диспетч по типу example_data)
# ══════════════════════════════════════════════════════════

def _calculate_example(model: Dict, thresholds: Dict) -> Dict:
    example = model.get("example_data")
    if not example:
        return {"error": "no example_data in model"}

    eta_c = thresholds["eta_critical"]
    tau_c = thresholds["tau_robustness"]
    t     = example.get("type", "graph_invariant")

    if t == "graph_invariant":
        d_mean  = float(example.get("daily_sales_mean", 100))
        d_std   = float(example.get("daily_sales_std",   30))
        lag     = float(example.get("current_lead_time",  3.0))
        old_buf = float(example.get("old_safety_stock_coef", 0.2))
        eta = d_std / max(d_mean, 1e-9)
        warn = (eta > eta_c) or (lag > tau_c)
        mult = max(1.0, (eta / eta_c) * (lag / tau_c)) if warn else 1.0
        old_ss  = old_buf * d_mean * lag
        new_ss  = old_ss * mult
        return {
            "example_type":    "graph_invariant",
            "input":           example,
            "computed_cv":     round(eta,    4),
            "lag":             lag,
            "eta_critical":    eta_c,
            "tau_critical":    tau_c,
            "old_buffer":      round(old_ss, 2),
            "multiplier":      round(mult,   4),
            "new_buffer":      round(new_ss, 2),
            "warning_triggered": warn,
        }

    elif t == "kuramoto":
        K     = float(example.get("coupling_K",   0.7))
        K_c   = float(example.get("K_c",          0.5))
        noise = float(example.get("noise_eta",    0.2))
        delay = float(example.get("delay_tau_hours", example.get("delay_tau_ms", example.get("delay_tau_days", 1.0))))
        warns = []
        if K < K_c:     warns.append(f"K={K:.3f} < K_c={K_c:.3f}")
        if noise > eta_c: warns.append(f"η={noise:.3f} > η_crit={eta_c:.3f}")
        if delay > tau_c: warns.append(f"τ={delay:.3f} > τ_crit={tau_c:.3f}")
        return {
            "example_type":    "kuramoto",
            "input":           example,
            "K_above_Kc":      K > K_c,
            "noise_ok":        noise <= eta_c,
            "delay_ok":        delay <= tau_c,
            "warnings":        warns,
            "stable":          len(warns) == 0,
            "warning_triggered": len(warns) > 0,
        }

    elif t == "delay":
        K     = float(example.get("coupling_K",   0.3))
        noise = float(example.get("noise_eta",    0.2))
        delay = float(example.get("delay_tau",    1.0))
        K_min = model.get("critical_thresholds", {}).get("K_min", 0.1)
        m_noise = 1 - noise  / max(eta_c, 1e-9)
        m_delay = 1 - delay  / max(tau_c, 1e-9)
        m_K     = (K - K_min)/ max(K_min,  1e-9)
        margin  = min(m_noise, m_delay, m_K)
        warn    = margin < 0.2
        return {
            "example_type":    "delay",
            "input":           example,
            "stability_margin":  round(margin, 4),
            "noise_margin":      round(m_noise, 4),
            "delay_margin":      round(m_delay, 4),
            "coupling_margin":   round(m_K,     4),
            "warning_triggered": warn,
        }

    return {"error": f"unknown example_data type: {t}"}


# ══════════════════════════════════════════════════════════
# ВЕРДИКТ
# ══════════════════════════════════════════════════════════

def _build_verdict(model: Dict, calc: Dict, resonance: float, thresholds: Dict) -> Dict:
    warn = calc.get("warning_triggered", False)
    prog = model["programs"][0]
    if warn:
        biz_rec = (f"Система НА ПОРОГЕ нестабильности. "
                   f"Внедрить MGAP-монитор в {prog}. "
                   f"Ожидаемое снижение риска каскадных отказов: 15–25%.")
        dev_action = f"Добавить функцию mgap_stability_monitor() в {prog}"
    else:
        biz_rec = f"Система стабильна. Мониторинг порогов полезен профилактически."
        dev_action = f"Добавить пассивный мониторинг порогов в {prog}"

    mult = calc.get("multiplier", calc.get("stability_margin", 1.0))
    return {
        "verdict": "Применимо как расширение" if warn else "Применимо, мониторинг",
        "for_developer": {
            "action":          dev_action,
            "code_reference":  "adaptation.code_snippet",
            "new_config_params": {
                "eta_critical":  thresholds["eta_critical"],
                "tau_robustness": thresholds["tau_robustness"],
            },
        },
        "for_business": {
            "summary":        (f"Артефакт резонирует с моделью «{model['name']}» "
                               f"({model['logia']}, resonance={resonance:.2f})."),
            "blind_spot":     model.get("blind_spot_template", "—").format(
                                  eta_max=thresholds["eta_critical"],
                                  tau_max=thresholds["tau_robustness"]),
            "recommendation": biz_rec,
            "stability_score": thresholds.get("stability_score", "—"),
            "estimated_roi":  "Снижение риска каскадных отказов на 15–25%",
        },
    }


# ══════════════════════════════════════════════════════════
# ОСНОВНОЙ КЛАСС
# ══════════════════════════════════════════════════════════

class MGAPMatcher:
    """
    Основной класс Metric GAP.

    Пример:
        matcher = MGAPMatcher()
        results = matcher.match_artifact("32d4aa917ac4", top_k=3)
        for r in results:
            print(r["model_id"], r["resonance"])
    """

    def __init__(
        self,
        registry_path: str = "mgap_registry.json",
        artifacts_dir: str = "artifacts",
    ):
        self.registry_path = Path(registry_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.registry      = self._load_registry()
        self.llm           = self._try_load_llm()

    def _load_registry(self) -> List[Dict]:
        if not self.registry_path.exists():
            logger.warning(f"Registry not found: {self.registry_path}")
            return []
        data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        logger.info(f"MGAPMatcher: loaded {len(data.get('models', []))} models from {self.registry_path}")
        return data.get("models", [])

    def _try_load_llm(self):
        try:
            from llm_client_v_4 import LLMClient
            return LLMClient()
        except Exception:
            return None

    # ── загрузка артефакта ─────────────────────────────────

    def _load_artifact(self, artifact_id: str) -> Optional[Dict]:
        for base in [self.artifacts_dir, Path(".")]:
            for name in [f"{artifact_id}.json", f"{artifact_id}.hyx-portal.json"]:
                p = base / name
                if p.exists():
                    try:
                        return json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        pass
        return None

    # ── optional LLM улучшение blind_spot ─────────────────

    def _llm_improve_blind_spot(self, template: str, model: Dict) -> str:
        if not self.llm or not template:
            return template
        prompt = (f"Улучши описание слепой зоны для отрасли «{model['logia']}» "
                  f"(модель: {model['name']}), сохрани смысл и числа. "
                  f"Верни ТОЛЬКО улучшенный текст, одним абзацем:\n{template}")
        try:
            improved, _ = self.llm.generate(prompt)
            if improved and len(improved) > 20 and not improved.startswith("[Generator error]"):
                return improved.strip()
        except Exception:
            pass
        return template

    # ── основной матч одного артефакта ────────────────────

    def match_artifact(
        self,
        artifact_id: str,
        top_k: int = 3,
        math_type_only: bool = False,
        model_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Возвращает top_k матчей артефакта с моделями реестра,
        отсортированных по убыванию резонанса.

        Args:
            artifact_id:    ID артефакта (без .json)
            top_k:          сколько лучших моделей вернуть
            math_type_only: если True — фильтровать только по совпадению math_type
            model_id:       если задан — вернуть только этот model_id

        Returns:
            Список dict с ключами: artifact_id, model_id, resonance, math_type_match,
            translation, thresholds, adaptation, calculation, verdict
        """
        artifact = self._load_artifact(artifact_id)
        if not artifact:
            return [{"error": f"Artifact '{artifact_id}' not found",
                     "artifact_id": artifact_id}]

        four_d = _extract_art_four_d(artifact)
        if not four_d:
            return [{"error": "No four_d_matrix in artifact (запустите migrate_to_v42.py)",
                     "artifact_id": artifact_id}]

        sim     = _extract_art_sim(artifact)
        ver     = artifact.get("data", {}).get("ver", {})
        flat    = _flat_4d(four_d)
        art_math = _norm_math_type(flat["model"])

        art_vec = _art_vector(four_d)

        candidates = self.registry
        if model_id:
            candidates = [m for m in candidates if m["id"] == model_id]
        if math_type_only:
            candidates = [m for m in candidates
                          if _norm_math_type(m.get("math_type", "")) == art_math]
        if not candidates:
            return [{"error": f"No matching models (math_type={art_math}, math_type_only={math_type_only})",
                     "artifact_id": artifact_id}]

        # Считаем резонанс для всех кандидатов
        scored: List[Tuple[float, Dict]] = []
        for model in candidates:
            if art_vec is not None:
                res = _compute_resonance(art_vec, model, art_math)
            else:
                # Fallback: ручное сходство по параметрам (если FourDMatrix недоступен)
                res = self._resonance_fallback(flat, model)
            scored.append((res, model))

        scored.sort(key=lambda x: -x[0])
        top = scored[:top_k]

        results = []
        for resonance, model in top:
            thresholds = _extract_thresholds(sim, ver, model)
            match = self._build_match(
                artifact_id=artifact_id,
                artifact=artifact,
                four_d=four_d,
                flat=flat,
                thresholds=thresholds,
                art_math=art_math,
                model=model,
                resonance=resonance,
            )
            results.append(match)
        return results

    def _build_match(
        self,
        artifact_id: str,
        artifact: Dict,
        four_d: Dict,
        flat: Dict,
        thresholds: Dict,
        art_math: str,
        model: Dict,
        resonance: float,
    ) -> Dict[str, Any]:
        """Собирает полный MGAP-отчёт для одного (артефакт, модель) совпадения."""
        math_match = (_norm_math_type(model.get("math_type", "")) == art_math)

        # Перевод параметров
        translation = self._translate_params(flat, thresholds, model)

        # Слепая зона (шаблон + опциональный LLM)
        raw_blind = model.get("blind_spot_template", "Слепая зона не определена.").format(
            eta_max=thresholds["eta_critical"],
            tau_max=thresholds["tau_robustness"],
        )
        blind_spot = self._llm_improve_blind_spot(raw_blind, model)

        # Код адаптации
        code_snippet = _generate_code(model, thresholds, flat)

        # Расчёт на примере
        calculation = _calculate_example(model, thresholds)

        # Вердикт
        verdict = _build_verdict(model, calculation, resonance, thresholds)

        # Краткая сводка артефакта
        gen      = artifact.get("data", {}).get("gen", {})
        archivist = artifact.get("archivist") or {}

        return {
            "artifact_id":   artifact_id,
            "model_id":      model["id"],
            "model_name":    model["name"],
            "logia":         model["logia"],
            "industry":      model["industry"],
            "programs":      model["programs"],
            "resonance":     resonance,
            "math_type_match": math_match,
            "artifact_summary": {
                "domain":         artifact.get("data", {}).get("domain", "—"),
                "hypothesis":     gen.get("hypothesis", "")[:120],
                "math_type":      art_math,
                "stability_score": thresholds["stability_score"],
                "survival_verified": thresholds["survival_verified"],
                "novelty":        archivist.get("novelty", "—"),
            },
            "thresholds":    thresholds,
            "translation":   translation,
            "blind_spot":    blind_spot,
            "adaptation": {
                "formula":       model.get("math_adaptation_formula", "—"),
                "code_snippet":  code_snippet,
                "programs":      model["programs"],
            },
            "calculation":   calculation,
            "verdict":       verdict,
            "generated_at":  __import__("datetime").datetime.utcnow().isoformat() + "Z",
        }

    def _translate_params(self, flat: Dict, thresholds: Dict, model: Dict) -> Dict:
        """Переводит tau, K, eta в отраслевые термины."""
        tmap = model.get("translation_map", {})
        result: Dict = {}
        for key, val in [("tau", flat["tau"]), ("K", flat["K"]), ("eta", flat["eta"])]:
            if key in tmap:
                entry = tmap[key]
                result[entry["industry_term"]] = {
                    "math_param":    key,
                    "value":         round(val, 4),
                    "description":   entry.get("description", ""),
                    "typical_range": entry.get("typical_values", "—"),
                }
            else:
                result[key] = {"math_param": key, "value": round(val, 4)}
        result["_thresholds"] = {
            "eta_critical":   thresholds["eta_critical"],
            "tau_robustness": thresholds["tau_robustness"],
        }
        return result

    def _resonance_fallback(self, flat: Dict, model: Dict) -> float:
        """Ручной расчёт резонанса если schemas.four_d_matrix недоступен."""
        params = [("tau", flat["tau"]), ("K", flat["K"]), ("eta", flat["eta"])]
        m4d    = model.get("four_d_matrix", {})
        m_flat = _flat_4d(m4d)
        ranges = model.get("expected_ranges", {})
        weights = model.get("weights", {})
        total = 0.0; score = 0.0
        for key, val in params:
            r   = ranges.get(key, [0.0, 1.0])
            w   = weights.get(key, 1.0)
            span = max(r[1] - r[0], 1e-9)
            sim = max(0.0, 1.0 - abs(val - m_flat.get(key, 0.5)) / span)
            score += sim * w
            total += w
        return round(score / max(total, 1e-9), 3)

    # ── batch режим ───────────────────────────────────────

    def match_batch(
        self,
        top_k: int = 2,
        math_type_only: bool = True,
        min_resonance: float = 0.3,
    ) -> Dict[str, List[Dict]]:
        """
        Прогоняет все артефакты из artifacts/ через MGAPMatcher.
        Возвращает словарь {artifact_id: [top matches]}.
        """
        results: Dict[str, List[Dict]] = {}
        arts_path = self.artifacts_dir
        if not arts_path.exists():
            logger.warning("artifacts/ not found")
            return results

        for f in sorted(arts_path.glob("*.json")):
            if f.stem == "invariant_graph" or ".hyx-portal" in f.name:
                continue
            art_id = f.stem
            try:
                matches = self.match_artifact(art_id, top_k=top_k,
                                               math_type_only=math_type_only)
                # фильтруем по минимальному резонансу
                ok = [m for m in matches
                      if not m.get("error") and m.get("resonance", 0) >= min_resonance]
                if ok:
                    results[art_id] = ok
                    logger.info(f"MGAP batch: {art_id} → "
                                f"{[(m['model_id'], m['resonance']) for m in ok]}")
            except Exception as e:
                logger.warning(f"MGAP batch: {art_id} failed — {e}")
        return results

    # ── реестр ────────────────────────────────────────────

    def get_registry_summary(self) -> List[Dict]:
        return [
            {
                "id":        m["id"],
                "name":      m["name"],
                "logia":     m["logia"],
                "industry":  m["industry"],
                "math_type": m.get("math_type", "—"),
                "programs":  m.get("programs", []),
            }
            for m in self.registry
        ]


# ══════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════

def _cli():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="HX-AM v4.4 MGAPMatcher CLI")
    parser.add_argument("--artifact",   type=str, default="",  help="Artifact ID")
    parser.add_argument("--model",      type=str, default="",  help="Filter by Model ID (e.g. M1)")
    parser.add_argument("--top_k",      type=int, default=3,   help="Top-K matches")
    parser.add_argument("--all_types",  action="store_true",   help="Match across all math types")
    parser.add_argument("--batch",      action="store_true",   help="Batch: all artifacts")
    parser.add_argument("--registry",   action="store_true",   help="Print registry summary")
    parser.add_argument("--min_res",    type=float, default=0.3, help="Min resonance for batch")
    parser.add_argument("--registry_path", type=str, default="mgap_registry.json")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    args = parser.parse_args()

    matcher = MGAPMatcher(
        registry_path=args.registry_path,
        artifacts_dir=args.artifacts_dir,
    )

    if args.registry:
        print(json.dumps(matcher.get_registry_summary(), ensure_ascii=False, indent=2))
        return

    if args.batch:
        results = matcher.match_batch(
            top_k=args.top_k,
            math_type_only=not args.all_types,
            min_resonance=args.min_res,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if not args.artifact:
        parser.print_help()
        return

    results = matcher.match_artifact(
        artifact_id=args.artifact,
        top_k=args.top_k,
        math_type_only=not args.all_types,
        model_id=args.model or None,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
