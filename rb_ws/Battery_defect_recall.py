"""결함 검출 Recall 계산.

표면 커버리지(Battery_coverage_metrics)가 '물체 표면을 얼마나 봤는가'를
재는 반면, 여기서는 '심어둔 결함 중 몇 개를 실제로 봤는가'를 잰다.

    Recall = 관측된 결함 수 / 전체 결함 수

분모는 setup_defects()가 내보낸 GT(defect_gt_<scenario>.json)의 결함
목록이고, 분자는 캡처 때마다 쌓인 관측 기록(defect_observations.json)이다.

판정 기준이 왜 '픽셀 수 임계값'인가:

    결함은 semantic segmentation에서 자기만의 ID를 갖는다. 그래서 어떤
    뷰에서 그 ID의 픽셀이 하나라도 있으면 원리상 '보인' 것이다. 하지만
    1~2픽셀은 스치듯 지나간 것이라 실제 검출기가 잡을 수 있는 신호가
    아니다. 흠집 폭이 0.5~0.8mm이고 GSD가 약 0.189mm/px이므로 정면에서는
    폭만 3~4픽셀이 나오지만, 비스듬히 보면 몇 픽셀로 줄어든다.

    그래서 '최소 몇 픽셀 이상 보여야 검출로 친다'를 임계값으로 두고,
    report()는 여러 임계값에서의 Recall을 함께 출력한다. 임계값에 따라
    결론이 흔들리는지 눈으로 보고 정하라는 뜻이다.

판정은 '시퀀스 전체 프레임의 합집합'이다:

    NBV 10프레임 중 '한 번이라도' 충분히 보인 결함은 검출로 친다. 어느
    한 프레임이 전부를 볼 필요는 없다. 결함 A가 frame 1에서만, 결함 B가
    frame 5에서만 보였어도 둘 다 검출로 센다. 애초에 시점을 바꿔가며
    보는 이유가 그것이다.

한 결함의 픽셀 수를 프레임끼리 '더하지는' 않는다 (criterion="max"):

    합집합은 결함 단위로 잡고, 픽셀 수는 프레임별 최댓값을 쓴다. 같은
    흠집을 여러 뷰에서 보면 같은 면적이 매번 다시 세어지므로, 더하면
    '많이 봤다'가 아니라 '여러 번 봤다'를 재게 된다. 뷰가 늘어날수록
    값이 커져서 임계값의 의미가 사라진다.

    다만 3D로 융합하는 검출기라면 여러 뷰의 증거를 합치는 것이 맞을 수도
    있어서 criterion="sum"도 준비해 두었다. 이때는 뷰 수에 값이 비례해
    커지므로 임계값을 그만큼 높여야 한다.

CLI:

    python Battery_defect_recall.py
    python Battery_defect_recall.py --min-pixels 20 --defect-type scratch
    python Battery_defect_recall.py --criterion sum --min-pixels 50
"""

from __future__ import annotations

import argparse
import json
import os

try:
    from rb_ws.Battery_defect_config import (
        DEFECT_GT_EXPORT_DIR,
        DEFECT_REFERENCE_DISTANCE_M,
        DEFECT_REFERENCE_FX_PX,
        DEFECT_REFERENCE_GSD_MM_PER_PX,
        DEFECT_SCENARIO_NAME,
    )
except ModuleNotFoundError:
    try:
        from Battery_defect_config import (
            DEFECT_GT_EXPORT_DIR,
            DEFECT_REFERENCE_DISTANCE_M,
            DEFECT_REFERENCE_FX_PX,
            DEFECT_REFERENCE_GSD_MM_PER_PX,
            DEFECT_SCENARIO_NAME,
        )
    except ModuleNotFoundError:
        DEFECT_GT_EXPORT_DIR = None
        DEFECT_REFERENCE_DISTANCE_M = 0.400
        DEFECT_REFERENCE_FX_PX = 2120.0
        DEFECT_REFERENCE_GSD_MM_PER_PX = 400.0 / 2120.0
        DEFECT_SCENARIO_NAME = "nbv"


def gsd_for_distance(distance_m, fx_px=None):
    """mm/px of a frontal view at ``distance_m`` (recall denominator basis)."""
    fx_px = DEFECT_REFERENCE_FX_PX if fx_px is None else float(fx_px)
    return 1000.0 * float(distance_m) / fx_px


def comparison_distances_from_env():
    """Extra nominal distances [m] to compare, e.g. "0.41,0.45"."""
    return [
        float(value)
        for value in os.environ.get(
            "BATTERY_RECALL_COMPARE_DISTANCES_M", ""
        ).split(",")
        if value.strip()
    ]


# 검출로 인정할 최소 픽셀 수.
#
# 1~2픽셀은 결함 가장자리를 스친 것이라 신호로 보기 어렵다. 폭
# 0.5~0.8mm 흠집이 GSD 0.189mm/px에서 정면이면 폭 3~4px에 길이 48~79px
# 이므로 100px 이상 나온다. 10px은 '면적의 10분의 1도 못 봤다'에
# 해당하므로 느슨한 하한이다.
DEFAULT_MIN_PIXELS = 10

# report()가 함께 보여줄 임계값들. 결론이 임계값에 얼마나 민감한지
# 드러내려는 것이다.
REPORT_THRESHOLDS = (1, 5, 10, 25, 50, 100)
REPORT_RATIO_THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)

# 면적 비율로 판정할 때 쓰는 기본 임계값.
#
# 주의: 이 비율의 분모는 '정면에서 공칭 거리로 봤을 때의 픽셀 수'다.
# 그래서 가려지지 않았더라도 비스듬히 보는 것만으로 비율이 깎인다.
# 투영 면적이 cos(입사각)에 비례하므로:
#
#     입사각  0도 -> 100%      45도 -> 71%
#            30도 ->  87%      46도 -> 69%
#            40도 ->  77%      60도 -> 50%
#
# 즉 임계값은 사실상 '입사각 몇 도 이내에서 본 적이 있어야 하는가'다:
#
#     0.70 -> 45도 이내
#     0.50 -> 60도 이내   <- 현재값
#
# 0.70에서 0.50으로 낮춘 이유: 안착 자세에서 정면 결함의 입사각이
# 실측 46도였다. 0.70이면 그 뷰가 69%로 탈락해, 결함이 화면 한가운데
# 멀쩡히 보이는데도 미검출로 집계된다. 판정선이 실제 관측 형상 바로
# 위에 걸려 있어서 지표가 불안정해진다.
#
# 0.50은 60도까지 허용하므로 46도 뷰가 여유 있게 통과하고, edge-on에
# 가까운 96도(theta 90)나 뒷면 136~147도는 여전히 걸러낸다.
DEFAULT_MIN_RATIO = 0.50

DEFAULT_OBSERVATION_FILENAME = "defect_observations.json"


def default_gt_path(scenario_name=None, gt_dir=None):
    """GT JSON의 기본 경로."""
    if gt_dir is None:
        gt_dir = DEFECT_GT_EXPORT_DIR

    if gt_dir is None:
        raise ValueError(
            "GT 디렉터리를 알 수 없습니다. --gt-path로 직접 지정하세요."
        )

    if scenario_name is None:
        scenario_name = DEFECT_SCENARIO_NAME

    return os.path.join(gt_dir, f"defect_gt_{scenario_name}.json")


def load_defect_gt(gt_path, gsd_mm_per_px=DEFECT_REFERENCE_GSD_MM_PER_PX):
    """GT를 읽어 결함별 메타데이터로 편다.

    면적 비율 판정을 위해 footprint_area_mm2로부터 '정면에서 봤을 때의
    픽셀 수'(ideal_pixels)를 함께 계산한다.
    """
    with open(gt_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    catalog = {}

    for record in payload.get("defects", []):
        name = record.get("name")

        if not name:
            continue

        footprint_mm2 = (record.get("size") or {}).get(
            "footprint_area_mm2"
        )
        # 결함이 정면에서 공칭 거리로 보였을 때의 픽셀 수. 면적 비율의
        # 분모다. alpha cutout 실제 footprint 면적이라 decal 패치 크기가
        # 아니라 흠집 자체의 면적이다.
        ideal_pixels = (
            float(footprint_mm2) / (gsd_mm_per_px ** 2)
            if footprint_mm2
            else None
        )
        catalog[name] = {
            "defect_type": record.get("defect_type", "unknown"),
            "difficulty": record.get("difficulty", "unknown"),
            "visible_at_rest": bool(record.get("visible_at_rest", False)),
            "footprint_area_mm2": (
                float(footprint_mm2) if footprint_mm2 else None
            ),
            "ideal_pixels": ideal_pixels,
        }

    if not catalog:
        raise ValueError(f"GT에 결함이 없습니다: {gt_path}")

    return catalog


def load_defect_observations(observation_path):
    """관측 기록을 읽어 {name: {frame_id: pixels}}로 편다."""
    if not os.path.isfile(observation_path):
        return {}, 0

    with open(observation_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    frames = payload.get("frames", [])
    observed = {}

    for frame in frames:
        frame_id = int(frame.get("frame_id", -1))

        for name, entry in (frame.get("defects") or {}).items():
            observed.setdefault(name, {})[frame_id] = int(
                entry.get("pixels", 0)
            )

    return observed, len(frames)


def best_pixels_per_defect(catalog, observations, criterion="max"):
    """결함마다 시퀀스 전체에서의 대표 픽셀 수를 낸다.

    criterion="max":
        프레임별 픽셀 수의 최댓값. 기본값이다.
    criterion="sum":
        모든 프레임의 픽셀 수 합. 같은 면적이 여러 번 세어지므로
        뷰 수에 비례해 커진다.

    어느 쪽이든 '한 번이라도 보인 결함'을 집계 대상에 넣는다는 점은
    같다. 프레임 간 합집합은 여기서 결정되는 것이 아니라, 관측 기록에
    이름이 한 번이라도 등장했는지로 결정된다.

    GT에는 있는데 한 번도 안 잡힌 결함은 0으로 채운다. 그래야 분모가
    전체 결함 수로 유지된다.
    """
    if criterion not in ("max", "sum"):
        raise ValueError(
            f'criterion은 "max" 또는 "sum"이어야 합니다: {criterion}'
        )

    best = {}

    for name in catalog:
        per_frame = observations.get(name, {})

        if not per_frame:
            best[name] = {
                "pixels": 0,
                "frame_id": None,
                "frame_count": 0,
            }

            continue

        peak_frame_id, peak_pixels = max(
            per_frame.items(),
            key=lambda item: item[1],
        )
        pixels = (
            peak_pixels
            if criterion == "max"
            else sum(per_frame.values())
        )
        best[name] = {
            "pixels": int(pixels),
            "frame_id": peak_frame_id,
            "frame_count": len(per_frame),
        }

    return best


def defect_recall(
    catalog,
    observations,
    min_pixels=DEFAULT_MIN_PIXELS,
    defect_type=None,
    criterion="max",
    min_ratio=None,
):
    """Recall과 그 근거를 함께 돌려준다.

    min_ratio를 주면 절대 픽셀 수 대신 '면적 비율'로 판정한다.
    비율 = 관측 픽셀 / 정면·공칭거리에서의 픽셀(ideal_pixels).
    GT에 footprint 면적이 없는 결함은 비율을 낼 수 없으므로 그때만
    min_pixels로 되돌아간다.

    시퀀스 전체 프레임의 합집합으로 센다. 한 프레임에서만 보인 결함도
    검출로 들어간다.

    defect_type을 주면 그 타입만 센다 (예: "scratch").
    criterion은 best_pixels_per_defect()와 같다.
    """
    best = best_pixels_per_defect(
        catalog,
        observations,
        criterion=criterion,
    )
    selected = [
        name
        for name, info in catalog.items()
        if defect_type is None or info["defect_type"] == defect_type
    ]

    if not selected:
        return {
            "recall": 0.0,
            "detected": 0,
            "total": 0,
            "min_pixels": int(min_pixels),
            "min_ratio": (
                float(min_ratio) if min_ratio is not None else None
            ),
            "defect_type": defect_type,
            "criterion": criterion,
            "per_defect": {},
        }

    per_defect = {}

    for name in sorted(selected):
        pixels = best[name]["pixels"]
        ideal_pixels = catalog[name].get("ideal_pixels")
        ratio = (
            pixels / ideal_pixels
            if ideal_pixels
            else None
        )

        if min_ratio is not None and ratio is not None:
            detected = ratio >= float(min_ratio)
        else:
            detected = pixels >= int(min_pixels)

        per_defect[name] = {
            "defect_type": catalog[name]["defect_type"],
            "difficulty": catalog[name]["difficulty"],
            "visible_at_rest": catalog[name]["visible_at_rest"],
            "best_pixels": pixels,
            "ideal_pixels": (
                int(round(ideal_pixels)) if ideal_pixels else None
            ),
            "area_ratio": ratio,
            "best_frame_id": best[name]["frame_id"],
            "seen_frame_count": best[name]["frame_count"],
            "detected": detected,
        }

    detected = sum(
        1 for info in per_defect.values() if info["detected"]
    )

    return {
        "recall": detected / len(selected),
        "detected": detected,
        "total": len(selected),
        "min_pixels": int(min_pixels),
        "min_ratio": (
            float(min_ratio) if min_ratio is not None else None
        ),
        "defect_type": defect_type,
        "criterion": criterion,
        "per_defect": per_defect,
    }


def report(
    gt_path=None,
    observation_path=None,
    min_pixels=DEFAULT_MIN_PIXELS,
    defect_type=None,
    scenario_name=None,
    criterion="max",
    min_ratio=DEFAULT_MIN_RATIO,
    log_prefix="[RECALL]",
    gsd_mm_per_px=DEFECT_REFERENCE_GSD_MM_PER_PX,
):
    """Recall을 계산해 사람이 읽을 형태로 출력한다."""
    if gt_path is None:
        gt_path = default_gt_path(scenario_name=scenario_name)

    if observation_path is None:
        raise ValueError("observation_path가 필요합니다.")

    catalog = load_defect_gt(gt_path, gsd_mm_per_px=gsd_mm_per_px)
    observations, frame_count = load_defect_observations(observation_path)

    if frame_count == 0:
        print(
            f"{log_prefix} 관측 기록이 없습니다: {observation_path}"
        )

        return None

    result = defect_recall(
        catalog,
        observations,
        min_pixels=min_pixels,
        defect_type=defect_type,
        criterion=criterion,
        min_ratio=min_ratio,
    )
    scope = defect_type or "전체"

    print(
        f"{log_prefix} {scope} Recall: "
        f"{result['detected']}/{result['total']} = "
        f"{result['recall'] * 100:.1f}%  "
        f"(프레임 {frame_count}개 전체에서 한 번이라도 "
        + (
            f"면적의 {min_ratio * 100:.0f}% 이상"
            if min_ratio is not None
            else f"{result['min_pixels']}px 이상"
        )
        + f" 보인 결함, 픽셀 기준 {criterion})"
    )

    for name, info in result["per_defect"].items():
        mark = "O" if info["detected"] else "X"
        frame_text = (
            "미관측"
            if info["best_frame_id"] is None
            else (
                f"최대 frame {info['best_frame_id']}, "
                f"{info['seen_frame_count']}개 프레임에서 보임"
            )
        )
        ratio_text = (
            "   비율 -  "
            if info["area_ratio"] is None
            else f"{info['area_ratio'] * 100:6.1f}% "
        )
        print(
            f"{log_prefix}   [{mark}] {name:22s} "
            f"{info['defect_type']:8s} "
            f"{info['best_pixels']:6d}/{info['ideal_pixels'] or 0:<6d}px "
            f"{ratio_text} {frame_text}"
        )

    # 임계값을 바꾸면 결론이 흔들리는지 함께 보여준다.
    sensitivity = []

    if min_ratio is None:
        for threshold in REPORT_THRESHOLDS:
            trial = defect_recall(
                catalog,
                observations,
                min_pixels=threshold,
                defect_type=defect_type,
                criterion=criterion,
            )
            sensitivity.append(
                f"{threshold}px:{trial['recall'] * 100:.0f}%"
            )
    else:
        for threshold in REPORT_RATIO_THRESHOLDS:
            trial = defect_recall(
                catalog,
                observations,
                defect_type=defect_type,
                criterion=criterion,
                min_ratio=threshold,
            )
            sensitivity.append(
                f"{threshold * 100:.0f}%:{trial['recall'] * 100:.0f}%"
            )

    print(f"{log_prefix}   임계값 민감도  " + "  ".join(sensitivity))

    if min_ratio is not None:
        # 비율의 분모가 '정면 기준'이라 비스듬히 보면 그것만으로 깎인다.
        # 임계값이 사실상 어떤 입사각을 요구하는지 함께 알린다.
        import math

        limit = min(max(float(min_ratio), -1.0), 1.0)
        print(
            f"{log_prefix}   비율 분모는 '정면·공칭거리 픽셀 수'라, "
            f"{min_ratio * 100:.0f}% 기준은 가림이 없어도 "
            f"입사각 {math.degrees(math.acos(limit)):.0f}도 이내에서 "
            "본 적이 있어야 통과한다."
        )
    result["frame_count"] = frame_count

    return result


def report_distance_comparison(
    gt_path,
    observation_path,
    distances_m,
    min_pixels=DEFAULT_MIN_PIXELS,
    criterion="max",
    min_ratio=DEFAULT_MIN_RATIO,
    log_prefix="[RECALL]",
):
    """Recall with the frontal-pixel denominator at several nominal distances.

    Observed pixels are unchanged; only the denominator footprint / GSD^2
    changes, so ratio(d) = ratio(d_ref) * (d / d_ref)^2.
    """
    observations, frame_count = load_defect_observations(observation_path)
    if frame_count == 0 or not distances_m:
        return {}
    results = {}
    for distance in distances_m:
        catalog = load_defect_gt(gt_path, gsd_mm_per_px=gsd_for_distance(distance))
        results[distance] = defect_recall(
            catalog, observations, min_pixels=min_pixels,
            criterion=criterion, min_ratio=min_ratio,
        )
    names = list(next(iter(results.values()))["per_defect"])
    header = "  ".join(f"{d * 1000:.0f}mm" .rjust(13) for d in distances_m)
    print(
        f"{log_prefix} 공칭거리(GSD) 비교: 분모=정면 픽셀 수(면적/GSD^2), "
        f"GSD=1000*d/{DEFECT_REFERENCE_FX_PX:.0f}px, "
        f"기준 {min_ratio * 100:.0f}%"
        if min_ratio is not None
        else f"{log_prefix} 공칭거리(GSD) 비교"
    )
    print(f"{log_prefix}   {'':44s}" + header)
    print(
        f"{log_prefix}   {'GSD [mm/px]':44s}"
        + "  ".join(f"{gsd_for_distance(d):13.4f}" for d in distances_m)
    )
    for name in names:
        cells = []
        for distance in distances_m:
            info = results[distance]["per_defect"][name]
            ratio = info["area_ratio"]
            mark = "O" if info["detected"] else "X"
            cells.append(
                f"{mark} {ratio * 100:6.1f}%" .rjust(13)
                if ratio is not None
                else "-".rjust(13)
            )
        print(f"{log_prefix}   {name:44s}" + "  ".join(cells))
    print(
        f"{log_prefix}   {'Recall':44s}"
        + "  ".join(
            f"{r['detected']}/{r['total']} {r['recall'] * 100:5.1f}%".rjust(13)
            for r in results.values()
        )
    )
    # Same threshold sensitivity line as report(), once per distance.
    for distance in distances_m:
        catalog = load_defect_gt(gt_path, gsd_mm_per_px=gsd_for_distance(distance))
        if min_ratio is None:
            cells = [
                f"{threshold}px:"
                f"{defect_recall(catalog, observations, min_pixels=threshold, criterion=criterion)['recall'] * 100:.0f}%"
                for threshold in REPORT_THRESHOLDS
            ]
        else:
            cells = [
                f"{threshold * 100:.0f}%:"
                f"{defect_recall(catalog, observations, criterion=criterion, min_ratio=threshold)['recall'] * 100:.0f}%"
                for threshold in REPORT_RATIO_THRESHOLDS
            ]
        print(
            f"{log_prefix}   임계값 민감도 {distance * 1000:.0f}mm  "
            + "  ".join(cells)
        )
    return results


def report_defect_recall(
    zivid_save_dir,
    scenario_name=None,
    min_pixels=DEFAULT_MIN_PIXELS,
    criterion="max",
    min_ratio=DEFAULT_MIN_RATIO,
    log_prefix="[RECALL]",
):
    """NBV 시퀀스가 끝난 뒤 부르는 편의 함수.

    GT/관측 파일이 없으면 조용히 None을 돌려준다. 결함을 끄고 돌리는
    경우가 있어서 시퀀스를 실패시키지 않는다.
    """
    observation_path = os.path.join(
        zivid_save_dir,
        DEFAULT_OBSERVATION_FILENAME,
    )

    try:
        gt_path = default_gt_path(scenario_name=scenario_name)
    except ValueError:
        return None

    if not os.path.isfile(gt_path):
        print(f"{log_prefix} GT가 없어 Recall을 건너뜁니다: {gt_path}")

        return None

    if not os.path.isfile(observation_path):
        print(
            f"{log_prefix} 관측 기록이 없어 Recall을 건너뜁니다: "
            f"{observation_path}"
        )

        return None

    overall = report(
        gt_path=gt_path,
        observation_path=observation_path,
        min_pixels=min_pixels,
        criterion=criterion,
        min_ratio=min_ratio,
        log_prefix=log_prefix,
    )
    compare = comparison_distances_from_env()
    if compare:
        distances = [DEFECT_REFERENCE_DISTANCE_M] + [
            d for d in compare if abs(d - DEFECT_REFERENCE_DISTANCE_M) > 1e-9
        ]
        report_distance_comparison(
            gt_path, observation_path, distances,
            min_pixels=min_pixels, criterion=criterion,
            min_ratio=min_ratio, log_prefix=log_prefix,
        )

    # 타입별로도 나눠 본다. scratch만 보고 싶다는 요구가 있었다.
    catalog = load_defect_gt(gt_path)
    types = sorted(
        {info["defect_type"] for info in catalog.values()}
    )

    if len(types) > 1:
        observations, _ = load_defect_observations(observation_path)

        for defect_type in types:
            trial = defect_recall(
                catalog,
                observations,
                min_pixels=min_pixels,
                defect_type=defect_type,
                criterion=criterion,
                min_ratio=min_ratio,
            )
            print(
                f"{log_prefix}   {defect_type:8s} Recall: "
                f"{trial['detected']}/{trial['total']} = "
                f"{trial['recall'] * 100:.1f}%"
            )

    return overall


def parse_args():
    parser = argparse.ArgumentParser(
        description="심어둔 결함 중 몇 개를 관측했는지(Recall) 계산한다."
    )
    parser.add_argument(
        "--gt-path",
        default=None,
        help="defect_gt_<scenario>.json 경로. 생략하면 시나리오로 찾는다.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="GT 시나리오 이름 (기본: Battery_defect_config의 값).",
    )
    parser.add_argument(
        "--observation-path",
        default=None,
        help=(
            "defect_observations.json 경로. 생략하면 "
            "--zivid-dir 아래에서 찾는다."
        ),
    )
    parser.add_argument(
        "--zivid-dir",
        default="/home/seungjun/isaac_ws/zivid_capture",
        help="캡처 저장 디렉터리.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=DEFAULT_MIN_PIXELS,
        help=(
            "검출로 인정할 최소 픽셀 수. 한 뷰에서 이만큼 이상 보인 "
            f"결함만 검출로 센다 (기본 {DEFAULT_MIN_PIXELS})."
        ),
    )
    parser.add_argument(
        "--defect-type",
        default=None,
        help="이 타입만 센다 (예: scratch). 생략하면 전체.",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=DEFAULT_MIN_RATIO,
        help=(
            "검출로 인정할 최소 면적 비율 (관측 픽셀 / 정면 기준 픽셀). "
            f"기본 {DEFAULT_MIN_RATIO}. 0 이하를 주면 비율 대신 "
            "--min-pixels로 판정한다."
        ),
    )
    parser.add_argument(
        "--criterion",
        choices=("max", "sum"),
        default="max",
        help=(
            "결함 하나의 대표 픽셀 수를 정하는 방법. "
            "max=프레임별 최댓값(기본), sum=모든 프레임 합. "
            "어느 쪽이든 '한 번이라도 보인 결함'을 세는 것은 같다."
        ),
    )
    parser.add_argument(
        "--compare-distances",
        default=os.environ.get("BATTERY_RECALL_COMPARE_DISTANCES_M", ""),
        help=(
            "비교할 공칭거리[m], 예: 0.41,0.45. 기본 "
            f"{DEFECT_REFERENCE_DISTANCE_M:.3f} m와 나란히 출력한다."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    observation_path = args.observation_path

    if observation_path is None:
        observation_path = os.path.join(
            args.zivid_dir,
            DEFAULT_OBSERVATION_FILENAME,
        )

    gt_path = args.gt_path

    if gt_path is None:
        gt_path = default_gt_path(scenario_name=args.scenario)

    print()
    print("======================================")
    print("Defect Detection Recall")
    print("======================================")
    print("GT:", gt_path)
    print("관측 기록:", observation_path)
    print("======================================")
    print()

    report(
        gt_path=gt_path,
        observation_path=observation_path,
        min_pixels=args.min_pixels,
        defect_type=args.defect_type,
        criterion=args.criterion,
        min_ratio=(
            args.min_ratio if args.min_ratio > 0.0 else None
        ),
    )
    compare = [float(v) for v in args.compare_distances.split(",") if v.strip()]
    if compare:
        print()
        report_distance_comparison(
            gt_path, observation_path,
            [DEFECT_REFERENCE_DISTANCE_M] + [
                d for d in compare if abs(d - DEFECT_REFERENCE_DISTANCE_M) > 1e-9
            ],
            min_pixels=args.min_pixels, criterion=args.criterion,
            min_ratio=args.min_ratio if args.min_ratio > 0.0 else None,
        )


if __name__ == "__main__":
    main()
