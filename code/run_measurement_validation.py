from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.stats import spearmanr
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
SEED = 20261030
RNG = np.random.default_rng(SEED)
DIMENSIONS = ('opportunity_framing', 'evidence_falsifiability', 'resource_finance', 'adaptive_action', 'stakeholder_ethics')
WEEKS = (1, 2, 3, 4)
TURNS = tuple(range(1, 17))
WEEK_END_TURNS = {week: week * 4 for week in WEEKS}
TURN_CENTER = 8.5
TURN_SCALE = 7.5
STATE_COLUMNS = ('state_z_cash', 'state_z_team', 'state_z_product', 'state_z_market_evidence', 'state_z_reputation', 'state_z_compliance')
TRUE_FACET_COLUMNS = ('blueprint_difficulty_true', 'surface_effect_true', 'state_effect_true', 'personalized_path_difficulty_true', 'rater_severity_true', 'version_effect_true')
METHOD_ORDER = ('ecd_equal_weight', 'final_resource_state', 'raw_rubric_mean', 'ctt_anchor_equating', 'static_multifacet', 'no_anchor_dynamic', 'anchored_dynamic_simple', 'full_common_anchor_dynamic_multifacet', 'oracle')
METHOD_LABELS = {'ecd_equal_weight': 'Equal-weight ECD aggregation', 'final_resource_state': 'Final resource state', 'raw_rubric_mean': 'Raw rubric mean', 'ctt_anchor_equating': 'CTT anchor equating', 'static_multifacet': 'Static multifacet', 'no_anchor_dynamic': 'Unanchored dynamic', 'anchored_dynamic_simple': 'Simplified anchored dynamic', 'full_common_anchor_dynamic_multifacet': 'Complete anchored dynamic multifacet', 'oracle': 'Oracle (simulation only)'}
SMOOTHING_GRID = (0.5, 2.0, 8.0, 32.0)
RIDGE_ALPHA_GRID = (0.3, 2.0, 10.0)
ANCHOR_WEIGHT_GRID = (1.5, 3.0)
RIDGE_LEVEL = 0.65
RIDGE_GLOBAL = 1.25
ANCHOR_WEIGHT = 1.35
DYNAMIC_PRIOR_STRENGTH = 0.3
N_BOOTSTRAP = 2000
FULL_METHOD = 'full_common_anchor_dynamic_multifacet'
MULTIRATER_METHODS = {'static_multifacet', 'no_anchor_dynamic', 'anchored_dynamic_simple', FULL_METHOD, 'oracle'}

def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__}')

def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_jsonable) + '\n', encoding='utf-8')

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def _ratings_input_paths(data_dir: Path) -> list[Path]:
    """Use the original CSV when available, otherwise require both ordered parts."""
    original = data_dir / 'ratings_long.csv'
    if original.is_file():
        return [original]
    parts = [data_dir / f'ratings_long_part{index:02d}.csv' for index in (1, 2)]
    for path in parts:
        if not path.is_file():
            raise FileNotFoundError(f'Missing ratings input: {path}; both CSV parts are required')
    return parts

def _ordinal_signal(values: pd.Series | np.ndarray) -> np.ndarray:
    lookup = np.array([-1.55, -0.7, 0.0, 0.7, 1.55], dtype=float)
    array = np.asarray(values, dtype=float)
    integer = np.clip(np.rint(array).astype(int), 0, 4)
    return lookup[integer]

def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.nanstd(x[mask]) < 1e-12 or np.nanstd(y[mask]) < 1e-12:
        return float('nan')
    return float(spearmanr(x[mask], y[mask]).statistic)

def _stable_uniform(*parts: object) -> float:
    text = '\x1f'.join(map(str, (SEED, *parts)))
    raw = hashlib.sha256(text.encode('utf-8')).digest()
    return int.from_bytes(raw[:8], 'big') / 2 ** 64

def _is_missing(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({'1', 'true', 'yes'})

def _validate_input(obs: pd.DataFrame) -> None:
    required = {'split', 'learner_id', 'turn', 'week', 'event_kind', 'blueprint_id', 'form_id', 'form_kind', 'target_dimension', 'primary_rater_id', 'primary_version_id', 'rating_time_id', 'score', 'raw_score', 'rule_score', 'score_missing', 'theta_true', 'group', *STATE_COLUMNS, *TRUE_FACET_COLUMNS}
    missing = sorted(required - set(obs.columns))
    if missing:
        raise ValueError(f'trajectory_observations.csv is missing columns: {missing}')
    unknown_dimensions = sorted(set(obs['target_dimension'].unique()) - set(DIMENSIONS))
    if unknown_dimensions:
        raise ValueError(f'Unknown target dimensions: {unknown_dimensions}')
    expected = {'calibration': 600, 'validation': 150, 'test': 300}
    actual = obs.groupby('split')['learner_id'].nunique().to_dict()
    if actual != expected:
        raise ValueError(f'Split learner counts differ from frozen design: {actual}')
    learner_splits = obs.groupby('learner_id')['split'].nunique()
    if int(learner_splits.max()) != 1:
        raise ValueError('A learner appears in more than one split')
    if len(obs) != 50400:
        raise ValueError(f'Expected 50,400 observation rows, found {len(obs):,}')

def _build_multirater_observations(observations: pd.DataFrame, ratings: pd.DataFrame, facet_truth: pd.DataFrame) -> pd.DataFrame:
    keys = ['split', 'learner_id', 'event_id', 'target_dimension']
    required = {*keys, 'rating_id', 'rating_round', 'rater_id', 'rating_time_id', 'version_id', 'score', 'is_missing'}
    missing = sorted(required - set(ratings.columns))
    if missing:
        raise ValueError(f'Ratings input is missing columns: {missing}')
    if len(ratings) != 151200:
        raise ValueError(f'Expected 151,200 rating rows, found {len(ratings):,}')
    counts = ratings.groupby(keys).size()
    if not counts.eq(3).all():
        raise ValueError('Every event×dimension must have exactly three rating records')
    event_metadata = observations.drop(columns=['primary_rater_id', 'primary_version_id', 'rating_time_id', 'score', 'score_missing', 'rater_severity_true', 'version_effect_true'])
    rating_fields = ratings[keys + ['rating_id', 'rating_round', 'rater_id', 'rating_time_id', 'version_id', 'score', 'is_missing']].rename(columns={'rater_id': 'primary_rater_id', 'version_id': 'primary_version_id', 'is_missing': 'score_missing'})
    expanded = rating_fields.merge(event_metadata, on=keys, how='left', validate='many_to_one')
    rater_truth = facet_truth.loc[facet_truth['facet_type'].eq('rater_severity'), ['facet_id', 'dimension', 'time_id', 'value_true']].rename(columns={'facet_id': 'primary_rater_id', 'dimension': 'target_dimension', 'time_id': 'rating_time_id', 'value_true': 'rater_severity_true'})
    version_truth = facet_truth.loc[facet_truth['facet_type'].eq('version_effect'), ['facet_id', 'dimension', 'value_true']].rename(columns={'facet_id': 'primary_version_id', 'dimension': 'target_dimension', 'value_true': 'version_effect_true'})
    expanded['rating_time_id'] = expanded['rating_time_id'].astype(float)
    rater_truth['rating_time_id'] = rater_truth['rating_time_id'].astype(float)
    expanded = expanded.merge(rater_truth, on=['primary_rater_id', 'target_dimension', 'rating_time_id'], how='left', validate='many_to_one').merge(version_truth, on=['primary_version_id', 'target_dimension'], how='left', validate='many_to_one')
    if expanded[['rater_severity_true', 'version_effect_true']].isna().any().any():
        raise ValueError('Could not resolve all rater/version truth cells for audit diagnostics')
    return expanded

def _normalize_ability_truth(path: Path, obs: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        truth = pd.read_csv(path)
        rename = {}
        if 'dimension' in truth.columns and 'target_dimension' not in truth.columns:
            rename['dimension'] = 'target_dimension'
        if 'theta' in truth.columns and 'theta_true' not in truth.columns:
            rename['theta'] = 'theta_true'
        truth = truth.rename(columns=rename)
        needed = {'learner_id', 'turn', 'week', 'target_dimension', 'theta_true'}
        if not needed.issubset(truth.columns):
            truth = pd.DataFrame()
    else:
        truth = pd.DataFrame()
    if truth.empty:
        truth = obs[['split', 'learner_id', 'turn', 'week', 'target_dimension', 'theta_true', 'group']].drop_duplicates()
    else:
        learner_meta = obs[['learner_id', 'split', 'group']].drop_duplicates('learner_id')
        truth = truth.merge(learner_meta, on='learner_id', how='left', suffixes=('', '_obs'))
        for column in ('split', 'group'):
            other = f'{column}_obs'
            if other in truth.columns:
                truth[column] = truth.get(column).fillna(truth[other])
                truth = truth.drop(columns=other)
    truth['week'] = truth['week'].astype(int)
    truth['turn'] = truth['turn'].astype(int)
    weekly = truth.sort_values('turn').groupby(['split', 'learner_id', 'group', 'target_dimension', 'week'], as_index=False).tail(1)[['split', 'learner_id', 'group', 'target_dimension', 'week', 'theta_true']].drop_duplicates(['learner_id', 'target_dimension', 'week'])
    expected_rows = obs['learner_id'].nunique() * len(DIMENSIONS) * len(WEEKS)
    if len(weekly) != expected_rows:
        raise ValueError(f'Ability truth must cover every learner×dimension×week ({expected_rows}); found {len(weekly)}')
    return weekly.reset_index(drop=True)

def _endpoint_truth(weekly_truth: pd.DataFrame) -> pd.DataFrame:
    wide = weekly_truth.pivot_table(index=['split', 'learner_id', 'group', 'target_dimension'], columns='week', values='theta_true', aggfunc='first').reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={1: 'true_week1', 4: 'true_week4'})
    wide['true_change'] = wide['true_week4'] - wide['true_week1']
    return wide

@dataclass(frozen=True)
class FeatureSpec:
    blueprint: bool = True
    surface: bool = True
    state: bool = True
    rater_time: bool = True
    version: bool = True
    path: bool = True

class FeatureEncoder:

    def __init__(self, spec: FeatureSpec, dynamic: bool):
        self.spec = spec
        self.dynamic = dynamic
        self.categories: dict[str, tuple[str, ...]] = {}
        self.slices: dict[str, slice] = {}
        self.feature_names: list[str] = []

    @staticmethod
    def _surface_key(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(np.where(frame['form_kind'].astype(str).eq('surface'), frame['form_id'].astype(str), '__REFERENCE__'), index=frame.index)

    @staticmethod
    def _rater_time_key(frame: pd.DataFrame) -> pd.Series:
        return frame['primary_rater_id'].astype(str) + '@' + frame['rating_time_id'].astype(str)

    @staticmethod
    def _path_key(frame: pd.DataFrame) -> pd.Series:
        is_anchor = frame['event_kind'].astype(str).str.contains('anchor')
        return pd.Series(np.where(is_anchor, '__COMMON_ANCHOR__', frame['profile_stratum'].astype(str) + '::' + frame['blueprint_id'].astype(str)), index=frame.index)

    def fit(self, frame: pd.DataFrame) -> 'FeatureEncoder':
        if self.spec.blueprint:
            self.categories['blueprint'] = tuple(sorted(frame['blueprint_id'].astype(str).unique()))
        if self.spec.surface:
            self.categories['surface'] = tuple(sorted(self._surface_key(frame).unique()))
        if self.spec.rater_time:
            self.categories['rater_time'] = tuple(sorted(self._rater_time_key(frame).unique()))
        if self.spec.version:
            self.categories['version'] = tuple(sorted(frame['primary_version_id'].astype(str).unique()))
        if self.spec.path:
            self.categories['path'] = tuple(sorted(self._path_key(frame).unique()))
        self._layout()
        return self

    def _layout(self) -> None:
        names = ['ability_mean:intercept']
        self.slices['ability_mean'] = slice(0, 1 + int(self.dynamic))
        if self.dynamic:
            names.append('ability_mean:turn_center')
        cursor = len(names)
        for component in ('blueprint', 'surface', 'rater_time', 'version', 'path'):
            if component not in self.categories:
                continue
            levels = self.categories[component][1:]
            self.slices[component] = slice(cursor, cursor + len(levels))
            names.extend((f'{component}:{level}' for level in levels))
            cursor += len(levels)
        if self.spec.state:
            self.slices['state'] = slice(cursor, cursor + len(STATE_COLUMNS))
            names.extend((f'state:{column}' for column in STATE_COLUMNS))
        self.feature_names = names

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        n = len(frame)
        blocks: list[sparse.csr_matrix] = [sparse.csr_matrix(np.ones((n, 1)))]
        if self.dynamic:
            blocks.append(sparse.csr_matrix(((frame['turn'].to_numpy(float) - TURN_CENTER) / TURN_SCALE)[:, None]))
        sources: dict[str, pd.Series] = {'blueprint': frame['blueprint_id'].astype(str), 'surface': self._surface_key(frame), 'rater_time': self._rater_time_key(frame), 'version': frame['primary_version_id'].astype(str), 'path': self._path_key(frame)}
        for component in ('blueprint', 'surface', 'rater_time', 'version', 'path'):
            if component not in self.categories:
                continue
            levels = self.categories[component][1:]
            level_map = {value: index for index, value in enumerate(levels)}
            rows: list[int] = []
            cols: list[int] = []
            for row_index, value in enumerate(sources[component].tolist()):
                column = level_map.get(value)
                if column is not None:
                    rows.append(row_index)
                    cols.append(column)
            data = np.ones(len(rows), dtype=float)
            blocks.append(sparse.csr_matrix((data, (rows, cols)), shape=(n, len(levels))))
        if self.spec.state:
            blocks.append(sparse.csr_matrix(frame.loc[:, STATE_COLUMNS].to_numpy(float)))
        return sparse.hstack(blocks, format='csr')

class DimensionMAPModel:

    def __init__(self, dimension: str, *, spec: FeatureSpec, dynamic: bool, include_anchor: bool, include_main: bool, smooth_lambda: float, method: str):
        self.dimension = dimension
        self.spec = spec
        self.dynamic = dynamic
        self.include_anchor = include_anchor
        self.include_main = include_main
        self.smooth_lambda = float(smooth_lambda)
        self.method = method
        self.encoder = FeatureEncoder(spec, dynamic)
        self.global_coef: np.ndarray | None = None
        self.train_learners: tuple[str, ...] = ()
        self.train_local: np.ndarray | None = None
        self.fit_diagnostics: dict[str, Any] = {}

    def _select(self, frame: pd.DataFrame) -> pd.DataFrame:
        selected = frame.loc[frame['target_dimension'].eq(self.dimension)].copy()
        missing = _is_missing(selected['score_missing']) | selected['score'].isna()
        selected = selected.loc[~missing]
        is_anchor = selected['event_kind'].astype(str).str.contains('anchor')
        if not self.include_anchor:
            selected = selected.loc[~is_anchor]
        if not self.include_main:
            selected = selected.loc[is_anchor]
        return selected.reset_index(drop=True)

    def fit(self, frame: pd.DataFrame) -> 'DimensionMAPModel':
        selected = self._select(frame)
        if selected.empty:
            raise ValueError(f'No observations for {self.method}/{self.dimension}')
        self.encoder.fit(selected)
        x_global = self.encoder.transform(selected)
        learners = tuple(sorted(selected['learner_id'].astype(str).unique()))
        learner_map = {value: index for index, value in enumerate(learners)}
        if self.dynamic:
            local_index = np.array([learner_map[str(p)] * 4 + int(w) - 1 for p, w in zip(selected['learner_id'], selected['week'])], dtype=int)
            n_local = len(learners) * 4
        else:
            local_index = np.array([learner_map[str(p)] for p in selected['learner_id']], dtype=int)
            n_local = len(learners)
        z_local = sparse.csr_matrix((np.ones(len(selected)), (np.arange(len(selected)), local_index)), shape=(len(selected), n_local))
        design = sparse.hstack([x_global, z_local], format='csr')
        y = _ordinal_signal(selected['score'])
        anchor = selected['event_kind'].astype(str).str.contains('anchor').to_numpy()
        weights = np.where(anchor, ANCHOR_WEIGHT, 1.0)
        root_w = np.sqrt(weights)
        weighted_design = sparse.diags(root_w) @ design
        weighted_y = root_w * y
        n_global = self.encoder.n_features
        penalty_blocks: list[sparse.csr_matrix] = []
        penalty_rhs: list[np.ndarray] = []
        global_penalty = np.full(n_global, math.sqrt(RIDGE_GLOBAL), dtype=float)
        ability_slice = self.encoder.slices['ability_mean']
        global_penalty[ability_slice] = math.sqrt(1e-05)
        penalty_blocks.append(sparse.hstack([sparse.diags(global_penalty), sparse.csr_matrix((n_global, n_local))], format='csr'))
        penalty_rhs.append(np.zeros(n_global))
        level_block = sparse.hstack([sparse.csr_matrix((n_local, n_global)), sparse.eye(n_local, format='csr') * math.sqrt(RIDGE_LEVEL)], format='csr')
        penalty_blocks.append(level_block)
        penalty_rhs.append(np.zeros(n_local))
        if self.dynamic:
            rows: list[int] = []
            cols: list[int] = []
            data: list[float] = []
            row = 0
            for learner_index in range(len(learners)):
                base = learner_index * 4
                for week_index in range(1, 4):
                    rows.extend([row, row])
                    cols.extend([base + week_index - 1, base + week_index])
                    data.extend([-math.sqrt(self.smooth_lambda), math.sqrt(self.smooth_lambda)])
                    row += 1
            difference = sparse.csr_matrix((data, (rows, cols)), shape=(row, n_local))
            penalty_blocks.append(sparse.hstack([sparse.csr_matrix((row, n_global)), difference], format='csr'))
            penalty_rhs.append(np.zeros(row))
        augmented = sparse.vstack([weighted_design, *penalty_blocks], format='csr')
        target = np.concatenate([weighted_y, *penalty_rhs])
        solution = lsqr(augmented, target, atol=1e-08, btol=1e-08, iter_lim=6000)
        coef = solution[0]
        self.global_coef = coef[:n_global]
        self.train_local = coef[n_global:]
        self.train_learners = learners
        self.fit_diagnostics = {'istop': int(solution[1]), 'iterations': int(solution[2]), 'residual_norm': float(solution[3]), 'condition_estimate': float(solution[6]), 'n_observations': int(len(selected)), 'n_global': int(n_global), 'n_local': int(n_local)}
        return self

    def _ability_mean(self, weeks: np.ndarray) -> np.ndarray:
        if self.global_coef is None:
            raise RuntimeError('Model is not fitted')
        sl = self.encoder.slices['ability_mean']
        coef = self.global_coef[sl]
        if self.dynamic:
            return coef[0] + coef[1] * ((weeks.astype(float) * 4.0 - TURN_CENTER) / TURN_SCALE)
        return np.full(len(weeks), coef[0], dtype=float)

    def raw_training_ability(self) -> pd.DataFrame:
        if self.train_local is None:
            raise RuntimeError('Model is not fitted')
        rows: list[dict[str, Any]] = []
        for learner_index, learner_id in enumerate(self.train_learners):
            if self.dynamic:
                local = self.train_local[learner_index * 4:learner_index * 4 + 4]
            else:
                local = np.repeat(self.train_local[learner_index], 4)
            mean = self._ability_mean(np.array(WEEKS))
            for week_index, week in enumerate(WEEKS):
                rows.append({'learner_id': learner_id, 'target_dimension': self.dimension, 'week': week, 'raw_prediction': float(mean[week_index] + local[week_index])})
        return pd.DataFrame(rows)

    def predict_raw(self, frame: pd.DataFrame, learner_ids: Sequence[str]) -> pd.DataFrame:
        if self.global_coef is None:
            raise RuntimeError('Model is not fitted')
        selected = self._select(frame)
        if not selected.empty:
            x_global = self.encoder.transform(selected)
            global_prediction = np.asarray(x_global @ self.global_coef).ravel()
            selected = selected.assign(y_signal=_ordinal_signal(selected['score']), global_prediction=global_prediction, local_residual=lambda x: x['y_signal'] - x['global_prediction'], observation_weight=np.where(selected['event_kind'].astype(str).str.contains('anchor'), ANCHOR_WEIGHT, 1.0))
        groups = {str(key): group for key, group in selected.groupby(selected['learner_id'].astype(str))} if not selected.empty else {}
        rows: list[dict[str, Any]] = []
        week_array = np.array(WEEKS)
        mean = self._ability_mean(week_array)
        for learner_id in map(str, learner_ids):
            group = groups.get(learner_id)
            if self.dynamic:
                matrix = np.eye(4) * RIDGE_LEVEL
                rhs = np.zeros(4)
                for index in range(3):
                    matrix[index, index] += self.smooth_lambda
                    matrix[index + 1, index + 1] += self.smooth_lambda
                    matrix[index, index + 1] -= self.smooth_lambda
                    matrix[index + 1, index] -= self.smooth_lambda
                if group is not None:
                    for item in group.itertuples(index=False):
                        wi = int(item.week) - 1
                        weight = float(item.observation_weight)
                        matrix[wi, wi] += weight
                        rhs[wi] += weight * float(item.local_residual)
                local = np.linalg.solve(matrix, rhs)
            else:
                numerator = 0.0
                denominator = RIDGE_LEVEL
                if group is not None:
                    numerator = float((group['observation_weight'] * group['local_residual']).sum())
                    denominator += float(group['observation_weight'].sum())
                local = np.repeat(numerator / denominator, 4)
            for wi, week in enumerate(WEEKS):
                rows.append({'learner_id': learner_id, 'target_dimension': self.dimension, 'week': week, 'raw_prediction': float(mean[wi] + local[wi])})
        return pd.DataFrame(rows)

    def component_effect(self, frame: pd.DataFrame, component: str) -> np.ndarray:
        if self.global_coef is None:
            raise RuntimeError('Model is not fitted')
        sl = self.encoder.slices.get(component)
        if sl is None or sl.stop == sl.start:
            return np.zeros(len(frame))
        x = self.encoder.transform(frame)
        return np.asarray(x[:, sl] @ self.global_coef[sl]).ravel()

class AffineScale:

    def __init__(self):
        self.parameters: dict[str, tuple[float, float]] = {}

    def fit(self, raw: pd.DataFrame, truth: pd.DataFrame) -> 'AffineScale':
        joined = raw.merge(truth[['learner_id', 'target_dimension', 'week', 'theta_true']], on=['learner_id', 'target_dimension', 'week'], how='inner', validate='one_to_one')
        for dimension in DIMENSIONS:
            part = joined.loc[joined['target_dimension'].eq(dimension)]
            x = part['raw_prediction'].to_numpy(float)
            y = part['theta_true'].to_numpy(float)
            design = np.column_stack([np.ones(len(x)), x])
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            slope = float(np.clip(coef[1], 0.15, 5.0))
            intercept = float(y.mean() - slope * x.mean())
            self.parameters[dimension] = (intercept, slope)
        return self

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        result = raw.copy()
        result['prediction'] = [self.parameters[str(d)][0] + self.parameters[str(d)][1] * float(value) for d, value in zip(result['target_dimension'], result['raw_prediction'])]
        return result

class MultiDimensionMAP:

    def __init__(self, method: str, *, spec: FeatureSpec, dynamic: bool, include_anchor: bool=True, include_main: bool=True, smooth_lambda: float=2.0):
        self.method = method
        self.spec = spec
        self.dynamic = dynamic
        self.include_anchor = include_anchor
        self.include_main = include_main
        self.smooth_lambda = smooth_lambda
        self.models: dict[str, DimensionMAPModel] = {}
        self.scale = AffineScale()

    def fit(self, calibration: pd.DataFrame, truth: pd.DataFrame) -> 'MultiDimensionMAP':
        raw_parts = []
        for dimension in DIMENSIONS:
            model = DimensionMAPModel(dimension, spec=self.spec, dynamic=self.dynamic, include_anchor=self.include_anchor, include_main=self.include_main, smooth_lambda=self.smooth_lambda, method=self.method).fit(calibration)
            self.models[dimension] = model
            raw_parts.append(model.raw_training_ability())
        self.scale.fit(pd.concat(raw_parts, ignore_index=True), truth)
        return self

    def predict_weekly(self, frame: pd.DataFrame, learner_ids: Sequence[str]) -> pd.DataFrame:
        raw = pd.concat([self.models[dimension].predict_raw(frame, learner_ids) for dimension in DIMENSIONS], ignore_index=True)
        return self.scale.transform(raw)

class CalibratedDimensionModel:
    facet_direction = 1.0

    def __init__(self, dimension: str, *, spec: FeatureSpec, dynamic: bool, include_anchor: bool, include_main: bool, smooth_lambda: float, ridge_alpha: float, anchor_weight: float, method: str):
        self.dimension = dimension
        self.spec = spec
        self.dynamic = dynamic
        self.include_anchor = include_anchor
        self.include_main = include_main
        self.smooth_lambda = float(smooth_lambda)
        self.ridge_alpha = float(ridge_alpha)
        self.anchor_weight = float(anchor_weight)
        self.method = method
        self.encoder = FeatureEncoder(spec, dynamic)
        self.regression: Ridge | None = None
        self.global_coef: np.ndarray | None = None
        self.turn_prior: dict[int, float] = {}
        self.trajectory_prior = np.zeros(2, dtype=float)
        self.fit_diagnostics: dict[str, Any] = {}

    def _select(self, frame: pd.DataFrame) -> pd.DataFrame:
        selected = frame.loc[frame['target_dimension'].eq(self.dimension)].copy()
        selected = selected.loc[~(_is_missing(selected['score_missing']) | selected['score'].isna())]
        is_anchor = selected['event_kind'].astype(str).str.contains('anchor')
        if not self.include_anchor:
            selected = selected.loc[~is_anchor]
        if not self.include_main:
            selected = selected.loc[is_anchor]
        return selected.reset_index(drop=True)

    @staticmethod
    def _score_block(frame: pd.DataFrame) -> sparse.csr_matrix:
        score = np.clip(np.rint(frame['score'].to_numpy(float)).astype(int), 0, 4)
        blocks: list[sparse.csr_matrix] = [sparse.csr_matrix(_ordinal_signal(score)[:, None])]
        rows: list[int] = []
        cols: list[int] = []
        for row, value in enumerate(score):
            if value > 0:
                rows.append(row)
                cols.append(value - 1)
        blocks.append(sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(frame), 4)))
        return sparse.hstack(blocks, format='csr')

    def _design(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        return sparse.hstack([self.encoder.transform(frame), self._score_block(frame)], format='csr')

    def fit(self, calibration: pd.DataFrame) -> 'CalibratedDimensionModel':
        selected = self._select(calibration)
        if selected.empty:
            raise ValueError(f'No observations for {self.method}/{self.dimension}')
        self.encoder.fit(selected)
        design = self._design(selected)
        target = selected['theta_true'].to_numpy(float)
        is_anchor = selected['event_kind'].astype(str).str.contains('anchor').to_numpy()
        weights = np.where(is_anchor, self.anchor_weight, 1.0)
        regression = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        regression.fit(design, target, sample_weight=weights)
        self.regression = regression
        self.global_coef = np.asarray(regression.coef_[:self.encoder.n_features], dtype=float)
        means = selected.groupby('turn')['theta_true'].mean()
        overall = float(selected['theta_true'].mean())
        self.turn_prior = {turn: float(means.get(turn, overall)) for turn in TURNS}
        prior_x = (np.asarray(TURNS, dtype=float) - TURN_CENTER) / TURN_SCALE
        self.trajectory_prior, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(TURNS)), prior_x]), np.array([self.turn_prior[turn] for turn in TURNS], dtype=float), rcond=None)
        fitted = regression.predict(design)
        residual = fitted - target
        self.fit_diagnostics = {'n_observations': int(len(selected)), 'n_features': int(design.shape[1]), 'ridge_alpha': self.ridge_alpha, 'anchor_weight': self.anchor_weight, 'calibration_rmse': float(np.sqrt(np.mean(residual ** 2))), 'calibration_bias': float(np.mean(residual))}
        return self

    def _event_predictions(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.regression is None:
            raise RuntimeError('Model is not fitted')
        selected = self._select(frame)
        if selected.empty:
            return selected.assign(event_prediction=np.nan, observation_weight=np.nan)
        prediction = self.regression.predict(self._design(selected))
        return selected.assign(event_prediction=prediction, observation_weight=np.where(selected['event_kind'].astype(str).str.contains('anchor'), self.anchor_weight, 1.0))

    def predict_raw(self, frame: pd.DataFrame, learner_ids: Sequence[str]) -> pd.DataFrame:
        events = self._event_predictions(frame)
        groups = {str(key): group for key, group in events.groupby(events['learner_id'].astype(str))} if not events.empty else {}
        rows: list[dict[str, Any]] = []
        report_x = np.array([(WEEK_END_TURNS[week] - TURN_CENTER) / TURN_SCALE for week in WEEKS], dtype=float)
        for learner_id in map(str, learner_ids):
            group = groups.get(learner_id)
            if self.dynamic:
                precision = np.diag([DYNAMIC_PRIOR_STRENGTH, self.smooth_lambda])
                rhs = precision @ self.trajectory_prior
                if group is not None:
                    x = (group['turn'].to_numpy(float) - TURN_CENTER) / TURN_SCALE
                    design = np.column_stack([np.ones(len(group)), x])
                    weights = group['observation_weight'].to_numpy(float)
                    target = group['event_prediction'].to_numpy(float)
                    precision = precision + design.T @ (weights[:, None] * design)
                    rhs = rhs + design.T @ (weights * target)
                coefficients = np.linalg.solve(precision, rhs)
                values = coefficients[0] + coefficients[1] * report_x
            else:
                numerator = DYNAMIC_PRIOR_STRENGTH * float(self.trajectory_prior[0])
                denominator = DYNAMIC_PRIOR_STRENGTH
                if group is not None:
                    numerator += float((group['observation_weight'] * group['event_prediction']).sum())
                    denominator += float(group['observation_weight'].sum())
                values = np.repeat(numerator / denominator, 4)
            for week, value in zip(WEEKS, values):
                rows.append({'learner_id': learner_id, 'target_dimension': self.dimension, 'week': week, 'raw_prediction': float(value)})
        return pd.DataFrame(rows)

    def component_effect(self, frame: pd.DataFrame, component: str) -> np.ndarray:
        if self.global_coef is None:
            raise RuntimeError('Model is not fitted')
        sl = self.encoder.slices.get(component)
        if sl is None or sl.stop == sl.start:
            return np.zeros(len(frame))
        x = self.encoder.transform(frame)
        return np.asarray(x[:, sl] @ self.global_coef[sl]).ravel()

class CalibratedMultifacetModel:
    facet_direction = 1.0

    def __init__(self, method: str, *, spec: FeatureSpec, dynamic: bool, include_anchor: bool=True, include_main: bool=True, smooth_lambda: float=2.0, ridge_alpha: float=1.0, anchor_weight: float=2.0):
        self.method = method
        self.spec = spec
        self.dynamic = dynamic
        self.include_anchor = include_anchor
        self.include_main = include_main
        self.smooth_lambda = smooth_lambda
        self.ridge_alpha = ridge_alpha
        self.anchor_weight = anchor_weight
        self.models: dict[str, CalibratedDimensionModel] = {}
        self.scale = AffineScale()

    def fit(self, calibration: pd.DataFrame, truth: pd.DataFrame) -> 'CalibratedMultifacetModel':
        raw_parts = []
        learners = sorted(calibration['learner_id'].astype(str).unique())
        for dimension in DIMENSIONS:
            model = CalibratedDimensionModel(dimension, spec=self.spec, dynamic=self.dynamic, include_anchor=self.include_anchor, include_main=self.include_main, smooth_lambda=self.smooth_lambda, ridge_alpha=self.ridge_alpha, anchor_weight=self.anchor_weight, method=self.method).fit(calibration)
            self.models[dimension] = model
            raw_parts.append(model.predict_raw(calibration, learners))
        self.scale.fit(pd.concat(raw_parts, ignore_index=True), truth)
        return self

    def predict_weekly(self, frame: pd.DataFrame, learner_ids: Sequence[str]) -> pd.DataFrame:
        raw = pd.concat([self.models[dimension].predict_raw(frame, learner_ids) for dimension in DIMENSIONS], ignore_index=True)
        return self.scale.transform(raw)

def _weekly_to_endpoints(weekly: pd.DataFrame, method: str, split: str) -> pd.DataFrame:
    pivot = weekly.pivot_table(index=['learner_id', 'target_dimension'], columns='week', values='prediction', aggfunc='first').reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={1: 'pred_week1', 4: 'pred_week4'})
    pivot['pred_change'] = pivot['pred_week4'] - pivot['pred_week1']
    pivot['method'] = method
    pivot['method_label'] = METHOD_LABELS.get(method, method)
    pivot['split'] = split
    return pivot

class AggregateBaseline:

    def __init__(self, method: str, value_column: str, *, anchors_only: bool=False, ctt: bool=False):
        self.method = method
        self.value_column = value_column
        self.anchors_only = anchors_only
        self.ctt = ctt
        self.prior: dict[tuple[str, int], float] = {}
        self.form_stats: pd.DataFrame | None = None
        self.scale = AffineScale()

    def _prepare(self, frame: pd.DataFrame, *, fitting: bool) -> pd.DataFrame:
        selected = frame.copy()
        if self.anchors_only:
            selected = selected.loc[selected['event_kind'].astype(str).str.contains('anchor')]
        selected = selected.loc[~selected[self.value_column].isna()]
        if self.ctt:
            keys = ['target_dimension', 'blueprint_id', 'form_id', 'form_kind']
            if fitting:
                stats = selected.groupby(keys)[self.value_column].agg(['mean', 'std', 'count']).reset_index()
                stats['std'] = stats['std'].fillna(0.0).clip(lower=0.35)
                self.form_stats = stats
            if self.form_stats is None:
                raise RuntimeError('CTT form statistics are not fitted')
            exact = self.form_stats.loc[self.form_stats['form_kind'].eq('exact')].rename(columns={'mean': 'exact_mean', 'std': 'exact_std'})[['target_dimension', 'blueprint_id', 'exact_mean', 'exact_std']]
            stats = self.form_stats.rename(columns={'mean': 'form_mean', 'std': 'form_std'})
            selected = selected.merge(stats[keys + ['form_mean', 'form_std']], on=keys, how='left')
            selected = selected.merge(exact, on=['target_dimension', 'blueprint_id'], how='left')
            fallback = selected.groupby('target_dimension')[self.value_column].transform('mean')
            selected['_value'] = (selected[self.value_column] - selected['form_mean'].fillna(fallback)) / selected['form_std'].fillna(1.0).clip(lower=0.35) * selected['exact_std'].fillna(1.0) + selected['exact_mean'].fillna(fallback)
        else:
            selected['_value'] = selected[self.value_column].astype(float)
        return selected

    def _raw_weekly(self, frame: pd.DataFrame, learner_ids: Sequence[str], *, fitting: bool) -> pd.DataFrame:
        selected = self._prepare(frame, fitting=fitting)
        grouped = selected.groupby(['learner_id', 'target_dimension', 'week'], as_index=False)['_value'].mean().rename(columns={'_value': 'raw_prediction'})
        if fitting:
            means = grouped.groupby(['target_dimension', 'week'])['raw_prediction'].mean()
            self.prior = {(str(d), int(w)): float(v) for (d, w), v in means.items()}
        rows: list[dict[str, Any]] = []
        lookup = {(str(row.learner_id), str(row.target_dimension), int(row.week)): float(row.raw_prediction) for row in grouped.itertuples(index=False)}
        for learner_id in map(str, learner_ids):
            for dimension in DIMENSIONS:
                observed_weeks = np.array([week for week in WEEKS if (learner_id, dimension, week) in lookup], dtype=float)
                observed_values = np.array([lookup[learner_id, dimension, int(week)] for week in observed_weeks], dtype=float)
                if len(observed_weeks):
                    values = np.interp(np.array(WEEKS, dtype=float), observed_weeks, observed_values)
                else:
                    values = np.array([self.prior[dimension, week] for week in WEEKS])
                for week, value in zip(WEEKS, values):
                    rows.append({'learner_id': learner_id, 'target_dimension': dimension, 'week': week, 'raw_prediction': float(value)})
        return pd.DataFrame(rows)

    def fit(self, calibration: pd.DataFrame, truth: pd.DataFrame) -> 'AggregateBaseline':
        learners = sorted(calibration['learner_id'].astype(str).unique())
        raw = self._raw_weekly(calibration, learners, fitting=True)
        self.scale.fit(raw, truth)
        return self

    def predict_weekly(self, frame: pd.DataFrame, learner_ids: Sequence[str]) -> pd.DataFrame:
        return self.scale.transform(self._raw_weekly(frame, learner_ids, fitting=False))

class OracleDynamic:

    def __init__(self, smooth_lambda: float):
        self.method = 'oracle'
        self.smooth_lambda = smooth_lambda
        self.prior: dict[tuple[str, int], float] = {}
        self.trajectory_prior: dict[str, np.ndarray] = {}
        self.category_locations: dict[str, dict[int, float]] = {}
        self.scale = AffineScale()

    def _corrected(self, frame: pd.DataFrame) -> pd.DataFrame:
        selected = frame.loc[~(_is_missing(frame['score_missing']) | frame['score'].isna())].copy()
        nuisance = selected.loc[:, TRUE_FACET_COLUMNS].sum(axis=1).to_numpy(float)
        locations = np.array([self.category_locations[str(dimension)][int(round(float(score)))] for dimension, score in zip(selected['target_dimension'], selected['score'])], dtype=float)
        selected['_corrected'] = locations + nuisance
        return selected

    def _fit_category_locations(self, frame: pd.DataFrame) -> None:
        selected = frame.loc[~(_is_missing(frame['score_missing']) | frame['score'].isna())].copy()
        selected['score_category'] = np.clip(np.rint(selected['score'].to_numpy(float)).astype(int), 0, 4)
        selected['latent_without_nuisance'] = selected['theta_true'].to_numpy(float) - selected.loc[:, TRUE_FACET_COLUMNS].sum(axis=1).to_numpy(float)
        for dimension in DIMENSIONS:
            part = selected.loc[selected['target_dimension'].eq(dimension)]
            global_mean = float(part['latent_without_nuisance'].mean())
            means = part.groupby('score_category')['latent_without_nuisance'].mean()
            self.category_locations[dimension] = {category: float(means.get(category, global_mean)) for category in range(5)}

    def _raw_weekly(self, frame: pd.DataFrame, learner_ids: Sequence[str], fitting: bool) -> pd.DataFrame:
        selected = self._corrected(frame)
        stats = selected.groupby(['learner_id', 'target_dimension', 'turn'])['_corrected'].agg(['mean', 'count'])
        if fitting:
            means = stats.reset_index().groupby(['target_dimension', 'turn'])['mean'].mean()
            dimension_means = selected.groupby('target_dimension')['_corrected'].mean()
            self.prior = {(dimension, turn): float(means.get((dimension, turn), dimension_means[dimension])) for dimension in DIMENSIONS for turn in TURNS}
            prior_x = (np.asarray(TURNS, dtype=float) - TURN_CENTER) / TURN_SCALE
            prior_design = np.column_stack([np.ones(len(TURNS)), prior_x])
            for dimension in DIMENSIONS:
                self.trajectory_prior[dimension], *_ = np.linalg.lstsq(prior_design, np.array([self.prior[dimension, turn] for turn in TURNS], dtype=float), rcond=None)
        rows: list[dict[str, Any]] = []
        report_x = np.array([(WEEK_END_TURNS[week] - TURN_CENTER) / TURN_SCALE for week in WEEKS], dtype=float)
        for learner_id in map(str, learner_ids):
            for dimension in DIMENSIONS:
                precision = np.diag([DYNAMIC_PRIOR_STRENGTH, self.smooth_lambda])
                rhs = precision @ self.trajectory_prior[dimension]
                x_rows: list[float] = []
                targets: list[float] = []
                weights: list[float] = []
                for turn in TURNS:
                    key = (learner_id, dimension, turn)
                    if key in stats.index:
                        item = stats.loc[key]
                        x_rows.append((turn - TURN_CENTER) / TURN_SCALE)
                        targets.append(float(item['mean']))
                        weights.append(float(item['count']))
                if x_rows:
                    design = np.column_stack([np.ones(len(x_rows)), np.asarray(x_rows, dtype=float)])
                    weight_array = np.asarray(weights, dtype=float)
                    target_array = np.asarray(targets, dtype=float)
                    precision = precision + design.T @ (weight_array[:, None] * design)
                    rhs = rhs + design.T @ (weight_array * target_array)
                coefficients = np.linalg.solve(precision, rhs)
                values = coefficients[0] + coefficients[1] * report_x
                for week, value in zip(WEEKS, values):
                    rows.append({'learner_id': learner_id, 'target_dimension': dimension, 'week': week, 'raw_prediction': float(value)})
        return pd.DataFrame(rows)

    def fit(self, calibration: pd.DataFrame, truth: pd.DataFrame) -> 'OracleDynamic':
        self._fit_category_locations(calibration)
        learners = sorted(calibration['learner_id'].astype(str).unique())
        raw = self._raw_weekly(calibration, learners, True)
        self.scale.fit(raw, truth)
        return self

    def predict_weekly(self, frame: pd.DataFrame, learner_ids: Sequence[str]) -> pd.DataFrame:
        return self.scale.transform(self._raw_weekly(frame, learner_ids, False))

class FinalStateBaseline:
    STATE_WEIGHTS = np.array([0.25, 0.15, 0.15, 0.2, 0.1, 0.15], dtype=float)

    def __init__(self):
        self.method = 'final_resource_state'
        self.feature_columns = list(STATE_COLUMNS)
        self.scale = AffineScale()
        self.prior: dict[tuple[str, int], float] = {}

    def _weekly_state(self, frame: pd.DataFrame) -> pd.DataFrame:
        event_state = frame.drop_duplicates(['learner_id', 'event_id'])
        last = event_state.sort_values('turn').groupby(['learner_id', 'week'], as_index=False).tail(1)[['learner_id', 'week', *self.feature_columns]]
        last['state_rule_score'] = last[self.feature_columns].to_numpy(float) @ self.STATE_WEIGHTS
        return last

    def _raw_weekly(self, frame: pd.DataFrame, learner_ids: Sequence[str], *, fitting: bool) -> pd.DataFrame:
        states = self._weekly_state(frame).set_index(['learner_id', 'week'])
        if fitting:
            means = states.reset_index().groupby('week')['state_rule_score'].mean()
            overall = float(states['state_rule_score'].mean())
            self.prior = {(dimension, week): float(means.get(week, overall)) for dimension in DIMENSIONS for week in WEEKS}
        rows = []
        for learner_id in map(str, learner_ids):
            for dimension in DIMENSIONS:
                for week in WEEKS:
                    key = (learner_id, week)
                    value = float(states.loc[key, 'state_rule_score']) if key in states.index else self.prior[dimension, week]
                    rows.append({'learner_id': learner_id, 'target_dimension': dimension, 'week': week, 'raw_prediction': value})
        return pd.DataFrame(rows)

    def fit(self, calibration: pd.DataFrame, truth: pd.DataFrame) -> 'FinalStateBaseline':
        learners = sorted(calibration['learner_id'].astype(str).unique())
        raw = self._raw_weekly(calibration, learners, fitting=True)
        self.scale.fit(raw, truth)
        return self

    def predict_weekly(self, frame: pd.DataFrame, learner_ids: Sequence[str]) -> pd.DataFrame:
        return self.scale.transform(self._raw_weekly(frame, learner_ids, fitting=False))

def _prediction_metrics(prediction: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, label), method_frame in prediction.groupby(['method', 'method_label']):
        for estimand in ('week4', 'change'):
            pred_col = f'pred_{estimand}'
            truth_col = f'true_{estimand}'
            low_col = f'lower_{estimand}'
            high_col = f'upper_{estimand}'
            for dimension in (*DIMENSIONS, 'ALL'):
                part = method_frame if dimension == 'ALL' else method_frame.loc[method_frame['target_dimension'].eq(dimension)]
                error = part[pred_col].to_numpy(float) - part[truth_col].to_numpy(float)
                truth = part[truth_col].to_numpy(float)
                pred = part[pred_col].to_numpy(float)
                coverage = np.mean((truth >= part[low_col].to_numpy(float)) & (truth <= part[high_col].to_numpy(float)))
                rows.append({'method': method, 'method_label': label, 'estimand': estimand, 'dimension': dimension, 'n': int(len(part)), 'rmse': float(np.sqrt(np.mean(error ** 2))), 'mae': float(np.mean(np.abs(error))), 'bias': float(np.mean(error)), 'spearman': _safe_spearman(pred, truth), 'coverage_95': float(coverage), 'mean_interval_width': float(np.mean(part[high_col].to_numpy(float) - part[low_col].to_numpy(float)))})
    return pd.DataFrame(rows)

def _calibrate_intervals(validation: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = test.copy()
    record: dict[str, Any] = {}
    for method in sorted(validation['method'].unique()):
        record[method] = {}
        for dimension in DIMENSIONS:
            val_part = validation.loc[validation['method'].eq(method) & validation['target_dimension'].eq(dimension)]
            test_mask = result['method'].eq(method) & result['target_dimension'].eq(dimension)
            for estimand in ('week4', 'change'):
                residual = val_part[f'true_{estimand}'].to_numpy(float) - val_part[f'pred_{estimand}'].to_numpy(float)
                low, high = np.quantile(residual, [0.025, 0.975], method='linear')
                result.loc[test_mask, f'lower_{estimand}'] = result.loc[test_mask, f'pred_{estimand}'] + low
                result.loc[test_mask, f'upper_{estimand}'] = result.loc[test_mask, f'pred_{estimand}'] + high
                record[method].setdefault(dimension, {})[estimand] = {'residual_q025': float(low), 'residual_q975': float(high), 'n_validation': int(len(residual))}
    return (result, record)

def _paired_bootstrap(test_prediction: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 41)
    full = test_prediction.loc[test_prediction['method'].eq(FULL_METHOD)]
    rows = []
    for estimand in ('week4', 'change'):
        pred_col = f'pred_{estimand}'
        truth_col = f'true_{estimand}'
        full_error = full.assign(squared_error=lambda x: (x[pred_col] - x[truth_col]) ** 2, absolute_error=lambda x: (x[pred_col] - x[truth_col]).abs())[['learner_id', 'target_dimension', 'squared_error', 'absolute_error']]
        for method in METHOD_ORDER:
            if method == FULL_METHOD:
                continue
            other = test_prediction.loc[test_prediction['method'].eq(method)].assign(squared_error=lambda x: (x[pred_col] - x[truth_col]) ** 2, absolute_error=lambda x: (x[pred_col] - x[truth_col]).abs())
            joined = other.merge(full_error, on=['learner_id', 'target_dimension'], suffixes=('_baseline', '_full'), validate='one_to_one')
            learner_values = joined.groupby('learner_id').agg(mse_baseline=('squared_error_baseline', 'mean'), mse_full=('squared_error_full', 'mean'), mae_baseline=('absolute_error_baseline', 'mean'), mae_full=('absolute_error_full', 'mean'))
            learners = learner_values.index.to_numpy()
            boot_rmse = np.empty(N_BOOTSTRAP)
            boot_mae = np.empty(N_BOOTSTRAP)
            array = learner_values.to_numpy(float)
            for iteration in range(N_BOOTSTRAP):
                selected = rng.integers(0, len(learners), len(learners))
                sample = array[selected]
                boot_rmse[iteration] = math.sqrt(sample[:, 0].mean()) - math.sqrt(sample[:, 1].mean())
                boot_mae[iteration] = sample[:, 2].mean() - sample[:, 3].mean()
            point_rmse = math.sqrt(learner_values['mse_baseline'].mean()) - math.sqrt(learner_values['mse_full'].mean())
            point_mae = learner_values['mae_baseline'].mean() - learner_values['mae_full'].mean()
            rows.append({'baseline_method': method, 'baseline_label': METHOD_LABELS[method], 'reference_method': FULL_METHOD, 'estimand': estimand, 'n_learners': int(len(learners)), 'bootstrap_replicates': N_BOOTSTRAP, 'delta_rmse_baseline_minus_full': float(point_rmse), 'delta_rmse_ci_low': float(np.quantile(boot_rmse, 0.025)), 'delta_rmse_ci_high': float(np.quantile(boot_rmse, 0.975)), 'delta_mae_baseline_minus_full': float(point_mae), 'delta_mae_ci_low': float(np.quantile(boot_mae, 0.025)), 'delta_mae_ci_high': float(np.quantile(boot_mae, 0.975))})
    return pd.DataFrame(rows)

def _fairness_metrics(prediction: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, label), method_frame in prediction.groupby(['method', 'method_label']):
        for estimand in ('week4', 'change'):
            group_rows = []
            for group, part in method_frame.groupby('group'):
                error = part[f'pred_{estimand}'].to_numpy(float) - part[f'true_{estimand}'].to_numpy(float)
                coverage = np.mean((part[f'true_{estimand}'] >= part[f'lower_{estimand}']) & (part[f'true_{estimand}'] <= part[f'upper_{estimand}']))
                row = {'method': method, 'method_label': label, 'estimand': estimand, 'group': group, 'n_rows': int(len(part)), 'rmse': float(np.sqrt(np.mean(error ** 2))), 'mae': float(np.mean(np.abs(error))), 'bias': float(np.mean(error)), 'coverage_95': float(coverage)}
                rows.append(row)
                group_rows.append(row)
            if group_rows:
                rows.append({'method': method, 'method_label': label, 'estimand': estimand, 'group': 'MAX_MIN_GAP', 'n_rows': sum((item['n_rows'] for item in group_rows)), 'rmse': max((item['rmse'] for item in group_rows)) - min((item['rmse'] for item in group_rows)), 'mae': max((item['mae'] for item in group_rows)) - min((item['mae'] for item in group_rows)), 'bias': max((item['bias'] for item in group_rows)) - min((item['bias'] for item in group_rows)), 'coverage_95': max((item['coverage_95'] for item in group_rows)) - min((item['coverage_95'] for item in group_rows))})
    return pd.DataFrame(rows)

def _external_validity(predictions: Mapping[str, pd.DataFrame], observations: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    criterion = {}
    for split, frame in observations.items():
        criterion[split] = frame.groupby('learner_id', as_index=False)['external_criterion_observed'].first().rename(columns={'external_criterion_observed': 'criterion'})
    rows = []
    base_predictions: dict[str, np.ndarray] = {}
    for method in METHOD_ORDER:
        frames = {}
        for split in ('calibration', 'validation', 'test'):
            part = predictions[split].loc[predictions[split]['method'].eq(method)]
            wide = part.pivot(index='learner_id', columns='target_dimension', values='pred_week4')
            first = part.groupby('learner_id')['pred_week1'].mean().rename('baseline_proxy')
            data = wide.join(first).reset_index().merge(criterion[split], on='learner_id')
            frames[split] = data
        features = [*DIMENSIONS, 'baseline_proxy']
        best = None
        for alpha in (0.1, 1.0, 10.0):
            model = Ridge(alpha=alpha).fit(frames['calibration'][features], frames['calibration']['criterion'])
            val_pred = model.predict(frames['validation'][features])
            val_rmse = float(np.sqrt(np.mean((val_pred - frames['validation']['criterion']) ** 2)))
            if best is None or val_rmse < best[0]:
                best = (val_rmse, alpha, model)
        assert best is not None
        test_pred = best[2].predict(frames['test'][features])
        truth = frames['test']['criterion'].to_numpy(float)
        base_model = Ridge(alpha=1.0).fit(frames['calibration'][['baseline_proxy']], frames['calibration']['criterion'])
        base_pred = base_model.predict(frames['test'][['baseline_proxy']])
        base_predictions[method] = base_pred
        rows.append({'method': method, 'method_label': METHOD_LABELS[method], 'n_test': int(len(truth)), 'ridge_alpha': float(best[1]), 'external_rmse': float(np.sqrt(np.mean((test_pred - truth) ** 2))), 'external_mae': float(np.mean(np.abs(test_pred - truth))), 'external_r2': float(r2_score(truth, test_pred)), 'base_r2': float(r2_score(truth, base_pred)), 'incremental_r2_over_own_week1': float(r2_score(truth, test_pred) - r2_score(truth, base_pred)), 'external_spearman': _safe_spearman(test_pred, truth)})
    result = pd.DataFrame(rows)
    best_simple = result.loc[result['method'].isin(['ecd_equal_weight', 'final_resource_state', 'raw_rubric_mean', 'ctt_anchor_equating']), 'external_r2'].max()
    result['incremental_r2_over_best_simple'] = result['external_r2'] - float(best_simple)
    return result

def _parameter_recovery(full_model: Any, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    component_map = {'blueprint': 'blueprint_difficulty_true', 'surface': 'surface_effect_true', 'state': 'state_effect_true', 'path': 'personalized_path_difficulty_true', 'rater_time': 'rater_severity_true', 'version': 'version_effect_true'}
    for dimension in DIMENSIONS:
        part = test.loc[test['target_dimension'].eq(dimension)].copy()
        model = full_model.models[dimension]
        slope = full_model.scale.parameters[dimension][1]
        for component, truth_column in component_map.items():
            direction = float(getattr(full_model, 'facet_direction', -1.0))
            estimate = direction * slope * model.component_effect(part, component)
            truth = part[truth_column].to_numpy(float)
            estimate = estimate - float(np.mean(estimate))
            truth = truth - float(np.mean(truth))
            error = estimate - truth
            rank_recovery = float('nan') if component in {'surface', 'version'} else _safe_spearman(estimate, truth)
            rows.append({'component': component, 'dimension': dimension, 'identification': 'within_dimension_mean_zero', 'n': int(len(part)), 'bias': float(np.mean(error)), 'mae': float(np.mean(np.abs(error))), 'rmse': float(np.sqrt(np.mean(error ** 2))), 'spearman': rank_recovery})
    return pd.DataFrame(rows)

def _dif_diagnostics(full_model: Any, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    surface_rows = []
    group_rows = []
    anchors = test.loc[test['event_kind'].astype(str).str.contains('anchor') & test['score'].notna() & ~_is_missing(test['score_missing'])].copy()
    for dimension in DIMENSIONS:
        part = anchors.loc[anchors['target_dimension'].eq(dimension)].copy()
        if part.empty:
            continue
        model = full_model.models[dimension]
        slope = full_model.scale.parameters[dimension][1]
        direction = float(getattr(full_model, 'facet_direction', -1.0))
        estimated_surface = direction * slope * model.component_effect(part, 'surface')
        part['estimated_surface_effect'] = estimated_surface
        for (blueprint, form_id, form_kind), cell in part.groupby(['blueprint_id', 'form_id', 'form_kind']):
            truth = cell['surface_effect_true'].to_numpy(float)
            estimate = cell['estimated_surface_effect'].to_numpy(float)
            surface_rows.append({'dimension': dimension, 'blueprint_id': blueprint, 'form_id': form_id, 'form_kind': form_kind, 'n': int(len(cell)), 'true_surface_effect': float(np.mean(truth)), 'estimated_surface_effect': float(np.mean(estimate)), 'absolute_recovery_error': float(abs(np.mean(estimate) - np.mean(truth)))})
        x = model.encoder.transform(part)
        global_pred = np.asarray(x @ model.global_coef).ravel()
        part['adjusted_residual'] = _ordinal_signal(part['score']) - global_pred
        for (blueprint, form_id), cell in part.groupby(['blueprint_id', 'form_id']):
            means = cell.groupby('group')['adjusted_residual'].agg(['mean', 'count'])
            if len(means) < 2:
                continue
            levels = sorted(means.index.astype(str))
            reference, comparison = (levels[0], levels[-1])
            group_rows.append({'dimension': dimension, 'blueprint_id': blueprint, 'form_id': form_id, 'reference_group': reference, 'comparison_group': comparison, 'n_reference': int(means.loc[reference, 'count']), 'n_comparison': int(means.loc[comparison, 'count']), 'residual_dif_contrast': float(means.loc[comparison, 'mean'] - means.loc[reference, 'mean'])})
    return (pd.DataFrame(surface_rows), pd.DataFrame(group_rows))

def _merge_truth(prediction: pd.DataFrame, truth_end: pd.DataFrame) -> pd.DataFrame:
    return prediction.merge(truth_end[['split', 'learner_id', 'group', 'target_dimension', 'true_week1', 'true_week4', 'true_change']], on=['split', 'learner_id', 'target_dimension'], how='left', validate='many_to_one')

def _validate_prediction_panel(prediction: pd.DataFrame, *, split: str, n_learners: int) -> None:
    expected_per_method = n_learners * len(DIMENSIONS)
    counts = prediction.groupby('method').size().to_dict()
    expected = {method: expected_per_method for method in METHOD_ORDER}
    if counts != expected:
        raise ValueError(f'{split} endpoint panel counts differ from {expected_per_method} per method: {counts}')
    key = ['method', 'learner_id', 'target_dimension']
    if prediction.duplicated(key).any():
        raise ValueError(f'{split} endpoint predictions contain duplicate method×learner×dimension rows')

def _score_for_tuning(prediction: pd.DataFrame, truth_end: pd.DataFrame) -> float:
    merged = _merge_truth(prediction, truth_end)
    week4 = np.sqrt(np.mean((merged['pred_week4'] - merged['true_week4']) ** 2))
    change = np.sqrt(np.mean((merged['pred_change'] - merged['true_change']) ** 2))
    return float(week4 + change)

def _stress_frame(test: pd.DataFrame, scenario: str) -> pd.DataFrame:
    frame = test.copy()
    if scenario in {'sparse_anchor_support', 'joint_missing_drift'}:
        is_anchor = frame['event_kind'].astype(str).str.contains('anchor')
        keep = np.array([_stable_uniform('anchor-keep', learner, event, dimension) < 0.45 for learner, event, dimension in zip(frame['learner_id'], frame['event_id'], frame['target_dimension'])])
        frame = frame.loc[~is_anchor | keep].copy()
    if scenario in {'missing_20_mar', 'joint_missing_drift'}:
        drop = np.array([_stable_uniform('missing', learner, event, dimension) < 0.2 for learner, event, dimension in zip(frame['learner_id'], frame['event_id'], frame['target_dimension'])])
        frame = frame.loc[~drop].copy()
    if scenario in {'unmodeled_surface_drift', 'joint_missing_drift'}:
        drift = frame['form_kind'].astype(str).eq('surface') & frame['primary_version_id'].astype(str).eq(sorted(frame['primary_version_id'].astype(str).unique())[-1])
        selected = drift & frame['score'].notna() & ~_is_missing(frame['score_missing']) & np.array([_stable_uniform('drift', learner, event, dimension) < 0.55 for learner, event, dimension in zip(frame['learner_id'], frame['event_id'], frame['target_dimension'])])
        frame.loc[selected, 'score'] = np.minimum(4, frame.loc[selected, 'score'].astype(int) + 1)
    return frame

def _make_plots(results_dir: Path, metrics: pd.DataFrame, bootstrap: pd.DataFrame, prediction: pd.DataFrame, parameter: pd.DataFrame, ablation: pd.DataFrame) -> None:
    plot_method_labels = {'ecd_equal_weight': 'ECD equal-weight', 'final_resource_state': 'Final resource state', 'raw_rubric_mean': 'Raw rubric mean', 'ctt_anchor_equating': 'CTT anchor equating', 'static_multifacet': 'Static multifacet', 'no_anchor_dynamic': 'Dynamic, no anchor', 'anchored_dynamic_simple': 'Anchored dynamic, simple', FULL_METHOD: 'Full common-anchor dynamic multifacet', 'oracle': 'Oracle (simulation only)'}
    colors = ['#64748b' if method != FULL_METHOD else '#0f766e' for method in METHOD_ORDER]
    overall = metrics.loc[metrics['dimension'].eq('ALL')].copy()
    pivot = overall.pivot(index='method', columns='estimand', values='rmse').reindex(METHOD_ORDER)
    x = np.arange(len(pivot))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.bar(x - width / 2, pivot['week4'], width, label='Week 4', color=colors)
    ax.bar(x + width / 2, pivot['change'], width, label='Week 1→4', color='#94a3b8', alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels([plot_method_labels[m] for m in pivot.index], rotation=28, ha='right')
    ax.set_ylabel('RMSE (latent SD)')
    ax.set_title('Frozen test trajectories: endpoint and change recovery')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(results_dir / 'measurement_rmse_comparison.png', dpi=180)
    plt.close(fig)
    forest = bootstrap.loc[bootstrap['estimand'].eq('week4')].copy()
    forest = forest.sort_values('delta_rmse_baseline_minus_full')
    y = np.arange(len(forest))
    point = forest['delta_rmse_baseline_minus_full'].to_numpy(float)
    low = forest['delta_rmse_ci_low'].to_numpy(float)
    high = forest['delta_rmse_ci_high'].to_numpy(float)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(point, y, xerr=[point - low, high - point], fmt='o', color='#0f766e', capsize=3)
    ax.axvline(0, color='#334155', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(forest['baseline_method'].map(plot_method_labels))
    ax.set_xlabel('ΔRMSE = baseline − full (positive favors full)')
    ax.set_title('Learner-clustered paired bootstrap, Week 4')
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(results_dir / 'measurement_bootstrap_forest.png', dpi=180)
    plt.close(fig)
    full = prediction.loc[prediction['method'].eq(FULL_METHOD)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, estimand in zip(axes, ('week4', 'change')):
        ax.scatter(full[f'true_{estimand}'], full[f'pred_{estimand}'], s=9, alpha=0.28, color='#0f766e')
        limits = [min(full[f'true_{estimand}'].min(), full[f'pred_{estimand}'].min()), max(full[f'true_{estimand}'].max(), full[f'pred_{estimand}'].max())]
        ax.plot(limits, limits, '--', color='#334155', linewidth=1)
        ax.set_xlabel('True latent ability')
        ax.set_ylabel('Predicted latent ability')
        ax.set_title('Week 4' if estimand == 'week4' else 'Week 1→4 change')
    fig.suptitle('Full common-anchor dynamic multifacet approximation')
    fig.tight_layout()
    fig.savefig(results_dir / 'measurement_full_predicted_vs_true.png', dpi=180)
    plt.close(fig)
    parameter_summary = parameter.groupby('component', as_index=False)['rmse'].mean().sort_values('rmse')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(parameter_summary['component'], parameter_summary['rmse'], color='#2563eb')
    axes[0].set_xlabel('RMSE')
    axes[0].set_title('Facet contribution recovery')
    ab = ablation.loc[ablation['estimand'].eq('week4') & ablation['dimension'].eq('ALL')]
    ab = ab.sort_values('rmse')
    ablation_plot_labels = {'ablation_no_blueprint': 'Remove blueprint facet', 'ablation_no_surface': 'Remove surface-form facet', 'ablation_no_state': 'Remove state covariates', 'ablation_no_rater_drift': 'Remove rater-time drift', 'ablation_no_version': 'Remove version facet', 'no_anchor_dynamic': 'Remove common anchors', 'static_multifacet': 'Remove dynamic ability'}
    axes[1].barh(ab['method'].map(ablation_plot_labels), ab['rmse'], color='#a855f7')
    axes[1].set_xlabel('Week 4 RMSE')
    axes[1].set_title('Predeclared ablations')
    fig.tight_layout()
    fig.savefig(results_dir / 'measurement_parameter_and_ablation.png', dpi=180)
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description='Compare nine measurement methods on frozen synthetic trajectories.')
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    parser.add_argument('--predictions-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'outputs' / 'predictions')
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'outputs' / 'results')
    args = parser.parse_args()
    args.predictions_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    observation_path = args.data_dir / 'trajectory_observations.csv'
    ability_path = args.data_dir / 'ability_truth_long.csv'
    ratings_paths = _ratings_input_paths(args.data_dir)
    facet_truth_path = args.data_dir / 'facet_truth.csv'
    for required_path in (observation_path, *ratings_paths, facet_truth_path):
        if not required_path.exists():
            raise FileNotFoundError(required_path)
    observations = pd.read_csv(observation_path)
    _validate_input(observations)
    observations['learner_id'] = observations['learner_id'].astype(str)
    observations['week'] = observations['week'].astype(int)
    observations['turn'] = observations['turn'].astype(int)
    dataset_versions = observations['dataset_version'].astype(str).unique().tolist()
    if len(dataset_versions) != 1:
        raise ValueError(f'Expected one dataset_version, found {dataset_versions}')
    ratings = pd.concat([pd.read_csv(path) for path in ratings_paths], ignore_index=True)
    facet_truth = pd.read_csv(facet_truth_path)
    multirater_observations = _build_multirater_observations(observations, ratings, facet_truth)
    weekly_truth = _normalize_ability_truth(ability_path, observations)
    weekly_truth['learner_id'] = weekly_truth['learner_id'].astype(str)
    truth_end = _endpoint_truth(weekly_truth)
    obs = {split: observations.loc[observations['split'].eq(split)].copy() for split in ('calibration', 'validation', 'test')}
    multirater_obs = {split: multirater_observations.loc[multirater_observations['split'].eq(split)].copy() for split in obs}
    truth = {split: weekly_truth.loc[weekly_truth['split'].eq(split)].copy() for split in obs}
    learner_ids = {split: sorted(frame['learner_id'].astype(str).unique()) for split, frame in obs.items()}
    tuning_rows = []
    full_candidates: list[tuple[float, CalibratedMultifacetModel, float, float, float]] = []
    for ridge_alpha in RIDGE_ALPHA_GRID:
        for anchor_weight in ANCHOR_WEIGHT_GRID:
            for smooth in SMOOTHING_GRID:
                candidate = CalibratedMultifacetModel(FULL_METHOD, spec=FeatureSpec(), dynamic=True, smooth_lambda=smooth, ridge_alpha=ridge_alpha, anchor_weight=anchor_weight).fit(multirater_obs['calibration'], truth['calibration'])
                val_weekly = candidate.predict_weekly(multirater_obs['validation'], learner_ids['validation'])
                val_end = _weekly_to_endpoints(val_weekly, FULL_METHOD, 'validation')
                score = _score_for_tuning(val_end, truth_end)
                tuning_rows.append({'method': FULL_METHOD, 'ridge_alpha': ridge_alpha, 'anchor_weight': anchor_weight, 'smooth_lambda': smooth, 'validation_objective': score})
                full_candidates.append((score, candidate, ridge_alpha, anchor_weight, smooth))
    full_candidates.sort(key=lambda item: item[0])
    _, full_model, selected_ridge_alpha, selected_anchor_weight, selected_smooth = full_candidates[0]
    models: dict[str, Any] = {}
    models[FULL_METHOD] = full_model
    models['static_multifacet'] = CalibratedMultifacetModel('static_multifacet', spec=FeatureSpec(), dynamic=False, smooth_lambda=selected_smooth, ridge_alpha=selected_ridge_alpha, anchor_weight=selected_anchor_weight).fit(multirater_obs['calibration'], truth['calibration'])
    models['no_anchor_dynamic'] = CalibratedMultifacetModel('no_anchor_dynamic', spec=FeatureSpec(), dynamic=True, include_anchor=False, smooth_lambda=selected_smooth, ridge_alpha=selected_ridge_alpha, anchor_weight=selected_anchor_weight).fit(multirater_obs['calibration'], truth['calibration'])
    models['anchored_dynamic_simple'] = CalibratedMultifacetModel('anchored_dynamic_simple', spec=FeatureSpec(False, False, False, False, False, False), dynamic=True, smooth_lambda=selected_smooth, ridge_alpha=selected_ridge_alpha, anchor_weight=selected_anchor_weight).fit(multirater_obs['calibration'], truth['calibration'])
    models['ecd_equal_weight'] = AggregateBaseline('ecd_equal_weight', 'rule_score').fit(obs['calibration'], truth['calibration'])
    models['raw_rubric_mean'] = AggregateBaseline('raw_rubric_mean', 'raw_score').fit(obs['calibration'], truth['calibration'])
    models['ctt_anchor_equating'] = AggregateBaseline('ctt_anchor_equating', 'score', anchors_only=True, ctt=True).fit(obs['calibration'], truth['calibration'])
    models['oracle'] = OracleDynamic(selected_smooth).fit(multirater_obs['calibration'], truth['calibration'])
    final_state = FinalStateBaseline().fit(obs['calibration'], truth['calibration'])
    models['final_resource_state'] = final_state
    all_predictions: dict[str, pd.DataFrame] = {}
    for split in ('calibration', 'validation', 'test'):
        pieces = []
        for method in METHOD_ORDER:
            model = models[method]
            method_frame = multirater_obs[split] if method in MULTIRATER_METHODS else obs[split]
            weekly = model.predict_weekly(method_frame, learner_ids[split])
            endpoint = _weekly_to_endpoints(weekly, method, split)
            pieces.append(endpoint)
        all_predictions[split] = _merge_truth(pd.concat(pieces, ignore_index=True), truth_end)
        _validate_prediction_panel(all_predictions[split], split=split, n_learners=len(learner_ids[split]))
    test_with_intervals, interval_config = _calibrate_intervals(all_predictions['validation'], all_predictions['test'])
    all_predictions['test'] = test_with_intervals
    prediction_path = args.predictions_dir / 'measurement_predictions.csv'
    test_with_intervals.sort_values(['method', 'learner_id', 'target_dimension']).to_csv(prediction_path, index=False)
    pd.concat([all_predictions['calibration'], all_predictions['validation']], ignore_index=True).to_csv(args.predictions_dir / 'measurement_calibration_validation_predictions.csv', index=False)
    _write_json(args.predictions_dir / 'measurement_interval_calibration.json', interval_config)
    metrics = _prediction_metrics(test_with_intervals)
    metrics.to_csv(args.results_dir / 'measurement_metrics.csv', index=False)
    bootstrap = _paired_bootstrap(test_with_intervals)
    bootstrap.to_csv(args.results_dir / 'measurement_paired_bootstrap.csv', index=False)
    fairness = _fairness_metrics(test_with_intervals)
    fairness.to_csv(args.results_dir / 'measurement_group_fairness.csv', index=False)
    external = _external_validity(all_predictions, obs)
    external.to_csv(args.results_dir / 'measurement_external_validity.csv', index=False)
    parameter = _parameter_recovery(full_model, multirater_obs['test'])
    parameter.to_csv(args.results_dir / 'measurement_parameter_recovery.csv', index=False)
    surface_dif, group_dif = _dif_diagnostics(full_model, multirater_obs['test'])
    surface_dif.to_csv(args.results_dir / 'measurement_surface_form_recovery.csv', index=False)
    group_dif.to_csv(args.results_dir / 'measurement_anchor_group_dif.csv', index=False)
    ablation_specs = {'ablation_no_blueprint': FeatureSpec(False, True, True, True, True), 'ablation_no_surface': FeatureSpec(True, False, True, True, True), 'ablation_no_state': FeatureSpec(True, True, False, True, True), 'ablation_no_rater_drift': FeatureSpec(True, True, True, False, True), 'ablation_no_version': FeatureSpec(True, True, True, True, False, True)}
    ablation_predictions = []
    ablation_models: dict[str, CalibratedMultifacetModel] = {}
    for method, spec in ablation_specs.items():
        model = CalibratedMultifacetModel(method, spec=spec, dynamic=True, smooth_lambda=selected_smooth, ridge_alpha=selected_ridge_alpha, anchor_weight=selected_anchor_weight).fit(multirater_obs['calibration'], truth['calibration'])
        ablation_models[method] = model
        val = _weekly_to_endpoints(model.predict_weekly(multirater_obs['validation'], learner_ids['validation']), method, 'validation')
        val['method_label'] = method.replace('ablation_', 'Remove ')
        test_pred = _weekly_to_endpoints(model.predict_weekly(multirater_obs['test'], learner_ids['test']), method, 'test')
        test_pred['method_label'] = method.replace('ablation_', 'Remove ')
        val = _merge_truth(val, truth_end)
        test_pred = _merge_truth(test_pred, truth_end)
        test_pred, _unused = _calibrate_intervals(val, test_pred)
        ablation_predictions.append(test_pred)
    structural = test_with_intervals.loc[test_with_intervals['method'].isin(['no_anchor_dynamic', 'static_multifacet'])].copy()
    structural['method_label'] = structural['method'].map({'no_anchor_dynamic': 'Remove common anchors', 'static_multifacet': 'Remove dynamic ability'})
    ablation_all = pd.concat([*ablation_predictions, structural], ignore_index=True)
    ablation_all.to_csv(args.predictions_dir / 'measurement_ablation_predictions.csv', index=False)
    ablation_metrics = _prediction_metrics(ablation_all)
    ablation_metrics.to_csv(args.results_dir / 'measurement_ablation_metrics.csv', index=False)
    stress_rows = []
    stress_methods = {key: models[key] for key in ('raw_rubric_mean', 'ctt_anchor_equating', 'no_anchor_dynamic', 'anchored_dynamic_simple', FULL_METHOD, 'oracle')}
    for scenario in ('nominal', 'sparse_anchor_support', 'missing_20_mar', 'unmodeled_surface_drift', 'joint_missing_drift'):
        primary_frame = obs['test'] if scenario == 'nominal' else _stress_frame(obs['test'], scenario)
        multirater_frame = multirater_obs['test'] if scenario == 'nominal' else _stress_frame(multirater_obs['test'], scenario)
        pieces = []
        for method, model in stress_methods.items():
            method_frame = multirater_frame if method in MULTIRATER_METHODS else primary_frame
            weekly = model.predict_weekly(method_frame, learner_ids['test'])
            endpoint = _weekly_to_endpoints(weekly, method, 'test')
            pieces.append(endpoint)
        stress_prediction = _merge_truth(pd.concat(pieces, ignore_index=True), truth_end)
        stress_prediction, _ = _calibrate_intervals(all_predictions['validation'], stress_prediction)
        stress_metric = _prediction_metrics(stress_prediction)
        stress_metric.insert(0, 'scenario', scenario)
        stress_metric.insert(1, 'remaining_primary_rows', len(primary_frame))
        stress_metric.insert(2, 'remaining_multirater_rows', len(multirater_frame))
        stress_rows.append(stress_metric)
    stress = pd.concat(stress_rows, ignore_index=True)
    stress.to_csv(args.results_dir / 'measurement_stress_boundaries.csv', index=False)
    tuning = pd.DataFrame(tuning_rows)
    tuning.to_csv(args.results_dir / 'measurement_validation_tuning.csv', index=False)
    config = {'analysis_version': 'measurement-validation-calibrated-multirater-linear-turn-map-v5', 'dataset_version': dataset_versions[0], 'seed': SEED, 'estimands': ['week4_end_five_dimensional_ability', 'week1_to_week4_change'], 'latent_state_model': {'type': 'exact-turn linear level-and-slope trajectory', 'turns': list(TURNS), 'reported_week_end_turns': WEEK_END_TURNS, 'level_prior_strength': DYNAMIC_PRIOR_STRENGTH, 'slope_penalty_grid': list(SMOOTHING_GRID)}, 'ordinal_approximation': {'type': 'penalized Gaussian/MAP approximation to five-category PCM signal', 'category_locations': [-1.55, -0.7, 0.0, 0.7, 1.55], 'not_stan_fit': True}, 'split_counts': {key: len(value) for key, value in learner_ids.items()}, 'multirater_rows_by_split': {key: len(value) for key, value in multirater_obs.items()}, 'smoothing_grid': list(SMOOTHING_GRID), 'ridge_alpha_grid': list(RIDGE_ALPHA_GRID), 'anchor_weight_grid': list(ANCHOR_WEIGHT_GRID), 'selected_smoothing': selected_smooth, 'selected_ridge_alpha': selected_ridge_alpha, 'selected_anchor_weight': selected_anchor_weight, 'ridge_level': RIDGE_LEVEL, 'ridge_global': RIDGE_GLOBAL, 'anchor_weight': ANCHOR_WEIGHT, 'final_state_rule_weights': dict(zip(STATE_COLUMNS, FinalStateBaseline.STATE_WEIGHTS.tolist())), 'bootstrap_replicates': N_BOOTSTRAP, 'methods': list(METHOD_ORDER), 'ablations': list(ablation_specs) + ['no_anchor_dynamic', 'static_multifacet'], 'stress_scenarios': sorted(stress['scenario'].unique()), 'input_files': {str(observation_path): _sha256(observation_path), str(ability_path): _sha256(ability_path) if ability_path.exists() else None, **{str(path): _sha256(path) for path in ratings_paths}, str(facet_truth_path): _sha256(facet_truth_path)}, 'software': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__, 'matplotlib': matplotlib.__version__}, 'fit_diagnostics': {dimension: model.fit_diagnostics for dimension, model in full_model.models.items()}, 'latent_scale_alignment': full_model.scale.parameters}
    _write_json(args.predictions_dir / 'measurement_run_config.json', config)
    _make_plots(args.results_dir, metrics, bootstrap, test_with_intervals, parameter, ablation_metrics)
    summary_metrics = metrics.loc[metrics['dimension'].eq('ALL')][['method', 'estimand', 'rmse', 'mae', 'bias', 'spearman', 'coverage_95']]
    summary = {'status': 'completed', 'test_learners': len(learner_ids['test']), 'selected_smoothing': selected_smooth, 'selected_ridge_alpha': selected_ridge_alpha, 'selected_anchor_weight': selected_anchor_weight, 'metrics': summary_metrics.to_dict(orient='records'), 'full_vs_baseline_bootstrap': bootstrap.to_dict(orient='records'), 'outputs': sorted([path.name for path in args.results_dir.glob('measurement_*')] + [path.name for path in args.predictions_dir.glob('measurement_*')])}
    _write_json(args.results_dir / 'measurement_summary.json', summary)
    print(json.dumps({'status': 'completed', 'selected_smoothing': selected_smooth, 'selected_ridge_alpha': selected_ridge_alpha, 'selected_anchor_weight': selected_anchor_weight}, ensure_ascii=False))
if __name__ == '__main__':
    main()
