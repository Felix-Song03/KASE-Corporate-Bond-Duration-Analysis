from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

LOCAL_PACKAGES = Path(__file__).resolve().parent / '.python_packages'
if LOCAL_PACKAGES.exists():
    sys.path.append(str(LOCAL_PACKAGES))

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test

ANALYSIS_CUTOFF = pd.Timestamp('2026-07-03')
THREE_YEARS_DAYS = 3 * 365.25
KAPLAN_MEIER_DISPLAY_MAX_DAYS = 500
COMMON_PREDICTORS = [
    'log_issue_volume',
    'coupon_rate',
    'circulation_term',
    'log_total_assets',
    'debt_to_asset',
    'foreign_currency',
    'esg_bonds',
    'financial_sector',
]
PRIVATE_PREDICTOR = 'private_placement'
VARIABLE_LABELS = {
    'log_issue_volume': 'Log (issue volume)',
    'coupon_rate': 'Coupon rate (%)',
    'circulation_term': 'Circulation term',
    'log_total_assets': 'Log (issuer size)',
    'debt_to_asset': 'Leverage',
    'foreign_currency': 'Foreign-currency denomination',
    'esg_bonds': 'ESG bond status',
    'financial_sector': 'Financial-sector issuer',
    'private_placement': 'Private placement',
}
SOURCE_COLUMNS = {
    'log_issue_volume': 'issue_volume_kzt',
    'coupon_rate': 'coupon_rate',
    'circulation_term': 'circulation_term',
    'log_total_assets': 'total_assets_tminus1',
    'debt_to_asset': 'debt_to_asset_tminus1',
    'foreign_currency': 'currency',
    'esg_bonds': 'esg_bonds',
    'financial_sector': 'financial_sector',
    'private_placement': 'placement_type',
}

@dataclass(frozen=True)
class ModelSpec:
    name: str
    include_private: bool
    drop_over_three_years: bool


MODEL_SPECS = [
    ModelSpec('Model (1) Baseline', True, False),
    ModelSpec('Model (2) Excluding private placements', False, False),
    ModelSpec('Model (3) Excluding durations over three years', True, True),
]


def clean_text(series: pd.Series) -> pd.Series:
    result = series.astype('string').str.strip()
    return result.replace({
        '': pd.NA,
        'N/A': pd.NA,
        'NA': pd.NA,
        'n/a': pd.NA,
        'None': pd.NA,
        'nan': pd.NA,
    })

def normalize_currency(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().upper()
    aliases = {
        'DOLLARS': 'USD',
        'DOLLAR': 'USD',
        'TENGE': 'KZT',
        'YUAN': 'CNY',
    }
    return aliases.get(text, text)

def winsorize(
    series: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> tuple[pd.Series, float, float]:
    p01 = float(series.quantile(lower))
    p99 = float(series.quantile(upper))
    return (series.clip(lower=p01, upper=p99), p01, p99)

def load_local_fx_tables(cny_fx_path: Path, usd_fx_path: Path) -> dict[str, pd.DataFrame]:
    specs = {
        'CNY': (Path(cny_fx_path), 'CNY/KZT'),
        'USD': (Path(usd_fx_path), 'USD/KZT'),
    }
    fx_tables: dict[str, pd.DataFrame] = {}
    for currency, (path, rate_column) in specs.items():
        if not path.exists():
            raise FileNotFoundError(f'Exchange-rate workbook not found for {currency}: {path}')
        workbook = pd.ExcelFile(path)
        if 'Sheet1' not in workbook.sheet_names:
            raise ValueError(f'{path} does not contain a Sheet1 worksheet.')
        table = pd.read_excel(path, sheet_name='Sheet1')
        required_columns = {'Date', rate_column}
        missing_columns = sorted(required_columns.difference(table.columns))
        if missing_columns:
            raise ValueError(f'{path} is missing columns: {missing_columns}')
        parsed = pd.DataFrame({
            'date': pd.to_datetime(table['Date'], errors='coerce').dt.normalize(),
            'kzt_per_unit': pd.to_numeric(table[rate_column], errors='coerce'),
        })
        parsed = parsed.dropna(subset=['date', 'kzt_per_unit'])
        parsed = parsed.loc[parsed['kzt_per_unit'].gt(0)].copy()
        if parsed.empty:
            raise ValueError(f'{path} does not contain usable exchange-rate data.')
        fx_tables[currency] = (
            parsed.sort_values('date')
            .drop_duplicates(subset=['date'], keep='last')
            .reset_index(drop=True)
        )
    return fx_tables

def get_local_fx_rate(
    currency: str,
    requested_date: pd.Timestamp,
    fx_tables: dict[str, pd.DataFrame],
) -> dict:
    requested_date = pd.Timestamp(requested_date).normalize()
    if currency == 'KZT':
        return {
            'requested_date': requested_date,
            'applied_date': requested_date,
            'kzt_per_unit': 1.0,
            'source': 'KZT base currency',
            'fallback_days': 0,
        }
    if currency not in fx_tables:
        raise ValueError(
            f'Local exchange-rate tables do not support {currency}; '
            'only KZT, CNY, and USD are supported.'
        )
    table = fx_tables[currency]
    # Use the admission-date exchange rate; if unavailable, use the most recent
    # available rate within the preceding 10 calendar days.
    for fallback_days in range(11):
        candidate = requested_date - timedelta(days=fallback_days)
        match = table.loc[table['date'].eq(candidate)]
        if not match.empty:
            row = match.iloc[-1]
            return {
                'requested_date': requested_date,
                'applied_date': pd.Timestamp(row['date']),
                'kzt_per_unit': float(row['kzt_per_unit']),
                'source': f'local Excel {currency}-KZT exchange rate.xlsx',
                'fallback_days': fallback_days,
            }
    min_date = table['date'].min().strftime('%Y-%m-%d')
    max_date = table['date'].max().strftime('%Y-%m-%d')
    raise ValueError(
        f'No {currency} exchange rate is available for '
        f'{requested_date:%Y-%m-%d} or the preceding 10 days. '
        f'Local coverage is {min_date} to {max_date}.'
    )

def choose_issue_fx_date(row: pd.Series) -> tuple[pd.Timestamp, str]:
    admission_date = pd.to_datetime(row.get('admission_date'), errors='coerce')
    if pd.isna(admission_date):
        raise ValueError(f"{row.get('trading_code')} is missing a trade-list admission date.")
    return (pd.Timestamp(admission_date).normalize(), 'trade_list_admission_date')

def convert_issue_volume_to_kzt(
    data: pd.DataFrame,
    fx_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    converted = data.copy()
    converted['issue_volume_original'] = converted['issue_volume']
    converted['issue_volume_kzt'] = np.nan
    converted['issue_volume_fx_rate'] = np.nan
    converted['issue_volume_fx_date'] = pd.NaT
    converted['issue_volume_fx_date_basis'] = pd.NA
    converted['issue_volume_fx_source'] = pd.NA
    rate_cache: dict[tuple[str, pd.Timestamp], dict] = {}
    for index, row in converted.iterrows():
        if not bool(row['pure_corporate_bond']):
            continue
        currency = normalize_currency(row['currency'])
        original = row['issue_volume']
        if currency is None or pd.isna(original):
            continue
        fx_date, date_basis = choose_issue_fx_date(row)
        key = (currency, fx_date)
        if key not in rate_cache:
            rate_cache[key] = get_local_fx_rate(currency, fx_date, fx_tables)
        rate = rate_cache[key]
        kzt_per_unit = float(rate['kzt_per_unit'])
        converted.loc[index, 'issue_volume_kzt'] = round(float(original) * kzt_per_unit, 2)
        converted.loc[index, 'issue_volume_fx_rate'] = kzt_per_unit
        converted.loc[index, 'issue_volume_fx_date'] = rate['applied_date']
        converted.loc[index, 'issue_volume_fx_date_basis'] = date_basis
        converted.loc[index, 'issue_volume_fx_source'] = rate['source']
    return converted

def load_and_clean(
    input_path: Path,
    sheet_name: str,
    cutoff: pd.Timestamp,
    fx_tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict]:
    data = pd.read_excel(input_path, sheet_name=sheet_name)
    original_rows = len(data)
    required_columns = {
        'company_name',
        'trading_code',
        'isin',
        'security_category',
        'issuer_type',
        'placement_type',
        'currency',
        'sector',
        'esg-bonds',
        'trade_list_admission_date',
        'trade_opening_date',
        'circulation_start_date',
        'issue_volume',
        'coupon_rate',
        'circulation_term',
        'total_assets_tminus1',
        'debt_to_asset_tminus1',
        'is_simplified_listing',
    }
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f'Input workbook is missing columns: {missing_columns}')
    dedupe_keys = ['company_name', 'trading_code', 'isin', 'trade_list_admission_date']
    data = data.drop_duplicates(subset=dedupe_keys, keep='first').copy()
    after_dedup = len(data)
    text_columns = [
        'company_name',
        'trading_code',
        'isin',
        'security_category',
        'issuer_type',
        'placement_type',
        'currency',
        'sector',
    ]
    for column in text_columns:
        data[column] = clean_text(data[column])
    numeric_columns = [
        'issue_volume',
        'coupon_rate',
        'circulation_term',
        'total_assets_tminus1',
        'debt_to_asset_tminus1',
        'esg-bonds',
        'is_simplified_listing',
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    date_mapping = {
        'admission_date': 'trade_list_admission_date',
        'opening_date': 'trade_opening_date',
        'circulation_start_date': 'circulation_start_date',
    }
    for target, source in date_mapping.items():
        data[target] = pd.to_datetime(data[source], errors='coerce')
    missing_admission = data['admission_date'].isna()
    admission_after_cutoff = data['admission_date'].gt(cutoff).fillna(False)
    data = data.loc[~missing_admission & ~admission_after_cutoff].copy()
    simplified_listing = pd.to_numeric(
        data['is_simplified_listing'],
        errors='coerce',
    ).eq(1)
    debt_security = data['security_category'].eq('debt securities')
    international_organisation = data['issuer_type'].eq(
        'Multilateral / international organisation'
    )
    # Retain eligible corporate debt securities and exclude international organisations
    # and simplified listings from the analytical sample.
    data['pure_corporate_bond'] = (
        debt_security
        & ~international_organisation
        & ~simplified_listing
    )
    data['foreign_currency'] = np.where(
        data['currency'].isna(),
        np.nan,
        data['currency'].ne('KZT').astype(float),
    )
    data['private_placement'] = np.where(
        data['placement_type'].isna(),
        np.nan,
        data['placement_type'].eq('Private placement').astype(float),
    )
    esg_raw = pd.to_numeric(data['esg-bonds'], errors='coerce')
    data['esg_bonds'] = np.where(esg_raw.isna(), np.nan, esg_raw.eq(1).astype(float))
    sector_clean = data['sector'].astype('string').str.strip().str.lower()
    data['financial_sector'] = sector_clean.map({
        'financial sector': 1.0,
        'real sector': 0.0,
    }).astype(float)
    data.loc[data['issue_volume'].le(0), 'issue_volume'] = np.nan
    data.loc[data['circulation_term'].le(0), 'circulation_term'] = np.nan
    data.loc[data['total_assets_tminus1'].le(0), 'total_assets_tminus1'] = np.nan
    data['issuer_cluster'] = pd.factorize(data['company_name'])[0]
    data = convert_issue_volume_to_kzt(data, fx_tables)
    pure = data.loc[data['pure_corporate_bond']].copy()
    international_exclusions = int((debt_security & international_organisation).sum())
    simplified_exclusions_after_international = int(
        (debt_security & simplified_listing & ~international_organisation).sum()
    )
    combined_exclusions = int(
        (debt_security & (international_organisation | simplified_listing)).sum()
    )
    cleaning_summary = {
        'original_rows': original_rows,
        'rows_after_deduplication': after_dedup,
        'missing_admission_dates_removed': int(missing_admission.sum()),
        'admissions_after_cutoff_removed': int(admission_after_cutoff.sum()),
        'initial_debt_securities': int(debt_security.sum()),
        'excluded_international_organisations': international_exclusions,
        'simplified_listing_total': int((debt_security & simplified_listing).sum()),
        'excluded_simplified_after_international': simplified_exclusions_after_international,
        'excluded_international_or_simplified': combined_exclusions,
        'eligible_corporate_bonds': int(len(pure)),
        'eligible_issuers': int(pure['company_name'].nunique()),
    }
    return (data, cleaning_summary)

def build_sample_construction_table(
    cleaning_summary: dict,
    baseline_audit: dict,
) -> pd.DataFrame:
    rows = [
        {
            'Sample construction step': 'Initial debt securities',
            'Number of bonds': cleaning_summary['initial_debt_securities'],
        },
        {
            'Sample construction step': 'International organisations',
            'Number of bonds': cleaning_summary['excluded_international_organisations'],
        },
        {
            'Sample construction step': 'Simplified listings',
            'Number of bonds': cleaning_summary['excluded_simplified_after_international'],
        },
        {
            'Sample construction step': 'Eligible corporate bonds',
            'Number of bonds': cleaning_summary['eligible_corporate_bonds'],
        },
        {
            'Sample construction step': 'Unresolved dates',
            'Number of bonds': baseline_audit['unresolved_date_removed_n'],
        },
        {
            'Sample construction step': 'Missing covariates',
            'Number of bonds': baseline_audit['covariate_missing_rows_removed_n'],
        },
        {
            'Sample construction step': 'Baseline sample',
            'Number of bonds': baseline_audit['complete_case_n'],
        },
        {
            'Sample construction step': 'Baseline issuers',
            'Number of bonds': baseline_audit['issuers'],
        },
    ]
    return pd.DataFrame(rows)

def prepare_model_sample(
    source: pd.DataFrame,
    spec: ModelSpec,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, dict, list[str]]:
    sample = source.loc[source['pure_corporate_bond']].copy()
    if not spec.include_private:
        sample = sample.loc[sample['private_placement'].eq(0)].copy()
    sample_before_date_policy = len(sample)
    is_private = sample['private_placement'].eq(1.0)

    # Define the event date using the recorded trade-opening date where available.
    # For private placements with no opening date, use the circulation start date as a proxy.
    # Eligible non-private bonds without an observed opening date are right-censored at the cutoff.
    actual_opening_valid = (
        sample['opening_date'].notna()
        & sample['opening_date'].le(cutoff)
        & sample['opening_date'].ge(sample['admission_date'])
    )
    opening_missing = sample['opening_date'].isna()
    private_valid_proxy = (
        is_private
        & opening_missing
        & sample['circulation_start_date'].notna()
        & sample['circulation_start_date'].ge(sample['admission_date'])
        & sample['circulation_start_date'].le(cutoff)
    )
    regular_right_censored = (
        ~is_private
        & ~actual_opening_valid
        & (opening_missing | sample['opening_date'].gt(cutoff).fillna(False))
    )
    sample['analysis_opening_date'] = pd.NaT
    sample.loc[actual_opening_valid, 'analysis_opening_date'] = sample.loc[
        actual_opening_valid,
        'opening_date',
    ]
    sample.loc[private_valid_proxy, 'analysis_opening_date'] = sample.loc[
        private_valid_proxy,
        'circulation_start_date',
    ]
    sample.loc[regular_right_censored, 'analysis_opening_date'] = cutoff
    sample['event'] = np.select(
        [actual_opening_valid | private_valid_proxy, regular_right_censored],
        [1, 0],
        default=np.nan,
    )
    sample['opening_date_source'] = np.select(
        [actual_opening_valid, private_valid_proxy, regular_right_censored],
        [
            'trade_opening_date',
            'private_circulation_start_date_proxy',
            'regular_right_censored_at_cutoff',
        ],
        default='unresolved_missing_or_invalid',
    )
    unresolved = sample['analysis_opening_date'].isna() | sample['event'].isna()
    private_proxy_used_n = int(private_valid_proxy.sum())
    regular_censored_n = int(regular_right_censored.sum())
    unresolved_n = int(unresolved.sum())
    sample = sample.loc[~unresolved].copy()
    sample['event'] = sample['event'].astype(int)
    sample['duration_days'] = (
        sample['analysis_opening_date'] - sample['admission_date']
    ).dt.total_seconds() / 86400
    same_day_n = int(sample['duration_days'].eq(0).sum())
    # Recode same-day observations from 0 to 1 day to ensure strictly positive durations.
    sample.loc[sample['duration_days'].eq(0), 'duration_days'] = 1.0
    sample_before_three_year_rule = len(sample)
    if spec.drop_over_three_years:
        sample = sample.loc[sample['duration_days'].le(THREE_YEARS_DAYS)].copy()
    over_three_removed = sample_before_three_year_rule - len(sample)
    predictors = list(COMMON_PREDICTORS)
    if spec.include_private:
        predictors.append(PRIVATE_PREDICTOR)
    required_raw = list(dict.fromkeys((SOURCE_COLUMNS[x] for x in predictors)))
    complete_mask = sample[required_raw].notna().all(axis=1)
    incomplete_n = int((~complete_mask).sum())
    model_data = sample.loc[complete_mask].copy()

    # Apply log transformations to issue volume and issuer size, then winsorise
    # continuous covariates at the 1st and 99th percentiles.
    model_data['log_issue_volume'] = np.log(model_data['issue_volume_kzt'])
    model_data['log_total_assets'] = np.log(model_data['total_assets_tminus1'])
    transform_map = {
        'log_issue_volume': 'log_issue_volume',
        'coupon_rate': 'coupon_rate',
        'circulation_term': 'circulation_term',
        'log_total_assets': 'log_total_assets',
        'debt_to_asset_tminus1': 'debt_to_asset',
    }
    transform_audit = {}
    for source_column, target_column in transform_map.items():
        raw_values = pd.to_numeric(model_data[source_column], errors='coerce')
        clipped, p01, p99 = winsorize(raw_values)
        model_data[target_column] = clipped
        transform_audit[target_column] = {
            'source_column': source_column,
            'n': int(raw_values.notna().sum()),
            'raw_min': float(raw_values.min()),
            'winsor_p01': p01,
            'winsor_p99': p99,
            'raw_max': float(raw_values.max()),
            'lower_tail_adjusted_n': int(raw_values.lt(p01).sum()),
            'upper_tail_adjusted_n': int(raw_values.gt(p99).sum()),
            'winsorised_min': float(clipped.min()),
            'winsorised_max': float(clipped.max()),
            'mean_after_winsor': float(clipped.mean()),
            'sd_after_winsor_ddof0': float(clipped.std(ddof=0)),
        }
    model_data['financial_sector'] = model_data['financial_sector'].astype(float)
    model_data['esg_bonds'] = model_data['esg_bonds'].astype(float)
    model_data['foreign_currency'] = model_data['foreign_currency'].astype(float)
    if 'private_placement' in model_data.columns:
        model_data['private_placement'] = model_data['private_placement'].astype(float)
    constant_predictors = [
        variable
        for variable in predictors
        if model_data[variable].nunique(dropna=True) <= 1
    ]
    if constant_predictors:
        predictors = [
            variable for variable in predictors if variable not in constant_predictors
        ]
    model_audit = {
        'model': spec.name,
        'include_private': spec.include_private,
        'drop_over_three_years': spec.drop_over_three_years,
        'sample_before_date_policy': sample_before_date_policy,
        'actual_trade_opening_event_n': int(
            (sample['opening_date_source'] == 'trade_opening_date').sum()
        ),
        'private_circulation_proxy_event_n': private_proxy_used_n,
        'regular_right_censored_n': regular_censored_n,
        'unresolved_date_removed_n': unresolved_n,
        'same_day_changed_to_one_n': same_day_n,
        'sample_before_three_year_rule': sample_before_three_year_rule,
        'over_three_years_removed_n': over_three_removed,
        'sample_before_complete_case': len(sample),
        'covariate_missing_rows_removed_n': incomplete_n,
        'complete_case_n': len(model_data),
        'events': int(model_data['event'].sum()),
        'censored': int((1 - model_data['event']).sum()),
        'issuers': int(model_data['company_name'].nunique()),
        'predictors': predictors,
        'constant_predictors_removed': constant_predictors,
        'transform_audit': transform_audit,
    }
    return (model_data, model_audit, predictors)

CHAPTER4_CONTINUOUS_VARIABLES = [
    'duration_days',
    'log_issue_volume',
    'coupon_rate',
    'circulation_term',
    'log_total_assets',
    'debt_to_asset',
]
CHAPTER4_DUMMY_VARIABLES = [
    'event',
    'foreign_currency',
    'esg_bonds',
    'financial_sector',
]
CHAPTER4_LABELS = {
    'duration_days': 'Observed duration (days)',
    'event': 'Trading commencement event',
    'log_issue_volume': 'Log (issue volume)',
    'coupon_rate': 'Coupon rate (%)',
    'circulation_term': 'Circulation term (years)',
    'log_total_assets': 'Log (issuer size)',
    'debt_to_asset': 'Leverage',
    'foreign_currency': 'Foreign-currency denomination',
    'esg_bonds': 'ESG bond status',
    'financial_sector': 'Financial-sector issuer',
    'private_placement': 'Private placement',
}
DURATION_BAND_LABELS = [
    '1-7 days',
    '8-30 days',
    '31-90 days',
    '91-365 days',
    '366 days-3 years',
    'Over 3 years',
]

def build_descriptive_statistics(
    model_data: pd.DataFrame,
    sample_name: str,
    include_private: bool,
) -> pd.DataFrame:
    continuous_variables = list(CHAPTER4_CONTINUOUS_VARIABLES)
    dummy_variables = list(CHAPTER4_DUMMY_VARIABLES)
    if include_private:
        dummy_variables.append('private_placement')
    rows: list[dict] = []
    for variable in continuous_variables:
        values = pd.to_numeric(model_data[variable], errors='coerce').dropna()
        rows.append({
            'sample': sample_name,
            'variable': CHAPTER4_LABELS[variable],
            'variable_type': 'Continuous',
            'N': int(values.size),
            'mean': float(values.mean()),
            'std_dev': float(values.std(ddof=1)),
            'skewness': float(values.skew()),
            'minimum': float(values.min()),
            'median': float(values.median()),
            'maximum': float(values.max()),
            'count_0': np.nan,
            'count_1': np.nan,
            'percent_1': np.nan,
        })
    for variable in dummy_variables:
        values = pd.to_numeric(model_data[variable], errors='coerce').dropna()
        count_zero = int(values.eq(0).sum())
        count_one = int(values.eq(1).sum())
        rows.append({
            'sample': sample_name,
            'variable': CHAPTER4_LABELS[variable],
            'variable_type': 'Dummy',
            'N': int(values.size),
            'mean': float(values.mean()),
            'std_dev': float(values.std(ddof=1)),
            'skewness': float(values.skew()),
            'minimum': float(values.min()),
            'median': float(values.median()),
            'maximum': float(values.max()),
            'count_0': count_zero,
            'count_1': count_one,
            'percent_1': 100.0 * count_one / values.size if values.size else np.nan,
        })
    return pd.DataFrame(rows)

def build_winsorisation_audit(model_name: str, transform_audit: dict) -> pd.DataFrame:
    rows = []
    for variable, audit in transform_audit.items():
        rows.append({
            'sample': model_name,
            'variable': CHAPTER4_LABELS.get(
                variable,
                VARIABLE_LABELS.get(variable, variable),
            ),
            'N': audit['n'],
            'raw_minimum': audit['raw_min'],
            'p01': audit['winsor_p01'],
            'p99': audit['winsor_p99'],
            'raw_maximum': audit['raw_max'],
            'lower_tail_adjusted_n': audit['lower_tail_adjusted_n'],
            'upper_tail_adjusted_n': audit['upper_tail_adjusted_n'],
        })
    return pd.DataFrame(rows)

def _calculate_vif_table(design_matrix: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    numeric = design_matrix.apply(pd.to_numeric, errors='coerce')
    for variable in numeric.columns:
        y = numeric[variable].to_numpy(dtype=float)
        other_variables = [column for column in numeric.columns if column != variable]
        if not other_variables:
            r_squared = np.nan
            vif = np.nan
        else:
            x_other = numeric[other_variables].to_numpy(dtype=float)
            x_other = np.column_stack([np.ones(len(x_other)), x_other])
            beta, *_ = np.linalg.lstsq(x_other, y, rcond=None)
            fitted = x_other @ beta
            residual_sum_squares = float(np.sum((y - fitted) ** 2))
            total_sum_squares = float(np.sum((y - y.mean()) ** 2))
            if total_sum_squares <= 0:
                r_squared = np.nan
                vif = np.nan
            else:
                r_squared = 1.0 - residual_sum_squares / total_sum_squares
                r_squared = min(max(r_squared, 0.0), 1.0)
                if np.isclose(1.0 - r_squared, 0.0):
                    vif = np.inf
                else:
                    vif = 1.0 / (1.0 - r_squared)
        rows.append({
            'model': model_name,
            'variable': VARIABLE_LABELS.get(variable, variable),
            'N': len(numeric),
            'R_squared_from_auxiliary_regression': r_squared,
            'VIF': vif,
            'tolerance': 1.0 / vif if pd.notna(vif) and np.isfinite(vif) else np.nan,
        })
    return pd.DataFrame(rows)

def build_baseline_multicollinearity_diagnostics(
    model_data: pd.DataFrame,
    predictors: list[str],
    model_name: str,
) -> pd.DataFrame:
    design_matrix = model_data[predictors].apply(pd.to_numeric, errors='coerce')
    complete_design = design_matrix.dropna(axis=0, how='any')
    correlation_matrix = complete_design.corr(method='pearson')
    correlation_matrix.index.name = 'variable'
    correlation_matrix = correlation_matrix.rename(
        index=VARIABLE_LABELS,
        columns=VARIABLE_LABELS,
    )
    vif_table = _calculate_vif_table(complete_design, model_name)
    vif_by_variable = vif_table.set_index('variable')['VIF']
    combined = correlation_matrix.copy()
    combined['VIF'] = vif_by_variable.reindex(combined.index)
    return combined.reset_index()

def build_duration_distribution(model_data: pd.DataFrame) -> pd.DataFrame:
    duration_data = model_data.copy()
    duration_data['admission_year'] = duration_data['admission_date'].dt.year.astype(int)
    bins = [0, 7, 30, 90, 365, THREE_YEARS_DAYS, np.inf]
    duration_data['duration_band'] = pd.cut(
        duration_data['duration_days'],
        bins=bins,
        labels=DURATION_BAND_LABELS,
        include_lowest=True,
        right=True,
        ordered=True,
    )
    band_counts = pd.crosstab(
        index=duration_data['admission_year'],
        columns=duration_data['duration_band'],
        dropna=True,
    ).reindex(columns=DURATION_BAND_LABELS, fill_value=0)
    band_counts['Total'] = band_counts.sum(axis=1)
    band_counts = band_counts.reset_index()
    band_counts.columns.name = None
    return band_counts

def save_duration_scatter(model_data: pd.DataFrame, output_path: Path) -> None:
    plot_data = model_data.copy().sort_values('admission_date')
    plot_data['placement_group'] = np.where(
        plot_data['private_placement'].eq(1),
        'Private placement',
        'Non-private',
    )
    figure, axis = plt.subplots(figsize=(11, 6.5))
    for group_name in ['Non-private', 'Private placement']:
        group = plot_data.loc[plot_data['placement_group'].eq(group_name)]
        if group.empty:
            continue
        axis.scatter(
            group['admission_date'],
            group['duration_days'],
            s=24,
            alpha=0.65,
            label=group_name,
        )
    censored = plot_data.loc[plot_data['event'].eq(0)]
    if not censored.empty:
        axis.scatter(
            censored['admission_date'],
            censored['duration_days'],
            marker='x',
            s=48,
            linewidths=1.2,
            label='Right-censored',
        )
    axis.set_title('Distribution of admission-to-commencement duration')
    axis.set_xlabel('Trade-list admission date')
    axis.set_ylabel('Observed duration (days)')
    axis.set_ylim(0, 500)
    axis.set_yticks(np.arange(0, 501, 50))
    axis.xaxis.set_major_locator(mdates.YearLocator(base=2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axis.grid(axis='y', alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(figure)

def _validate_survival_plot_data(model_data: pd.DataFrame) -> pd.DataFrame:
    required = {'duration_days', 'event', 'private_placement'}
    missing = sorted(required.difference(model_data.columns))
    if missing:
        raise ValueError(f'Kaplan-Meier plot is missing columns: {missing}')
    plot_data = model_data[['duration_days', 'event', 'private_placement']].copy()
    plot_data['duration_days'] = pd.to_numeric(
        plot_data['duration_days'],
        errors='coerce',
    )
    plot_data['event'] = pd.to_numeric(plot_data['event'], errors='coerce')
    plot_data['private_placement'] = pd.to_numeric(
        plot_data['private_placement'],
        errors='coerce',
    )
    plot_data = plot_data.dropna(subset=['duration_days', 'event', 'private_placement'])
    plot_data = plot_data.loc[
        plot_data['duration_days'].gt(0)
        & plot_data['event'].isin([0, 1])
        & plot_data['private_placement'].isin([0, 1])
    ].copy()
    if plot_data.empty:
        raise ValueError('No valid observations are available for Kaplan-Meier estimation.')
    plot_data['event'] = plot_data['event'].astype(int)
    plot_data['private_placement'] = plot_data['private_placement'].astype(int)
    return plot_data

def _format_kaplan_meier_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title)
    axis.set_xlabel('Admission-to-commencement duration (days)')
    axis.set_ylabel('Kaplan-Meier survival estimate')
    axis.set_xlim(0, KAPLAN_MEIER_DISPLAY_MAX_DAYS)
    axis.set_xticks(np.arange(0, KAPLAN_MEIER_DISPLAY_MAX_DAYS + 1, 50))
    axis.set_ylim(0, 1.02)
    axis.set_yticks(np.linspace(0, 1, 6))
    axis.grid(axis='both', alpha=0.2)

def save_kaplan_meier_overall(model_data: pd.DataFrame, output_path: Path) -> None:
    plot_data = _validate_survival_plot_data(model_data)
    survivor_color = '#1f77b4'
    ci_alpha = 0.2
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=plot_data['duration_days'],
        event_observed=plot_data['event'],
        label='Survivor function',
    )
    figure, axis = plt.subplots(figsize=(8.5, 6.0))
    kmf.plot_survival_function(
        ax=axis,
        ci_show=True,
        ci_alpha=ci_alpha,
        show_censors=True,
        censor_styles={'marker': '+', 'ms': 6, 'mew': 1.0},
        linewidth=2.0,
        color=survivor_color,
    )
    _format_kaplan_meier_axis(axis, 'Overall Kaplan-Meier survival estimate')
    overall_legend_handles = [
        Line2D(
            [0],
            [0],
            color=survivor_color,
            linewidth=2.0,
            label='Survivor function',
        ),
        Patch(
            facecolor=survivor_color,
            edgecolor='none',
            alpha=ci_alpha,
            label='95% CI',
        ),
    ]
    axis.legend(handles=overall_legend_handles, frameon=False, loc='upper right')
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(figure)

def save_kaplan_meier_by_placement(model_data: pd.DataFrame, output_path: Path) -> None:
    plot_data = _validate_survival_plot_data(model_data)
    groups = [
        ('Non-private', 0, '#1f77b4'),
        ('Private placement', 1, '#d62728'),
    ]
    figure, axis = plt.subplots(figsize=(8.5, 6.0))
    legend_handles = []

    for label, value, color in groups:
        group = plot_data.loc[plot_data['private_placement'].eq(value)]
        if group.empty:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=group['duration_days'],
            event_observed=group['event'],
            label=label,
        )
        kmf.plot_survival_function(
            ax=axis,
            ci_show=True,
            ci_alpha=0.18,
            show_censors=True,
            censor_styles={'marker': '+', 'ms': 6, 'mew': 1.0},
            linewidth=2.0,
            color=color,
        )
        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=2.0, label=label)
        )

    _format_kaplan_meier_axis(
        axis,
        'Kaplan-Meier survival estimates by placement type',
    )
    axis.legend(handles=legend_handles, frameon=False, loc='upper right')
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(figure)

def _smooth_by_event_time(
    x_values: pd.Series,
    y_values: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    plot_data = pd.DataFrame({
        'x': pd.to_numeric(x_values, errors='coerce'),
        'y': pd.to_numeric(y_values, errors='coerce'),
    }).dropna()
    if plot_data.empty:
        return (np.array([]), np.array([]))
    plot_data = plot_data.sort_values('x')
    window = max(5, min(25, int(np.ceil(len(plot_data) / 5))))
    smooth = plot_data['y'].rolling(window=window, center=True, min_periods=1).mean()
    return (plot_data['x'].to_numpy(), smooth.to_numpy())

def _draw_scaled_schoenfeld_axis(
    axis: plt.Axes,
    event_times: pd.Series,
    residuals: pd.Series,
    title: str,
    ylabel: str,
    p_value: float | None = None,
    display_max_days: float = KAPLAN_MEIER_DISPLAY_MAX_DAYS,
) -> None:
    plot_data = pd.DataFrame({
        'duration_days': pd.to_numeric(event_times, errors='coerce'),
        'residual': pd.to_numeric(residuals, errors='coerce'),
    }).dropna()
    plot_data = plot_data.loc[
        plot_data['duration_days'].between(0, display_max_days, inclusive='both')
    ]
    if plot_data.empty:
        axis.set_title(title)
        axis.text(
            0.5,
            0.5,
            'No residuals available',
            ha='center',
            va='center',
            transform=axis.transAxes,
        )
        axis.set_xlim(0, display_max_days)
        axis.set_xlabel('Duration days')
        return
    axis.scatter(
        plot_data['duration_days'],
        plot_data['residual'],
        s=14,
        alpha=0.55,
        color='#2f6f9f',
        edgecolors='none',
    )
    smooth_x, smooth_y = _smooth_by_event_time(
        plot_data['duration_days'],
        plot_data['residual'],
    )
    if len(smooth_x) > 0:
        axis.plot(
            smooth_x,
            smooth_y,
            color='#c43c39',
            linewidth=2.0,
            label='rolling mean',
        )
    axis.axhline(0, color='#333333', linewidth=0.9, linestyle='--', label='zero line')
    axis.set_xlim(0, display_max_days)
    axis.grid(axis='both', alpha=0.18)
    if p_value is None or pd.isna(p_value):
        subtitle = title
    else:
        subtitle = f'{title} (PH p={p_value:.4g})'
    axis.set_title(subtitle, fontsize=10)
    axis.set_xlabel('Duration days')
    axis.set_ylabel(ylabel)
    axis.legend(frameon=False, fontsize=8, loc='best')

def save_schoenfeld_residual_plots(
    cph: CoxPHFitter,
    fit_data: pd.DataFrame,
    model_name: str,
    model_slug: str,
    ph_result: pd.DataFrame,
    output_dir: Path,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    residuals = cph.compute_residuals(fit_data, kind='scaled_schoenfeld')
    variables = [
        str(variable)
        for variable in cph.params_.index
        if str(variable) in residuals.columns
    ]
    event_times = fit_data.loc[residuals.index, 'duration_days']
    ph_p_values = ph_result.set_index('variable')['ph_test_p_value'].to_dict()
    plot_records: list[dict] = []
    if variables:
        panel_count = len(variables)
        ncols = 2
        nrows = int(np.ceil(panel_count / ncols))
        figure, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(13.0, max(4.0, 3.2 * nrows)),
        )
        axes_array = np.atleast_1d(axes).ravel()
        for axis, variable in zip(axes_array, variables):
            _draw_scaled_schoenfeld_axis(
                axis,
                event_times,
                residuals[variable],
                variable,
                'Scaled Schoenfeld residual',
                ph_p_values.get(variable),
            )
        for axis in axes_array[panel_count:]:
            axis.set_visible(False)
        figure.suptitle('Scaled Schoenfeld residual plots', fontsize=13)
        figure.tight_layout(rect=[0, 0, 1, 0.97])
        combined_path = output_dir / f'{model_slug}_scaled_schoenfeld_residuals.png'
        figure.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close(figure)
        plot_records.append({
            'model': model_name,
            'plot_scope': 'model_combined',
            'variable': 'ALL_VARIABLES',
            'path': str(combined_path),
        })
    return plot_records

def fit_cox_model(
    model_data: pd.DataFrame,
    model_name: str,
    predictors: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, CoxPHFitter, pd.DataFrame]:
    fit_columns = ['duration_days', 'event', 'issuer_cluster', *predictors]
    fit_data = model_data[fit_columns].copy()
    cph = CoxPHFitter(penalizer=0.0)

    # Estimate the Cox proportional hazards model with standard errors
    # clustered by issuer to account for multiple bonds from the same issuer.
    cph.fit(
        fit_data,
        duration_col='duration_days',
        event_col='event',
        cluster_col='issuer_cluster',
        robust=True,
        show_progress=False,
    )
    coefficients = cph.summary.reset_index().rename(
        columns={
            'covariate': 'variable',
            'coef': 'coefficient',
            'exp(coef)': 'hazard_ratio',
            'se(coef)': 'robust_std_error',
            'z': 'z_value',
            'p': 'p_value',
            'exp(coef) lower 95%': 'hr_ci_95_low',
            'exp(coef) upper 95%': 'hr_ci_95_high',
        }
    )
    coefficients['variable'] = coefficients['variable'].map(VARIABLE_LABELS)
    coefficients = coefficients[
        [
            'variable',
            'coefficient',
            'robust_std_error',
            'z_value',
            'p_value',
            'hazard_ratio',
            'hr_ci_95_low',
            'hr_ci_95_high',
        ]
    ]
    coefficients.insert(0, 'model', model_name)
    lrt = cph.log_likelihood_ratio_test()
    metrics = pd.DataFrame([
        {
            'model': model_name,
            'n': len(model_data),
            'events': int(model_data['event'].sum()),
            'censored': int((1 - model_data['event']).sum()),
            'issuers': int(model_data['company_name'].nunique()),
            'concordance_index': float(cph.concordance_index_),
            'partial_log_likelihood': float(cph.log_likelihood_),
            'partial_AIC': float(cph.AIC_partial_),
            'likelihood_ratio_chi2': float(lrt.test_statistic),
            'likelihood_ratio_df': int(lrt.degrees_freedom),
            'likelihood_ratio_p_value': float(lrt.p_value),
        }
    ])

    # Test the proportional hazards assumption using scaled Schoenfeld residuals
    # with the Kaplan-Meier transformation of analysis time.
    ph_result = proportional_hazard_test(
        cph,
        fit_data,
        time_transform='km',
    ).summary.reset_index()
    first_column = ph_result.columns[0]
    ph_result = ph_result.rename(
        columns={
            first_column: 'variable',
            'test_statistic': 'ph_test_chi2',
            'p': 'ph_test_p_value',
        }
    )
    ph_result.insert(0, 'model', model_name)
    ph_result.insert(1, 'test_scope', 'variable')
    ph_result['ph_test_df'] = 1
    return (coefficients, metrics, ph_result, cph, fit_data)

def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / 'data'
    input_path = data_dir / 'KASE_Corporate_Bonds_2026-07-03.xlsx'
    cny_fx_path = data_dir / 'CNY-KZT exchange rate.xlsx'
    usd_fx_path = data_dir / 'USD-KZT exchange rate.xlsx'
    output_dir = script_dir / 'outputs'
    cutoff = ANALYSIS_CUTOFF

    output_dir.mkdir(parents=True, exist_ok=True)
    fx_tables = load_local_fx_tables(cny_fx_path, usd_fx_path)
    data, cleaning_summary = load_and_clean(input_path, 'Sheet1', cutoff, fx_tables)
    coefficient_tables = []
    metric_tables = []
    ph_tables = []
    model_audits = []
    baseline_sample: pd.DataFrame | None = None
    baseline_transform_audit: dict | None = None
    baseline_model_name: str | None = None
    baseline_predictors: list[str] | None = None
    schoenfeld_plot_dir = output_dir / 'scaled_schoenfeld_residual_plots'
    for model_index, spec in enumerate(MODEL_SPECS, start=1):
        model_data, model_audit, predictors = prepare_model_sample(data, spec, cutoff)
        model_slug = f'model_{model_index:02d}'
        coefficients, metrics, ph_result, cph, fit_data = fit_cox_model(
            model_data,
            spec.name,
            predictors,
        )
        save_schoenfeld_residual_plots(
            cph,
            fit_data,
            spec.name,
            model_slug,
            ph_result,
            schoenfeld_plot_dir,
        )
        coefficient_tables.append(coefficients)
        metric_tables.append(metrics)
        ph_tables.append(ph_result)
        model_audits.append(model_audit)
        if spec.include_private and (not spec.drop_over_three_years):
            baseline_sample = model_data.copy()
            baseline_transform_audit = model_audit['transform_audit']
            baseline_model_name = spec.name
            baseline_predictors = list(predictors)
    coefficients = pd.concat(coefficient_tables, ignore_index=True)
    metrics = pd.concat(metric_tables, ignore_index=True)
    ph_results = pd.concat(ph_tables, ignore_index=True)
    if (
        baseline_sample is None
        or baseline_transform_audit is None
        or baseline_model_name is None
        or baseline_predictors is None
    ):
        raise RuntimeError('Baseline sample was not created.')
    sample_construction = build_sample_construction_table(
        cleaning_summary,
        model_audits[0],
    )
    duration_band_counts = build_duration_distribution(baseline_sample)
    descriptive_all = build_descriptive_statistics(
        baseline_sample,
        sample_name='baseline',
        include_private=True,
    )
    winsor_audit_all = build_winsorisation_audit('baseline', baseline_transform_audit)
    baseline_correlation_and_vif = build_baseline_multicollinearity_diagnostics(
        baseline_sample,
        baseline_predictors,
        baseline_model_name,
    )
    save_duration_scatter(
        baseline_sample,
        output_dir / 'duration_distribution_by_admission_date.png',
    )
    save_kaplan_meier_overall(
        baseline_sample,
        output_dir / 'overall_kaplan_meier_survivor_function.png',
    )
    save_kaplan_meier_by_placement(
        baseline_sample,
        output_dir / 'kaplan_meier_by_placement_type.png',
    )
    ph_results['variable'] = ph_results['variable'].map(
        lambda value: VARIABLE_LABELS.get(value, value)
    )
    with pd.ExcelWriter(
        output_dir / 'chapter4_tables.xlsx',
        engine='openpyxl',
    ) as writer:
        sample_construction.to_excel(writer, sheet_name='Sample construction', index=False)
        duration_band_counts.to_excel(writer, sheet_name='Duration distribution', index=False)
        winsor_audit_all.to_excel(writer, sheet_name='Winsorisation', index=False)
        descriptive_all.to_excel(writer, sheet_name='Descriptive statistics', index=False)
        baseline_correlation_and_vif.to_excel(
            writer,
            sheet_name='Correlation and VIF',
            index=False,
        )
    with pd.ExcelWriter(
        output_dir / 'cox_model_results.xlsx',
        engine='openpyxl',
    ) as writer:
        coefficients.to_excel(writer, sheet_name='Cox coefficients', index=False)
        metrics.to_excel(writer, sheet_name='Model fit', index=False)
        ph_results.to_excel(writer, sheet_name='PH tests', index=False)


if __name__ == '__main__':
    main()
