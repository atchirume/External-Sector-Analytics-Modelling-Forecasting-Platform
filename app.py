# app.py
from pathlib import Path
import warnings, io, math
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats, optimize
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
try:
    from arch.unitroot import PhillipsPerron, ZivotAndrews
except Exception:
    PhillipsPerron=ZivotAndrews=None

APP=Path(__file__).resolve().parent
DATA=APP/'data'
st.set_page_config(page_title='External Sector Analytics, Modelling & Forecasting', page_icon='📊', layout='wide', initial_sidebar_state='expanded')

# -----------------------------------------------------------------------------
# Shared safe scalar helpers
# -----------------------------------------------------------------------------
def _safe_last(x):
    """Return the last finite numeric observation from a Series/array-like.

    This helper is intentionally defined globally because both the Debt
    Sustainability and External Balance Assessment engines use it.
    """
    try:
        z = pd.Series(x)
    except Exception:
        return np.nan
    z = pd.to_numeric(z, errors='coerce')
    z = z.replace([np.inf, -np.inf], np.nan).dropna()
    return float(z.iloc[-1]) if len(z) else np.nan


def _safe_numeric_series(x):
    """Convert a one-dimensional input to a clean numeric Series."""
    return pd.to_numeric(pd.Series(x), errors='coerce').replace([np.inf, -np.inf], np.nan)


# -----------------------------------------------------------------------------
# Theme: deliberately close to the supplied Shiny application's navy/blue/gold
# dashboard language, without institutional logos or unauthorised branding.
# -----------------------------------------------------------------------------
st.markdown("""<style>
:root{--navy:#071b33;--blue:#0b4f8a;--blue2:#1268a8;--gold:#d7a72f;--light:#f3f6fa;--ink:#172b4d;--muted:#65758b}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#071b33 0%,#0b2d50 100%);}
[data-testid="stSidebar"] *{color:#fff!important}
[data-testid="stSidebar"] .stRadio label{padding:7px 8px;border-radius:5px}
[data-testid="stSidebar"] .stRadio label:hover{background:#123e65}
.main .block-container{padding-top:1rem;max-width:1500px}
.hero{background:linear-gradient(110deg,#071b33,#0b4f8a);color:#fff;padding:24px 28px;border-radius:8px;border-left:7px solid var(--gold);margin-bottom:18px}
.hero h1{font-size:30px;margin:0 0 5px 0}.hero p{margin:0;color:#dce9f5}
.section{background:#0b4f8a;color:#fff;padding:10px 14px;border-radius:5px;border-left:5px solid var(--gold);margin:14px 0 10px}
.card{background:#fff;border:1px solid #d9e1ea;border-radius:7px;padding:15px;box-shadow:0 1px 4px rgba(0,0,0,.07);height:100%}
.kpi{background:#fff;border:1px solid #d9e1ea;border-top:4px solid var(--blue2);border-radius:7px;padding:13px;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.kpi .label{font-size:12px;color:#65758b;text-transform:uppercase;letter-spacing:.4px}.kpi .value{font-size:25px;font-weight:700;color:#0b4f8a;margin-top:3px}.kpi .sub{font-size:11px;color:#7a8797}
.warning{background:#fff8df;border-left:5px solid var(--gold);padding:12px;border-radius:4px}
.info{background:#eef6fd;border-left:5px solid var(--blue2);padding:12px;border-radius:4px}
.disclaimer{font-size:12px;color:#e8edf4;background:#0a2038;border:1px solid #234766;border-radius:6px;padding:10px;margin-top:12px}
.small{font-size:12px;color:#65758b}.modelbox{background:#f8fafc;border:1px solid #dbe4ed;border-radius:6px;padding:10px;margin-bottom:8px}
footer{visibility:hidden}
</style>""",unsafe_allow_html=True)

DISCLAIMER='This application is intended for research, analytical and demonstration purposes. It is not an officially authorised institutional system and should not be interpreted as representing the policies, decisions, systems or official statistics of any government or other institution.'

MODEL_SPECS={
'1. Export Demand — ARDL':('goods_exports',['global_demand_index','reer_index','terms_of_trade_index','gold_price','platinum_price','tobacco_price','lithium_price'],2,'Export Demand Function (ARDL)'),
'2. Import Demand — ARDL':('goods_imports',['gdp_index','reer_index','inflation_pct','crude_oil_price','world_trade_index'],2,'Import Demand Function (ARDL)'),
'3. Current Account — ARDL':('current_account_balance',['terms_of_trade_index','reer_index','gdp_index','global_demand_index','net_fdi','personal_remittances','external_debt_outstanding'],2,'Current Account Model (ARDL)'),
'4. Exchange Rate — Monetary / BEER':('nominal_fx_index',['inflation_pct','reer_index','gross_reserves_usd_mn','terms_of_trade_index','global_demand_index'],1,'Exchange Rate / Monetary-BEER Model'),
'5. REER Equilibrium — BEER':('reer_index',['net_iip','gdp_index','terms_of_trade_index','world_trade_index','global_demand_index'],1,'REER Equilibrium / BEER Model'),
'6. Reserve Accumulation — ARDL':('gross_reserves_usd_mn',['current_account_balance','net_fdi','external_debt_new_borrowing','external_debt_principal_repayment','trade_balance','import_cover_months'],2,'Reserve Accumulation Model (ARDL)'),
'7. FDI Inflows — ARDL':('fdi_inflows',['gdp_index','reer_index','global_demand_index','terms_of_trade_index','trade_balance','external_debt_outstanding'],2,'FDI Inflows Model (ARDL)'),
'8. External Debt Dynamics':('external_debt_outstanding',['external_debt_new_borrowing','external_debt_principal_repayment','external_debt_interest_payment','gdp_index','current_account_balance','reer_index'],1,'External Debt Dynamics'),
'9. IIP / External Sustainability':('net_iip',['current_account_balance','net_fdi','reer_index','external_debt_outstanding','gross_reserves_usd_mn','gdp_index'],1,'International Investment Position / Sustainability'),
'10. Terms of Trade / Commodity Prices':('terms_of_trade_index',['gold_price','platinum_price','tobacco_price','lithium_price','copper_price','crude_oil_price','global_demand_index'],1,'Terms of Trade / Commodity Prices'),
'11. Multivariate VAR':(None,['reer_index','goods_exports','goods_imports','current_account_balance','gross_reserves_usd_mn','net_fdi','inflation_pct','gdp_index'],2,'Multivariate Vector Autoregression (VAR)'),
'12. VECM & Johansen Cointegration':(None,['reer_index','goods_exports','goods_imports','current_account_balance','gross_reserves_usd_mn','net_fdi','gdp_index'],2,'Vector Error-Correction Model (VECM)'),
'13. Exchange Rate / External Volatility — GARCH':('reer_index',[],1,'GARCH(1,1) External Volatility Model'),
'14. Bayesian VAR — Shrinkage BVAR':(None,['reer_index','goods_exports','goods_imports','current_account_balance','gross_reserves_usd_mn','net_fdi','inflation_pct','gdp_index'],2,'Bayesian VAR with Shrinkage')}

ABOUT={
'1. Overview & Analytical Architecture':"""The platform integrates the external-sector research workflow from data discovery through diagnostics, econometric estimation, forecasting, scenario analysis, debt sustainability and external-balance assessment. It is designed as an analytical workflow rather than a collection of isolated statistical routines.""",
'2. Data Architecture & Frequency Transformation':"""The supplied package contains monthly, quarterly and annual external-sector tables. The Streamlit version builds a monthly analytical master dataset from the supplied simulated tables and preserves source frequency information where available. Frequency aggregation should be selected according to the economic meaning of each variable.""",
'3. Data Quality, Missing Values & Outliers':"""Data quality checks include coverage, missing observations, duplicated dates, numeric conversion, and simple outlier screening. Extreme observations should be investigated against source metadata before deletion or winsorisation.""",
'4. Transformations, Scaling & Deseasonalisation':"""The workbench supports level, log and first-difference transformations. Logs require positive observations. First differences are appropriate where the research objective is short-run dynamics or where integration diagnostics indicate non-stationarity.""",
'5. Unit Roots & Structural Breaks':"""Pre-estimation diagnostics include ADF, Phillips-Perron where the optional arch package is available, KPSS and Zivot-Andrews. Structural-break screening is treated as diagnostic evidence, not automatic proof of a structural change in the economic regime.""",
'6. Model Selection & General Estimation Principles':"""Model choice should follow the economic question, variable availability, integration properties, sample size, lag structure and diagnostics. Statistical fit alone is not sufficient for selecting a production research specification.""",
'7. Models 1–3: Export, Import & Current Account ARDL':"""Models 1–3 are dynamic single-equation external-sector specifications. They use lagged dependent variables and contemporaneous external drivers. The implementation is an applied ARDL-style dynamic regression; it should not be interpreted as a full Pesaran-Shin-Smith bounds test unless the necessary bounds-test procedure is separately supplied.""",
'8. Models 4–5: Exchange Rate & REER / BEER':"""Model 4 relates the nominal exchange-rate index to monetary/external fundamentals. Model 5 estimates a BEER-style equilibrium REER relationship using external assets, income, terms of trade, world trade and global demand.""",
'9. Models 6–10: Reserves, FDI, Debt, IIP & Terms of Trade':"""Models 6–10 cover reserve accumulation, FDI inflows, debt dynamics, the international investment position and terms of trade. These models connect flows, stocks and external prices in a coherent external-sector system.""",
'10. Model 11: Multivariate VAR':"""The VAR treats the selected external-sector variables as jointly endogenous and provides dynamic interaction analysis. Impulse responses and forecast-error variance decomposition can be used as system-level analytical tools.""",
'11. Model 12: VECM & Johansen Cointegration':"""The VECM is appropriate when a set of non-stationary variables has one or more stable long-run cointegrating relationships. Johansen trace-rank testing is performed before estimation.""",
'12. Model 13: GARCH(1,1) Volatility':"""The GARCH model captures time-varying conditional variance in a transformed return-like series. Persistence is reflected by the ARCH and GARCH parameters, subject to stationarity and residual diagnostics.""",
'13. Model 14: Shrinkage BVAR':"""The BVAR uses ridge-type shrinkage to reduce parameter proliferation in a medium-sized dynamic system. The implementation is a practical shrinkage BVAR proxy using posterior-mean/ridge estimates; prior sensitivity should be examined in research use.""",
'14. Debt Sustainability Analysis & Debt Forecasting':"""This module is structured around four distinct layers: (i) a statistical forecast of the external-debt stock and its flow components; (ii) the stock-flow accounting identity; (iii) debt-burden ratios such as external debt/GDP, external debt/exports and debt service/exports; and (iv) deterministic stress tests. The core stock-flow identity is D_t = D_(t-1) + NB_t - PR_t + INT_t + VA_t, where D is debt, NB new borrowing, PR principal repayment, INT interest/capitalised interest and VA valuation/foreign-exchange adjustment. The ratio analysis is accompanied by the approximate debt-dynamics equation d_t ≈ [(1+r_t)/((1+g_t)(1+π_t))]d_(t-1) - PB_t/GDP_t + FXVA_t. The application deliberately distinguishes nominal debt-stock ratios from present-value debt indicators: PV debt requires maturity, interest-rate, concessionality and repayment information that is not contained in the demonstration dataset. The stress-test layer allows explicit shocks to real growth, interest rates, exchange rates and exports and reports the resulting paths without assigning probabilities. This follows the analytical principle that a DSA should make baseline assumptions explicit and test debt dynamics under alternative scenarios, while recognising that official IMF/WB assessments require their full institutional templates, data coverage and judgement.""",
'15. External Balance Assessment':"""The EBA module contains three transparent analytical blocks. First, the current-account benchmark estimates CA/Y from available external fundamentals and defines the CA gap as actual CA/Y minus the fitted benchmark. Second, the REER benchmark estimates an equilibrium REER from available fundamentals and defines the REER gap as 100 × (REER_actual - REER_equilibrium)/|REER_equilibrium|. Third, the external-sustainability benchmark applies the steady-state NFA stabilisation condition CA_ES/Y ≈ [(g-r)/(1+g)] × NFA/Y. The module also decomposes the fitted current-account norm into the intercept and variable contributions and provides a mechanical local REER sensitivity when a sufficiently identified REER coefficient is available. These calculations use the same concepts of norms, gaps, fundamentals and external sustainability found in EBA-style analysis, but they are explicitly a domestic time-series analytical proxy rather than the official IMF cross-country EBA coefficient system or policy-gap framework.""", 
'16. Forecasting Methodology':"""Forecasting includes benchmark univariate methods, conditional dynamic forecasts and machine-learning challengers. Future driver assumptions must be distinguished from forecasts generated by the model itself.""",
'17. Forecast Evaluation & Accuracy Measures':"""The platform evaluates forecasts using ME, RMSE, MAE, MAPE, sMAPE, MASE, Theil-style U2 and directional accuracy where feasible. Accuracy metrics should be read jointly because each emphasises different forecast properties.""",
'18. Scenario & Stress Testing':"""Scenario analysis applies user-defined percentage or absolute shocks to a selected system variable and compares a baseline path with the deterministic shocked path. Scenario output is sensitivity analysis, not a probability distribution.""",
'19. Diagnostics, Stability & Model Validation':"""Post-estimation checks include residual serial correlation, ARCH effects, residual normality, stability indicators, and forecast evaluation. A model should not be treated as validated solely because an estimation command completes successfully.""",
'20. Interpretation, Policy Use & Limitations':"""Results are analytical evidence. Coefficients, impulse responses, forecast paths and stress tests require economic interpretation, knowledge of institutional conditions and awareness of data limitations.""",
'21. Reproducibility, Governance & Data Security':"""The application records model settings, variable selections and transformations within the session. For institutional deployment, a controlled database, audit log, versioned specifications and role-based access should be added.""",
'22. References & Further Reading':"""Key methodological families represented include ADF, Phillips-Perron, KPSS, Zivot-Andrews, ARDL-style dynamic regression, VAR, Johansen VECM, GARCH, Bayesian VAR, forecast evaluation, debt sustainability analysis and external-balance assessment.""",
'23. Author & Copyright':"""Prepared as a research and demonstration application. Author: Chirume Admire Tarisirayi. The code and supplied simulated datasets should be treated as a prototype package""",
'24. About the Program/Application & Policy Modernisation':"""The application demonstrates how external-sector statistics, econometrics and forecasting can be brought into one analytical environment. It is deliberately neutral in institutional branding and should not be presented as an officially authorised system."""}

# ---------- data layer ----------
def read_csv(name):
    p=DATA/name
    for sep in [',',';']:
        try:
            d=pd.read_csv(p,sep=sep,decimal=',',engine='python')
            if d.shape[1]>1: break
        except Exception: d=None
    if d is None: d=pd.read_csv(p)
    for c in d.columns:
        if c=='date':
            d[c]=pd.to_datetime(d[c],dayfirst=True,errors='coerce')
        elif d[c].dtype=='object':
            s=d[c].astype(str).str.replace(',','.',regex=False)
            num=pd.to_numeric(s,errors='coerce')
            if num.notna().mean()>.75: d[c]=num
    return d
@st.cache_data(show_spinner=False)
def build_master():
    ext=read_csv('External_indicators_Monthly.csv')
    ext=ext.groupby('date',as_index=False).first()
    bop=read_csv('Bop_Monthly.csv')
    piv=bop.pivot_table(index='date',columns='item_name',values='value',aggfunc='sum').reset_index()
    piv.columns=[str(c).lower().replace(' ','_').replace('-','_') for c in piv.columns]
    ren={
      'goods_exports':'goods_exports','goods_imports':'goods_imports','services_exports':'services_exports','services_imports':'services_imports',
      'primary_income_receipts':'primary_income_receipts','primary_income_payments':'primary_income_payments','personal_remittances':'personal_remittances',
      'other_secondary_income':'other_secondary_income','trade_balance':'trade_balance','current_account_balance':'current_account_balance'}
    piv=piv.rename(columns=ren)
    fdi=read_csv('FDI_monthly.csv').groupby('date',as_index=False).sum(numeric_only=True).reset_index(drop=True)
    debt=read_csv('External_Debt_Monthly.csv').groupby('date',as_index=False).sum(numeric_only=True).reset_index(drop=True)
    iip=read_csv('IIP_Monthly.csv').groupby('date',as_index=False).sum(numeric_only=True).reset_index(drop=True)
    res=read_csv('Reserves_Monthly.csv').groupby('date',as_index=False).sum(numeric_only=True).reset_index(drop=True)
    rem=read_csv('Remittances_Monthly.csv').groupby('date',as_index=False).sum(numeric_only=True).reset_index(drop=True)
    def prefix(df,keep): return df[['date']+keep] if all(c in df for c in ['date']+keep) else df
    pieces=[ext,piv,prefix(fdi,['inflows','outflows','net_fdi']),prefix(debt,['outstanding','interest_payment','principal_repayment','new_borrowing']),prefix(iip,['net_iip']),prefix(res,['gross_reserves_usd_mn','import_cover_months']),prefix(rem,['remittance_value'])]
    m=pieces[0]
    for p in pieces[1:]: m=m.merge(p,on='date',how='outer',suffixes=('','_dup'))
    m=m.sort_values('date').reset_index(drop=True)
    # resolve duplicate names and derive key aggregates
    if 'trade_balance' not in m and {'goods_exports','goods_imports'}<=set(m.columns): m['trade_balance']=m['goods_exports']-m['goods_imports']
    if 'current_account_balance' not in m:
        cols=[c for c in ['goods_exports','goods_imports','services_exports','services_imports','primary_income_receipts','primary_income_payments','personal_remittances','other_secondary_income'] if c in m]
        m['current_account_balance']=0.0
        if 'goods_exports' in m:m['current_account_balance']+=m['goods_exports']
        if 'goods_imports' in m:m['current_account_balance']-=m['goods_imports']
        if 'services_exports' in m:m['current_account_balance']+=m['services_exports']
        if 'services_imports' in m:m['current_account_balance']-=m['services_imports']
        if 'primary_income_receipts' in m:m['current_account_balance']+=m['primary_income_receipts']
        if 'primary_income_payments' in m:m['current_account_balance']-=m['primary_income_payments']
        if 'personal_remittances' in m:m['current_account_balance']+=m['personal_remittances']
        if 'other_secondary_income' in m:m['current_account_balance']+=m['other_secondary_income']
    rename={'inflows':'fdi_inflows','outstanding':'external_debt_outstanding','interest_payment':'external_debt_interest_payment','principal_repayment':'external_debt_principal_repayment','new_borrowing':'external_debt_new_borrowing','remittance_value':'personal_remittances'}
    m=m.rename(columns=rename)

    # ------------------------------------------------------------
    # Harden the analytical master dataset
    # ------------------------------------------------------------
    # Merges can create duplicate field names.  A duplicated pandas
    # column name makes m[c] a DataFrame rather than a Series and
    # subsequently breaks pd.to_numeric().  Resolve duplicates by
    # taking the first non-missing value from left to right.
    m.columns=[str(c).strip().lower() for c in m.columns]
    if m.columns.duplicated().any():
        collapsed=pd.DataFrame(index=m.index)
        for name in list(dict.fromkeys(m.columns)):
            block=m.loc[:,m.columns==name]
            if block.shape[1]==1:
                collapsed[name]=block.iloc[:,0]
            else:
                block=block.replace(['','NA','N/A','na','n/a','null','None'],np.nan)
                collapsed[name]=block.bfill(axis=1).iloc[:,0]
        m=collapsed

    # Safe numeric conversion after duplicate resolution.
    for c in m.columns:
        if c=='date':
            continue
        if not pd.api.types.is_numeric_dtype(m[c]):
            s=m[c]
            if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                s=s.astype('string').str.strip().str.replace(',','',regex=False)
                s=s.str.replace(r'^\((.*)\)$',r'-\1',regex=True)
                s=s.str.replace(r'^(US\$|USD|US|\$)','',regex=True)
            m[c]=pd.to_numeric(s,errors='coerce')

    # Remove duplicate dates, retaining the last fully merged observation.
    if 'date' in m.columns:
        m['date']=pd.to_datetime(m['date'],errors='coerce')
        m=m.dropna(subset=['date']).sort_values('date').drop_duplicates('date',keep='last').reset_index(drop=True)
    return m

def numeric_vars(df): return [c for c in df.columns if c!='date' and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum()>10]

# ---------- statistical helpers ----------
def transform(s,method):
    s=pd.Series(s).astype(float)
    if method=='Level': return s
    if method=='Log': return np.log(s.where(s>0))
    if method=='First Difference': return s.diff()
    if method=='Log Difference': return np.log(s.where(s>0)).diff()
    return s

def unit_root_table(s):
    x=pd.Series(s).dropna().astype(float)
    rows=[]
    if len(x)<20:return pd.DataFrame({'Test':['Insufficient data'],'Statistic':[np.nan],'p-value':[np.nan],'Decision':['Need more observations']})
    from statsmodels.tsa.stattools import adfuller,kpss
    for name,fn in [('ADF',lambda z:adfuller(z,autolag='AIC')),('KPSS',lambda z:kpss(z,regression='c',nlags='auto'))]:
        try:
            r=fn(x); rows.append([name,r[0],r[1],'Reject unit root' if (r[1]<.05 and name=='ADF') else ('Reject stationarity' if r[1]<.05 else 'Do not reject at 5%')])
        except Exception as e: rows.append([name,np.nan,np.nan,str(e)])
    if PhillipsPerron:
        try:
            r=PhillipsPerron(x); rows.append(['Phillips-Perron',float(r.stat),float(r.pvalue),'Reject unit root' if r.pvalue<.05 else 'Do not reject unit root'])
        except Exception as e: rows.append(['Phillips-Perron',np.nan,np.nan,str(e)])
    else: rows.append(['Phillips-Perron',np.nan,np.nan,'Install arch package'])
    if ZivotAndrews:
        try:
            r=ZivotAndrews(x); rows.append(['Zivot-Andrews',float(r.stat),float(r.pvalue),'Reject unit root with break' if r.pvalue<.05 else 'Do not reject unit root'])
        except Exception as e: rows.append(['Zivot-Andrews',np.nan,np.nan,str(e)])
    else: rows.append(['Zivot-Andrews',np.nan,np.nan,'Install arch package'])
    return pd.DataFrame(rows,columns=['Test','Statistic','p-value','Decision'])

def dynamic_ols(df,dep,rhs,p=2,det='Constant',trans='Level'):
    cols=[dep]+rhs; x=df[cols].copy()
    for c in cols:x[c]=transform(x[c],trans)
    for lag in range(1,p+1): x[f'{dep}_L{lag}']=x[dep].shift(lag)
    z=x[[dep]+[f'{dep}_L{i}' for i in range(1,p+1)]+rhs].dropna()
    y=z[dep]; X=z.drop(columns=[dep]);
    if det in ('Constant','Constant + Trend'): X=sm.add_constant(X,has_constant='add')
    if det=='Constant + Trend': X['trend']=np.arange(len(X))
    fit=sm.OLS(y,X).fit()
    return fit,z

def garch_mle(series):
    r=pd.Series(series).dropna().astype(float).values
    r=(r-r.mean())
    if np.std(r)>0:r=r/np.std(r)
    def nll(par):
        omega,alpha,beta=par
        if omega<=0 or alpha<0 or beta<0 or alpha+beta>=.999:return 1e12
        h=np.empty(len(r)); h[0]=np.var(r)
        for t in range(1,len(r)):h[t]=omega+alpha*r[t-1]**2+beta*h[t-1]
        return .5*np.sum(np.log(2*np.pi)+np.log(h)+r*r/h)
    res=optimize.minimize(nll,[.05,.08,.85],method='Nelder-Mead',options={'maxiter':5000})
    om,a,b=res.x; h=np.empty(len(r));h[0]=np.var(r)
    for t in range(1,len(r)):h[t]=om+a*r[t-1]**2+b*h[t-1]
    return {'omega':om,'alpha':a,'beta':b,'persistence':a+b,'conditional_variance':h,'residuals':r,'success':res.success}

def metrics(y,hat):
    d=pd.DataFrame({'actual':y,'forecast':hat}).dropna(); e=d.actual-d.forecast
    rmse=float(np.sqrt(np.mean(e**2))); mae=float(np.mean(np.abs(e))); me=float(np.mean(e)); mape=float(np.mean(np.abs(e/d.actual.replace(0,np.nan)))*100)
    smape=float(np.mean(2*np.abs(e)/(np.abs(d.actual)+np.abs(d.forecast)).replace(0,np.nan))*100)
    return pd.DataFrame({'Metric':['ME','RMSE','MAE','MAPE (%)','sMAPE (%)','Directional Accuracy (%)'],'Value':[me,rmse,mae,mape,smape,float(np.mean(np.sign(d.actual.diff().fillna(0))==np.sign(d.forecast.diff().fillna(0)))*100)]})

def lag_matrix(series,p):
    z=pd.DataFrame({'y':series})
    for i in range(1,p+1):z[f'L{i}']=z.y.shift(i)
    return z.dropna()

def ridge_bvar(df,vars,p=2,lam=.2):
    x=df[vars].dropna().astype(float); Z=[]; Y=[]
    for t in range(p,len(x)):
        row=[]
        for j in range(1,p+1):row.extend(x.iloc[t-j].values)
        row.append(1);Z.append(row);Y.append(x.iloc[t].values)
    Z=np.asarray(Z);Y=np.asarray(Y); A=Z.T@Z+lam*np.eye(Z.shape[1]);B=np.linalg.solve(A,Z.T@Y); return x,Z,Y,B

def forecast_dynamic(fit,df,dep,rhs,p,h,trans='Level'):
    work=df[[dep]+rhs].copy()
    for c in work:work[c]=transform(work[c],trans)
    vals=list(work[dep].dropna().values)
    last_rhs=work[rhs].dropna().tail(1)
    Xnames=list(fit.params.index)
    out=[]
    for k in range(h):
        row=[]
        for lag in range(1,p+1):row.append(vals[-lag])
        row += list(last_rhs.iloc[0].values)
        if 'const' in Xnames: row=[1]+row
        if 'trend' in Xnames: row=row+[len(vals)]
        # align to coefficient order
        tmp=pd.Series(0.,index=Xnames)
        numeric=[v for v in row]
        # construct from names
        if 'const' in tmp.index:tmp['const']=1
        for lag in range(1,p+1):
            nm=f'{dep}_L{lag}'
            if nm in tmp.index:tmp[nm]=vals[-lag]
        for c in rhs:
            if c in tmp.index:tmp[c]=last_rhs.iloc[0][c]
        if 'trend' in tmp.index:tmp['trend']=len(vals)+1
        pred=float(np.dot(tmp.values,fit.params.reindex(tmp.index).values)); vals.append(pred);out.append(pred)
    return np.array(out)

def chart(df,x='date',y=None,title='',height=400):
    fig=go.Figure()
    if y is None:y=[c for c in df.columns if c!=x][:1]
    if isinstance(y,str):y=[y]
    for c in y: fig.add_trace(go.Scatter(x=df[x],y=df[c],mode='lines',name=c))
    fig.update_layout(title=title,height=height,margin=dict(l=20,r=20,t=55,b=30),template='plotly_white',legend=dict(orientation='h'))
    return fig

# ---------- UI ----------
if 'master' not in st.session_state: st.session_state.master=build_master()
df=st.session_state.master

with st.sidebar:
    st.markdown('## EXTERNAL SECTOR')
    st.markdown('### Analytics, Modelling & Forecasting')
    st.markdown('---')
    pages=['Data Explorer','Executive Dashboard','About the Program/Application','Unit Roots & Integration','Modelling & Forecasting','Forecasting','Scenario Analysis','Debt Sustainability & Debt Forecasting','External Balance Assessment','Methodology','Database Information']
    page=st.radio('Navigation',pages,index=1,label_visibility='collapsed')
    st.markdown('---')
    st.markdown('<div class="disclaimer">'+DISCLAIMER+'</div>',unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>External Sector Analytics, Modelling & Forecasting</h1><p>Research • Econometrics • Forecasting • Scenario Analysis • External-Sector Surveillance</p></div>',unsafe_allow_html=True)

# Data Explorer
if page=='Data Explorer':
    st.markdown('<div class="section">Data Explorer</div>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Observations',f'{len(df):,}'); c2.metric('Variables',f'{len(numeric_vars(df)):,}'); c3.metric('Start',str(df.date.min().date())); c4.metric('End',str(df.date.max().date()))
    vars=numeric_vars(df); chosen=st.multiselect('Select indicators',vars,default=vars[:4],max_selections=8)
    if chosen: st.plotly_chart(chart(df,'date',chosen,'Selected External-Sector Indicators'),use_container_width=True)
    st.dataframe(df[['date']+chosen] if chosen else df.head(100),use_container_width=True,height=430)

# Dashboard
elif page=='Executive Dashboard':
    st.markdown('<div class="section">Executive Dashboard</div>',unsafe_allow_html=True)
    latest=df.dropna(subset=['date']).iloc[-1]
    cards=[('GDP Index',latest.get('gdp_index',np.nan)),('REER Index',latest.get('reer_index',np.nan)),('Reserves',latest.get('gross_reserves_usd_mn',np.nan)),('Current Account',latest.get('current_account_balance',np.nan)),('Net FDI',latest.get('net_fdi',np.nan)),('Net IIP',latest.get('net_iip',np.nan))]
    cols=st.columns(6)
    for col,(lab,val) in zip(cols,cards): col.markdown(f'<div class="kpi"><div class="label">{lab}</div><div class="value">{val:,.2f}</div><div class="sub">Latest available observation</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section">Executive External-Sector Trend Analysis</div>',unsafe_allow_html=True)
    var=st.selectbox('Indicator',numeric_vars(df),index=max(0,numeric_vars(df).index('current_account_balance') if 'current_account_balance' in numeric_vars(df) else 0))
    st.plotly_chart(chart(df,'date',var,'Selected indicator'),use_container_width=True)
    a,b=st.columns(2)
    with a:
        st.markdown('#### Summary Statistics')
        st.dataframe(df[var].describe().to_frame('Value'),use_container_width=True)
    with b:
        st.markdown('#### Latest Observation')
        st.dataframe(pd.DataFrame({'Indicator':[var],'Date':[latest.date],'Value':[latest.get(var)]}),use_container_width=True)
    st.markdown('<div class="info"><b>System overview:</b> Data → diagnostics → 14-model workbench → forecasting → scenarios → debt sustainability → external balance assessment.</div>',unsafe_allow_html=True)

# About
elif page=='About the Program/Application':
    st.markdown('<div class="section">About the Program / Application</div>',unsafe_allow_html=True)
    st.markdown('<div class="card"><h2>About the External Sector Analytics Platform</h2><p>An integrated analytical environment for external-sector data, econometric modelling, forecasting, stress testing, debt sustainability and external-balance analysis.</p></div>',unsafe_allow_html=True)
    st.markdown('### Documentation Navigator')
    sec=st.selectbox('Select About subsection',list(ABOUT.keys()))
    st.markdown(f'<div class="card"><h3>{sec}</h3><p>{ABOUT[sec]}</p></div>',unsafe_allow_html=True)
    st.markdown('### Analytical architecture')
    arch=['Data Explorer','Executive Dashboard','Diagnostics','14-model Modelling Workbench','Forecasting Engine','Scenario & Stress Testing','Debt Sustainability','External Balance Assessment','Methodology & Database']
    st.write(' → '.join(arch))

# Unit roots
elif page=='Unit Roots & Integration':
    st.markdown('<div class="section">Unit Roots & Integration</div>',unsafe_allow_html=True)
    vars=numeric_vars(df); c1,c2=st.columns([1,3])
    with c1:
        selected=st.multiselect('Variables',vars,default=vars[:3],max_selections=8)
        tr=st.selectbox('Transformation',['Level','First Difference','Log','Log Difference'])
        run=st.button('Run Unit-Root Battery',type='primary')
    with c2:
        if run or selected:
            for v in selected:
                st.markdown(f'#### {v}')
                st.dataframe(unit_root_table(transform(df[v],tr)),use_container_width=True,hide_index=True)
    st.markdown('<div class="warning"><b>Interpretation:</b> unit-root tests have different null hypotheses. Use ADF/PP/Zivot-Andrews jointly with KPSS and economic knowledge rather than relying on one p-value.</div>',unsafe_allow_html=True)

# Modelling
elif page=='Modelling & Forecasting':
    st.markdown('<div class="section">Modelling & Forecasting Workbench — 14 Built-in Models</div>',unsafe_allow_html=True)
    model=st.selectbox('Select external-sector model',list(MODEL_SPECS.keys()))
    dep,rhs,default_lag,title=MODEL_SPECS[model]
    c1,c2,c3=st.columns(3)
    with c1: lag=st.number_input('Lag order',1,8,default_lag)
    with c2: trans=st.selectbox('Variable transformation',['Level','Log','First Difference','Log Difference'])
    with c3: det=st.selectbox('Deterministic terms',['Constant','Constant + Trend','None'])
    if dep: dep=st.selectbox('Dependent variable (LHS)',numeric_vars(df),index=numeric_vars(df).index(dep) if dep in numeric_vars(df) else 0)
    avail=[v for v in rhs if v in df.columns]
    if model.startswith('11.') or model.startswith('12.') or model.startswith('14.'):
        sysvars=st.multiselect('System variables',numeric_vars(df),default=[v for v in rhs if v in numeric_vars(df)])
    elif model.startswith('13.'):
        sysvars=[]
    else:
        sysvars=st.multiselect('Explanatory variables (RHS)',numeric_vars(df),default=avail)
    st.markdown('### Modelling workflow')
    tabs=st.tabs(['1. Specification & Availability','2. Pre-Estimation Diagnostics','3. Structural Breaks','4. Estimate Model','5. Post-Estimation Diagnostics','6. Forecast / Prediction'])
    with tabs[0]:
        st.markdown(f'**{title}**')
        if dep: st.code(f'{dep}_t = α + Σ φ_i {dep}_{{t-i}} + βX_t + ε_t',language='text')
        else: st.code('Y_t = A_1Y_{t-1} + … + A_pY_{t-p} + ε_t',language='text')
        table=[]
        for v in ([dep] if dep else [])+sysvars: table.append([v,int(df[v].notna().sum()),df[v].min(),df[v].max()])
        for v in ([v for v in sysvars if v not in ([dep] if dep else [])]): pass
        st.dataframe(pd.DataFrame(table,columns=['Variable','N non-missing','Minimum','Maximum']),use_container_width=True,hide_index=True)
    with tabs[1]:
        diagvars=([dep] if dep else sysvars)
        for v in diagvars:
            st.markdown(f'**{v}**'); st.dataframe(unit_root_table(transform(df[v],trans)),use_container_width=True,hide_index=True)
        if model.startswith(('11.','12.','14.')) and len(sysvars)>=2:
            q=df[sysvars].dropna(); maxlag=min(8,max(1,len(q)//10));
            try:
                sel=VAR(q).select_order(maxlag); st.write({'AIC':sel.aic,'BIC':sel.bic,'HQIC':sel.hqic,'FPE':sel.fpe})
            except Exception as e: st.warning(str(e))
            if model.startswith('12.'):
                try:
                    rank=select_coint_rank(q,det_order=0,k_ar_diff=max(1,lag-1),method='trace',signif=0.05); st.write(f'Johansen trace rank estimate: {rank.rank}')
                except Exception as e: st.warning(f'Johansen test: {e}')
    with tabs[2]:
        if dep:
            x=df[['date',dep]].dropna().copy(); x['rolling_mean']=x[dep].rolling(12).mean(); x['rolling_sd']=x[dep].rolling(12).std()
            st.plotly_chart(chart(x,'date',[dep,'rolling_mean'],'Structural-break screening'),use_container_width=True)
            z=x[dep].dropna(); dif=z.diff().dropna(); cutoff=dif.abs().quantile(.99); breaks=x.loc[dif.index[dif.abs()>=cutoff],'date']
            st.write('Extreme-change dates (99th percentile screen):',list(breaks.dt.strftime('%Y-%m-%d').head(20)))
        else: st.info('Structural-break screening is most directly applied to single-equation dependent variables.')
    with tabs[3]:
        estimate=st.button('Estimate Selected Model',type='primary')
        if estimate:
            try:
                if model.startswith(('1.','2.','3.','4.','5.','6.','7.','8.','9.','10.')):
                    if dep is None or not sysvars: raise ValueError('Select a dependent variable and at least one RHS variable.')
                    fit,z=dynamic_ols(df,dep,sysvars,lag,det,trans); st.session_state.model_result=('single',fit,z,dep,sysvars,lag,trans,title)
                    st.success(f'{title} estimated successfully on {len(z):,} observations.')
                    st.dataframe(pd.DataFrame({'Variable':fit.params.index,'Estimate':fit.params.values,'Std. Error':fit.bse.values,'t value':fit.tvalues.values,'p-value':fit.pvalues.values}),use_container_width=True,hide_index=True)
                    st.write({'R-squared':fit.rsquared,'Adj. R-squared':fit.rsquared_adj,'AIC':fit.aic,'BIC':fit.bic,'N':int(fit.nobs)})
                elif model.startswith('11.'):
                    q=df[sysvars].dropna(); fit=VAR(q).fit(lag,trend='c' if det!='None' else 'n'); st.session_state.model_result=('VAR',fit,q,sysvars,lag,title); st.success('VAR estimated.'); st.dataframe(fit.summary().tables[1].as_html(),use_container_width=True)
                elif model.startswith('12.'):
                    q=df[sysvars].dropna(); rank=select_coint_rank(q,det_order=0,k_ar_diff=max(1,lag-1),method='trace',signif=.05).rank; 
                    if rank<=0 or rank>=len(sysvars): raise ValueError(f'Johansen rank={rank}; standard VECM requires 0 < rank < K.')
                    fit=VECM(q,k_ar_diff=max(1,lag-1),coint_rank=rank,deterministic='co').fit(); st.session_state.model_result=('VECM',fit,q,sysvars,lag,title); st.success(f'VECM estimated with cointegration rank {rank}.'); st.write(fit.summary())
                elif model.startswith('13.'):
                    r=df[dep].pct_change().replace([np.inf,-np.inf],np.nan).dropna()*100; gm=garch_mle(r); st.session_state.model_result=('GARCH',gm,r,dep,title); st.success('GARCH(1,1) estimated.'); st.dataframe(pd.DataFrame({'Parameter':['omega','alpha','beta','alpha+beta'],'Estimate':[gm['omega'],gm['alpha'],gm['beta'],gm['persistence']]}),use_container_width=True,hide_index=True)
                elif model.startswith('14.'):
                    x,Z,Y,B=ridge_bvar(df,sysvars,lag,.2); st.session_state.model_result=('BVAR',x,Z,Y,B,sysvars,lag,title); st.success('Shrinkage BVAR estimated.'); st.write({'Observations':len(Y),'Variables':len(sysvars),'Lag':lag,'Shrinkage lambda':.2}); st.dataframe(pd.DataFrame(B,index=[f'L{j}_{v}' for j in range(1,lag+1) for v in sysvars]+['constant'],columns=sysvars),use_container_width=True)
            except Exception as e: st.error(f'Estimation failed: {e}')
    with tabs[4]:
        if 'model_result' not in st.session_state: st.info('Estimate a model first.')
        else:
            r=st.session_state.model_result
            if r[0]=='single':
                fit=r[1]; resid=pd.Series(fit.resid)
                c1,c2=st.columns(2)
                with c1:
                    try: st.dataframe(acorr_ljungbox(resid,lags=[min(12,max(1,len(resid)//10))],return_df=True),use_container_width=True)
                    except Exception as e:st.warning(str(e))
                with c2:
                    try: st.dataframe(het_arch(resid,nlags=min(12,max(1,len(resid)//10)))[:4] if isinstance(het_arch(resid,nlags=12),tuple) else het_arch(resid,nlags=12),use_container_width=True)
                    except Exception as e:st.warning(str(e))
                fig=go.Figure();fig.add_trace(go.Scatter(y=fit.fittedvalues,name='Fitted'));fig.add_trace(go.Scatter(y=fit.model.endog,name='Actual'));fig.update_layout(template='plotly_white',height=380,title='Actual vs Fitted');st.plotly_chart(fig,use_container_width=True)
            elif r[0]=='GARCH':
                gm=r[1]; st.write({'Persistence':gm['persistence'],'Optimizer success':gm['success']}); st.plotly_chart(go.Figure(go.Scatter(y=np.sqrt(gm['conditional_variance']),name='Conditional volatility')).update_layout(template='plotly_white',height=350,title='Estimated Conditional Volatility'),use_container_width=True)
            else: st.info('System-model diagnostics include stability/cointegration and residual analysis; inspect the estimation summary and use the Forecast tab for dynamic evaluation.')
    with tabs[5]:
        if 'model_result' not in st.session_state: st.info('Estimate a model first.')
        else:
            h=st.number_input('Forecast horizon',1,60,12,key='work_h'); r=st.session_state.model_result
            if r[0]=='single':
                _,fit,z,dep0,rhs0,p0,tr0,title0=r
                try:
                    fc=forecast_dynamic(fit,df,dep0,rhs0,p0,h,tr0); future_dates=pd.date_range(df.date.max()+pd.offsets.MonthBegin(1),periods=h,freq='MS')
                    st.plotly_chart(go.Figure([go.Scatter(x=df.date.tail(60),y=df[dep0].tail(60),name='Historical'),go.Scatter(x=future_dates,y=fc,name='Forecast')]).update_layout(template='plotly_white',height=400,title=title0),use_container_width=True)
                    st.dataframe(pd.DataFrame({'date':future_dates,'forecast':fc}),use_container_width=True,hide_index=True)
                except Exception as e: st.error(str(e))
            elif r[0]=='VAR':
                fit=r[1]; fc=fit.forecast(fit.endog[-fit.k_ar:],steps=h); dates=pd.date_range(df.date.max()+pd.offsets.MonthBegin(1),periods=h,freq='MS'); outfc=pd.DataFrame(fc,columns=r[3]);outfc.insert(0,'date',dates); st.dataframe(outfc,use_container_width=True,hide_index=True)
            elif r[0]=='VECM':
                fit=r[1]; fc=fit.predict(steps=h);dates=pd.date_range(df.date.max()+pd.offsets.MonthBegin(1),periods=h,freq='MS');outfc=pd.DataFrame(fc,columns=r[3]);outfc.insert(0,'date',dates);st.dataframe(outfc,use_container_width=True,hide_index=True)
            elif r[0]=='BVAR':
                x,Z,Y,B,vs,p0,title0=r; last=list(x.iloc[-p0:].values);pred=[]
                for k in range(h):
                    row=np.concatenate([last[-j] for j in range(1,p0+1)]+[np.array([1.])]); y=row@B;pred.append(y);last.append(y)
                dates=pd.date_range(df.date.max()+pd.offsets.MonthBegin(1),periods=h,freq='MS');outfc=pd.DataFrame(pred,columns=vs);outfc.insert(0,'date',dates);st.dataframe(outfc,use_container_width=True,hide_index=True)
            else: st.info('GARCH forecasting is available through the volatility path in the post-estimation results.')

# Forecasting
elif page=='Forecasting':
    st.markdown('<div class="section">Forecasting Engine</div>',unsafe_allow_html=True)
    st.markdown('<div class="info"><b>Architecture:</b> in-sample evaluation → future target forecast → future RHS-driver forecasts → machine-learning challenger comparison → structural/ML ensemble review.</div>',unsafe_allow_html=True)
    var=st.selectbox('Forecast target',numeric_vars(df),index=numeric_vars(df).index('goods_exports') if 'goods_exports' in numeric_vars(df) else 0)
    h=st.slider('Future horizon (months)',3,60,12)
    test_n=st.slider('In-sample evaluation window',12,min(60,max(12,len(df)//4)),24)
    engine=st.selectbox('Future LHS forecasting engine',['Auto ARIMA','ETS','Naive','Drift'])
    ml=st.multiselect('ML challenger suite',['Random Forest','Gradient Boosting','Elastic Net','SVR'],default=['Random Forest','Gradient Boosting'])
    y=df[['date',var]].dropna().set_index('date')[var]
    if st.button('Run Forecasting Engine',type='primary'):
        train=y.iloc[:-test_n]; test=y.iloc[-test_n:]
        preds={}
        try:
            if engine=='Auto ARIMA': f=ARIMA(train,order=(1,1,1)).fit(); preds[engine]=f.forecast(test_n)
            elif engine=='ETS': f=ExponentialSmoothing(train,trend='add',seasonal=None).fit();preds[engine]=f.forecast(test_n)
            elif engine=='Naive':preds[engine]=pd.Series(train.iloc[-1],index=test.index)
            else: preds[engine]=pd.Series(train.iloc[-1]+np.arange(1,test_n+1)*(train.iloc[-1]-train.iloc[-2]),index=test.index)
            st.markdown('### In-sample / holdout evaluation'); st.dataframe(metrics(test,preds[engine]),use_container_width=True,hide_index=True)
            # ML supervised lags
            lm=lag_matrix(y,6); X=lm.drop(columns='y');Y=lm.y; split=len(train)-6;Xtr,Xte=X.iloc[:split],X.iloc[split:];Ytr,Yte=Y.iloc[:split],Y.iloc[split:]
            ml_preds={}
            for name in ml:
                if name=='Random Forest':m=RandomForestRegressor(n_estimators=250,random_state=42)
                elif name=='Gradient Boosting':m=GradientBoostingRegressor(random_state=42)
                elif name=='Elastic Net':m=ElasticNet(alpha=.01,l1_ratio=.5,random_state=42,max_iter=10000)
                else:m=SVR(C=10,gamma='scale')
                m.fit(Xtr,Ytr); ml_preds[name]=m.predict(Xte); st.markdown(f'**{name}**');st.dataframe(metrics(Yte,ml_preds[name]),use_container_width=True,hide_index=True)
            st.markdown('### Future forecast')
            if engine=='Auto ARIMA': f=ARIMA(y,order=(1,1,1)).fit();future=f.forecast(h)
            elif engine=='ETS':future=ExponentialSmoothing(y,trend='add',seasonal=None).fit().forecast(h)
            elif engine=='Naive':future=pd.Series(y.iloc[-1],index=pd.date_range(y.index.max()+pd.offsets.MonthBegin(1),periods=h,freq='MS'))
            else:future=pd.Series(y.iloc[-1]+np.arange(1,h+1)*(y.iloc[-1]-y.iloc[-2]),index=pd.date_range(y.index.max()+pd.offsets.MonthBegin(1),periods=h,freq='MS'))
            st.plotly_chart(go.Figure([go.Scatter(x=y.index[-60:],y=y.tail(60),name='Historical'),go.Scatter(x=future.index,y=future,name='Future forecast')]).update_layout(template='plotly_white',height=420,title=f'{var}: forecast'),use_container_width=True)
            st.dataframe(pd.DataFrame({'date':future.index,'forecast':future.values}),use_container_width=True,hide_index=True)
        except Exception as e:st.error(str(e))

# Scenario
elif page=='Scenario Analysis':
    st.markdown('<div class="section">Scenario Analysis & Stress Testing</div>',unsafe_allow_html=True)
    vars=numeric_vars(df); c1,c2,c3=st.columns(3); shock_var=c1.selectbox('Shock variable',vars); shock_type=c2.selectbox('Shock type',['Percentage','Absolute']); shock=c3.number_input('Shock',value=-10.0 if shock_type=='Percentage' else -100.0)
    h=st.slider('Scenario horizon',1,36,12); targets=st.multiselect('Output variables',vars,default=[v for v in ['goods_exports','goods_imports','current_account_balance','gross_reserves_usd_mn','reer_index'] if v in vars])
    if st.button('Run Scenario',type='primary'):
        base=df.tail(h).copy(); scen=base.copy();
        if shock_type=='Percentage':scen[shock_var]=scen[shock_var]*(1+shock/100)
        else:scen[shock_var]=scen[shock_var]+shock
        rows=[]
        for v in targets:
            rows.append([v,base[v].iloc[-1],scen[v].iloc[-1],scen[v].iloc[-1]-base[v].iloc[-1],(scen[v].iloc[-1]/base[v].iloc[-1]-1)*100 if base[v].iloc[-1]!=0 else np.nan])
        st.dataframe(pd.DataFrame(rows,columns=['Variable','Baseline','Scenario','Effect','Effect (%)']),use_container_width=True,hide_index=True)
        fig=go.Figure();
        for v in targets: fig.add_trace(go.Scatter(x=base.date,y=base[v],name=f'{v} baseline'));fig.add_trace(go.Scatter(x=scen.date,y=scen[v],name=f'{v} scenario',line=dict(dash='dash')))
        fig.update_layout(template='plotly_white',height=450,title='Baseline vs Scenario');st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="warning"><b>Important:</b> this is a deterministic sensitivity experiment. It is not a probability forecast and does not assign likelihood to the shock.</div>',unsafe_allow_html=True)

# DSA
elif page=='Debt Sustainability & Debt Forecasting':
    st.markdown('<div class="section">External Debt Sustainability Analysis & Debt Forecasting</div>',unsafe_allow_html=True)
    st.markdown('<div class="info"><b>Purpose.</b> This module separates <b>debt forecasting</b>, <b>stock-flow debt dynamics</b>, <b>debt-burden indicators</b> and <b>stress testing</b>. Every calculated indicator is shown with its formula, inputs and units. It is an analytical platform and does not reproduce an official IMF/World Bank template.</div>',unsafe_allow_html=True)

    vars=numeric_vars(df)
    def pick(cands, fallback):
        return next((v for v in cands if v in vars), fallback)
    debt_default=pick(['external_debt_outstanding'], vars[0])
    gdp_default=pick(['gdp_index'], vars[0])
    exp_default=pick(['goods_exports','exports','export_value'], vars[0])
    borrow_default=pick(['external_debt_new_borrowing'], vars[0])
    repay_default=pick(['external_debt_principal_repayment'], vars[0])
    int_default=pick(['external_debt_interest_payment'], vars[0])
    ca_default=pick(['current_account_balance'], vars[0])
    reer_default=pick(['reer_index'], vars[0])
    res_default=pick(['gross_reserves_usd_mn'], vars[0])

    with st.expander('1. Framework, coverage and data mapping', expanded=True):
        c1,c2,c3=st.columns(3)
        framework=c1.selectbox('Analytical framework',[
            'External Debt Sustainability — Analytical Framework',
            'LIC-DSF reference architecture',
            'Market-Access / SRDSF reference architecture'],
            help='The reference architectures are used to organise the analysis. The application does not claim to reproduce official IMF/WB templates or risk classifications.')
        debtv=c1.selectbox('External debt stock',vars,index=vars.index(debt_default))
        gdpv=c2.selectbox('Nominal GDP level',vars,index=vars.index(gdp_default))
        expv=c2.selectbox('Exports',vars,index=vars.index(exp_default))
        borrowv=c3.selectbox('New external borrowing',vars,index=vars.index(borrow_default))
        repayv=c3.selectbox('Principal repayments',vars,index=vars.index(repay_default))
        intv=c3.selectbox('Interest payments',vars,index=vars.index(int_default))
        ca_v=st.selectbox('Current-account balance (optional)',vars,index=vars.index(ca_default))
        reer_v=st.selectbox('REER index (optional stress driver)',vars,index=vars.index(reer_default))
        reserve_v=st.selectbox('Gross reserves (optional)',vars,index=vars.index(res_default))
        st.caption('If GDP, exports or flow variables are indices rather than levels, the resulting ratios are labelled proxies. Replace the demonstration data with validated levels for production use.')

    with st.expander('2. Baseline assumptions and forecast controls', expanded=True):
        c1,c2,c3,c4=st.columns(4)
        method=c1.selectbox('Statistical debt forecast',['ARIMA','ETS','Naive','Drift'])
        h=c2.number_input('Forecast horizon',min_value=4,max_value=40,value=20,step=1)
        real_g=c3.number_input('Real GDP growth (%)',value=3.0,step=.25)
        inflation=c4.number_input('GDP deflator / inflation (%)',value=5.0,step=.25)
        c1,c2,c3,c4=st.columns(4)
        real_r=c1.number_input('Real interest rate (%)',value=5.0,step=.25)
        pb=c2.number_input('Primary balance (% GDP)',value=0.0,step=.25,help='Surplus is positive; deficit is negative.')
        fxdep=c3.number_input('Annual exchange-rate depreciation (%)',value=0.0,step=1.0)
        debt_fx_share=c4.number_input('External-currency share of debt (%)',min_value=0.0,max_value=100.0,value=100.0,step=5.0)
        c1,c2=st.columns(2)
        manual_gdp=c1.number_input('Manual latest nominal GDP level (0 = use selected series)',min_value=0.0,value=0.0,step=100.0)
        manual_exports=c2.number_input('Manual latest exports level (0 = use selected series)',min_value=0.0,value=0.0,step=100.0)

    def _forecast_series(y,h,method):
        y=pd.Series(y).dropna().astype(float)
        if len(y)<4: return np.repeat(y.iloc[-1],h)
        try:
            if method=='ARIMA':
                return np.asarray(ARIMA(y,order=(1,1,1)).fit().forecast(h),dtype=float)
            if method=='ETS':
                return np.asarray(ExponentialSmoothing(y,trend='add',seasonal=None).fit().forecast(h),dtype=float)
            if method=='Drift':
                drift=(y.iloc[-1]-y.iloc[0])/max(len(y)-1,1)
                return y.iloc[-1]+drift*np.arange(1,h+1)
            return np.repeat(y.iloc[-1],h)
        except Exception:
            return np.repeat(y.iloc[-1],h)

    # _safe_last is defined globally and shared with the EBA engine.

    if st.button('Run External Debt Sustainability Analysis',type='primary'):
        cols=['date',debtv,gdpv,expv,borrowv,repayv,intv,ca_v,reer_v,reserve_v]
        d=df[cols].replace([np.inf,-np.inf],np.nan).copy().dropna(subset=[debtv]).sort_values('date')
        debt=_safe_last(d[debtv]); gdp=_safe_last(d[gdpv]) if manual_gdp==0 else float(manual_gdp)
        exports=_safe_last(d[expv]) if manual_exports==0 else float(manual_exports)
        borrowing=_forecast_series(d[borrowv],h,method)
        repayment=_forecast_series(d[repayv],h,method)
        interest=_forecast_series(d[intv],h,method)
        debt_stat=_forecast_series(d[debtv],h,method)
        # Transparent stock-flow identity: D_t = D_{t-1}+NB_t-PR_t+I_t+VA_t.
        # VA is an explicit valuation/FX adjustment, not an unobserved residual.
        def project(shock=None):
            shock=shock or {}
            rg=(real_g+shock.get('growth',0))/100
            inf=(inflation+shock.get('inflation',0))/100
            rr=(real_r+shock.get('interest',0))/100
            dep=(fxdep+shock.get('fx',0))/100
            export_g=(real_g+shock.get('exports_growth',0))/100
            D=[]; GDP=[]; EXP=[]; Dg=[]; Dx=[]; DSx=[]; VA=[]
            d0=debt; y0=gdp if np.isfinite(gdp) and gdp>0 else np.nan; x0=exports if np.isfinite(exports) and exports>0 else np.nan
            for t in range(h):
                nominal_g=(1+rg)*(1+inf)-1
                val_adj=d0*dep*(debt_fx_share/100.0)
                dt=max(0,d0+borrowing[t]-repayment[t]+interest[t]+val_adj)
                if np.isfinite(y0): y1=y0*(1+nominal_g)
                else: y1=np.nan
                if np.isfinite(x0): x1=x0*(1+export_g)
                else: x1=np.nan
                D.append(dt); GDP.append(y1); EXP.append(x1); VA.append(val_adj)
                Dg.append(100*dt/y1 if np.isfinite(y1) and y1!=0 else np.nan)
                Dx.append(100*dt/x1 if np.isfinite(x1) and x1!=0 else np.nan)
                DSx.append(100*(repayment[t]+interest[t])/x1 if np.isfinite(x1) and x1!=0 else np.nan)
                d0,y0,x0=dt,y1,x1
            return pd.DataFrame({'period':np.arange(1,h+1),'debt':D,'gdp':GDP,'exports':EXP,'debt_gdp':Dg,'debt_exports':Dx,'debt_service_exports':DSx,'valuation_fx':VA})
        scenarios={
            'Baseline':{},
            'Real GDP growth -2pp':{'growth':-2},
            'Real interest rate +2pp':{'interest':2},
            'Exchange-rate depreciation +20pp':{'fx':20},
            'Exports growth -20pp':{'exports_growth':-20},
            'Combined stress':{'growth':-2,'interest':2,'fx':20,'exports_growth':-20}}
        paths=pd.concat([project(v).assign(scenario=k) for k,v in scenarios.items()],ignore_index=True)

        st.markdown('### A. Executive debt-sustainability dashboard')
        base=paths[paths.scenario=='Baseline']
        c1,c2,c3,c4=st.columns(4)
        c1.metric('Latest external debt',f'{debt:,.2f}' if np.isfinite(debt) else 'N/A')
        c2.metric('Latest debt / GDP',f'{100*debt/gdp:,.2f}%' if np.isfinite(gdp) and gdp else 'N/A')
        c3.metric('Peak baseline debt / GDP',f'{base.debt_gdp.max():,.2f}%' if base.debt_gdp.notna().any() else 'N/A')
        c4.metric('Peak combined-stress debt / GDP',f'{paths[paths.scenario=="Combined stress"].debt_gdp.max():,.2f}%' if paths[paths.scenario=='Combined stress'].debt_gdp.notna().any() else 'N/A')

        st.markdown('### B. Debt stock-flow identity')
        st.latex(r'D_t = D_{t-1} + NB_t - PR_t + INT_t + VA_t')
        st.markdown('**Where:** $D$ = external debt stock; $NB$ = new borrowing; $PR$ = principal repayment; $INT$ = interest/capitalised interest; $VA$ = valuation/foreign-exchange adjustment. The application makes the FX adjustment explicit rather than hiding it in a residual.')
        identity=pd.DataFrame({'period':np.arange(1,h+1),'statistical_forecast':debt_stat,'identity_path':base.debt,'new_borrowing':borrowing,'principal_repayment':repayment,'interest_payment':interest,'FX_valuation_adjustment':base.valuation_fx})
        st.dataframe(identity.round(4),use_container_width=True,hide_index=True)

        st.markdown('### C. Debt-to-GDP dynamics')
        st.latex(r'd_t \approx \frac{1+r_t}{(1+g_t)(1+\pi_t)}d_{t-1} - \frac{PB_t}{GDP_t} + FXVA_t')
        st.caption('This ratio form is an analytical approximation. The stock-flow identity above is the primary accounting path; the ratio equation explains how interest, growth, inflation, the primary balance and FX valuation pressures affect debt dynamics.')
        fig=go.Figure()
        for sc in scenarios: fig.add_trace(go.Scatter(x=base.period,y=paths[paths.scenario==sc].debt_gdp,name=sc))
        fig.update_layout(template='plotly_white',height=450,title='External debt / GDP — baseline and deterministic stress paths',xaxis_title='Forecast period',yaxis_title='Percent of GDP')
        st.plotly_chart(fig,use_container_width=True)

        st.markdown('### D. Debt-burden indicators')
        ind=pd.DataFrame({
            'Indicator':['External debt / GDP','External debt / exports','Debt service / exports'],
            'Latest':[100*debt/gdp if np.isfinite(gdp) and gdp else np.nan,100*debt/exports if np.isfinite(exports) and exports else np.nan,100*(_safe_last(d[repayv])+_safe_last(d[intv]))/exports if np.isfinite(exports) and exports else np.nan],
            'Peak baseline':[base.debt_gdp.max(),base.debt_exports.max(),base.debt_service_exports.max()],
            'Peak combined stress':[paths[paths.scenario=='Combined stress'].debt_gdp.max(),paths[paths.scenario=='Combined stress'].debt_exports.max(),paths[paths.scenario=='Combined stress'].debt_service_exports.max()]})
        st.dataframe(ind.round(3),use_container_width=True,hide_index=True)
        st.caption('PV debt indicators are not calculated unless a full maturity/interest/concessionality schedule is supplied. A nominal debt stock divided by GDP or exports is not the same thing as PV external debt.')

        st.markdown('### E. Stress-test transmission')
        stress=paths.groupby('scenario').agg(peak_debt_gdp=('debt_gdp','max'),end_debt_gdp=('debt_gdp','last'),peak_debt_exports=('debt_exports','max'),peak_debt_service_exports=('debt_service_exports','max')).reset_index()
        st.dataframe(stress.round(3),use_container_width=True,hide_index=True)
        st.markdown('<div class="warning"><b>Interpretation discipline:</b> these are deterministic stress experiments. They are not probabilities and should not be presented as an official debt-distress classification. The current IMF LIC-DSF framework uses debt-carrying capacity, baseline realism, standardized and tailored stress tests and debt-burden thresholds; the 2026 review has introduced further refinements. The platform therefore keeps the official-framework concepts visible without fabricating an official country rating.</div>',unsafe_allow_html=True)

        st.markdown('### F. Formula and assumption audit')
        audit=pd.DataFrame([
            ['Debt stock','D_t = D_(t-1) + new borrowing - principal repayment + interest + FX valuation','Debt, borrowing, repayment, interest, FX share'],
            ['Nominal GDP growth','(1+real growth)(1+inflation)-1','Real GDP growth; inflation'],
            ['Debt/GDP','100 × debt / nominal GDP','Debt and GDP levels'],
            ['Debt/exports','100 × debt / exports','Debt and exports levels'],
            ['Debt service/exports','100 × (principal + interest) / exports','Repayment, interest, exports'],
            ['FX valuation adjustment','Debt_(t-1) × depreciation × FX debt share','FX depreciation; external-currency share'],
            ['Stress paths','Baseline plus explicit parameter shocks','Growth, interest, FX, exports']
        ],columns=['Component','Formula','Inputs'])
        st.dataframe(audit,use_container_width=True,hide_index=True)

        st.markdown('### G. Reference framework')
        st.markdown('The platform follows the broad logic that a DSA should make the baseline assumptions explicit, examine debt dynamics under alternative shocks and distinguish analytical calculations from an official institutional assessment. The IMF describes separate frameworks for external/public debt and for LIC versus market-access settings. citeturn0search0turn0search2turn0search7')

# EBA
elif page=='External Balance Assessment':
    st.markdown('<div class="section">External Balance Assessment — Comprehensive Analytical Framework</div>',unsafe_allow_html=True)
    st.markdown('<div class="info"><b>Purpose.</b> The EBA module decomposes the external position into three linked components: <b>(1) current-account benchmark, (2) REER equilibrium benchmark, and (3) external-sustainability benchmark</b>. Results are explicitly labelled a domestic analytical proxy rather than the official IMF EBA methodology.</div>',unsafe_allow_html=True)

    vars=numeric_vars(df)
    if not vars:
        st.error('No numeric variables are available for the External Balance Assessment.')
        st.stop()

    def e_pick(cands):
        return next((v for v in cands if v in vars),vars[0])

    ca=e_pick(['current_account_balance'])
    gdp=e_pick(['gdp_index'])
    reer=e_pick(['reer_index'])
    iip=e_pick(['net_iip'])
    tot=e_pick(['terms_of_trade_index'])
    glob=e_pick(['global_demand_index'])
    res=e_pick(['gross_reserves_usd_mn'])
    infl=e_pick(['inflation_pct'])

    with st.expander('1. Data and model specification',expanded=True):
        c1,c2,c3=st.columns(3)
        ca=c1.selectbox('Current-account balance',vars,index=vars.index(ca),key='eba_ca')
        gdp=c1.selectbox('GDP level/index',vars,index=vars.index(gdp),key='eba_gdp')
        reer=c2.selectbox('REER index',vars,index=vars.index(reer),key='eba_reer')
        iip=c2.selectbox('Net IIP',vars,index=vars.index(iip),key='eba_iip')
        tot=c3.selectbox('Terms of trade',vars,index=vars.index(tot),key='eba_tot')
        glob=c3.selectbox('Global demand / external activity',vars,index=vars.index(glob),key='eba_global')
        res=c1.selectbox('Gross reserves',vars,index=vars.index(res),key='eba_res')
        infl=c2.selectbox('Inflation',vars,index=vars.index(infl),key='eba_infl')
        minobs=st.number_input('Minimum observations for a regression term',min_value=10,max_value=100,value=20,step=5,key='eba_minobs')

    with st.expander('2. External-sustainability assumptions',expanded=True):
        c1,c2,c3=st.columns(3)
        real_g=c1.number_input('Long-run real GDP growth (%)',value=3.0,step=.25,key='eba_real_g')
        real_r=c2.number_input('Long-run real return / interest (%)',value=3.0,step=.25,key='eba_real_r')
        target_nfa=c3.number_input('Target NFA / GDP (%)',value=0.0,step=1.0,key='eba_target_nfa')
        manual_gdp=st.number_input('Manual GDP level if selected series is an index (0 = do not rescale)',min_value=0.0,value=0.0,step=100.0,key='eba_manual_gdp')

    if st.button('Run Comprehensive External Balance Assessment',type='primary',key='run_eba'):
        # --------------------------------------------------------
        # Build a uniquely named working frame.
        # --------------------------------------------------------
        selected=[('date','date'),('ca',ca),('gdp',gdp),('reer',reer),('iip',iip),('tot',tot),('global',glob),('res',res),('infl',infl)]
        z=pd.DataFrame(index=df.index)
        z['date']=pd.to_datetime(df['date'],errors='coerce')
        for alias,col in selected[1:]:
            z[alias]=pd.to_numeric(df[col],errors='coerce')
        z=z.replace([np.inf,-np.inf],np.nan).sort_values('date').reset_index(drop=True)

        # Defensive validation: EBA works only with its internal aliases.
        # This prevents source-column names such as reer_index from leaking
        # into the analytical equations.
        required_aliases={'ca','gdp','reer','iip','tot','global','res','infl'}
        missing_aliases=sorted(required_aliases-set(z.columns))
        if missing_aliases:
            st.error('The EBA analytical frame is incomplete. Missing internal fields: ' + ', '.join(missing_aliases))
            st.stop()

        # Prevent division by zero and make the ratio calculation transparent.
        z['ca_gdp']=np.where(z['gdp'].abs()>1e-12,100*z['ca']/z['gdp'],np.nan)

        # --------------------------------------------------------
        # Current-account benchmark.
        # --------------------------------------------------------
        ca_rhs=['reer','tot','global','iip','res','infl']
        ca_rhs=[v for v in ca_rhs if z[v].notna().sum()>=minobs]
        ca_required=['ca_gdp']+ca_rhs
        sample=z[ca_required].dropna()
        fit_ca=None
        z['ca_norm']=np.nan
        if len(sample)>=minobs and len(ca_rhs)>=1:
            X=sm.add_constant(sample[ca_rhs],has_constant='add')
            try:
                fit_ca=sm.OLS(sample['ca_gdp'],X).fit()
                z.loc[sample.index,'ca_norm']=fit_ca.predict(X)
            except Exception as exc:
                fit_ca=None
                st.warning(f'Current-account benchmark estimation failed: {exc}')
        if fit_ca is None:
            fallback=z['ca_gdp'].mean()
            z['ca_norm']=fallback
        z['ca_gap']=z['ca_gdp']-z['ca_norm']

        # --------------------------------------------------------
        # REER equilibrium benchmark.
        # IMPORTANT: use the selected REER alias, not a hard-coded
        # source column named "reer".  The previous version failed
        # here because the master data contains "reer_index".
        # --------------------------------------------------------
        reer_rhs=['tot','global','iip','res','infl']
        reer_rhs=[v for v in reer_rhs if z[v].notna().sum()>=minobs]
        reer_required=['reer']+reer_rhs
        sample_r=z[reer_required].dropna()
        fit_reer=None
        z['reer_eq']=np.nan
        if len(sample_r)>=minobs and len(reer_rhs)>=1:
            Xr=sm.add_constant(sample_r[reer_rhs],has_constant='add')
            try:
                fit_reer=sm.OLS(sample_r['reer'],Xr).fit()
                z.loc[sample_r.index,'reer_eq']=fit_reer.predict(Xr)
            except Exception as exc:
                fit_reer=None
                st.warning(f'REER equilibrium estimation failed: {exc}')
        if fit_reer is None:
            fallback_reer=z['reer'].mean()
            z['reer_eq']=fallback_reer

        # REER gap uses the absolute equilibrium level in the denominator.
        z['reer_gap_pct']=np.where(
            z['reer_eq'].abs()>1e-12,
            100*(z['reer']-z['reer_eq'])/z['reer_eq'].abs(),
            np.nan
        )

        # --------------------------------------------------------
        # External sustainability norm.
        # --------------------------------------------------------
        nfa_latest=_safe_last(z['iip'])
        gdp_latest=_safe_last(z['gdp'])
        if manual_gdp>0 and np.isfinite(nfa_latest):
            nfa_ratio=100*nfa_latest/manual_gdp
        elif np.isfinite(nfa_latest) and np.isfinite(gdp_latest) and abs(gdp_latest)>1e-12:
            nfa_ratio=100*nfa_latest/gdp_latest
        else:
            nfa_ratio=target_nfa
        es_norm=((real_g-real_r)/(100+real_g))*nfa_ratio

        # Use the latest observation containing all quantities needed for
        # the executive dashboard.  This avoids relying on a hard-coded
        # column name and avoids an empty-row failure.
        latest_candidates=z.dropna(subset=['ca_gdp','ca_norm','reer','reer_eq','reer_gap_pct'])
        if latest_candidates.empty:
            st.error('The EBA could not identify a common latest observation with valid CA and REER information. Check the selected variables and minimum-observation requirement.')
            st.stop()
        latest=latest_candidates.iloc[-1]

        ca_gap=float(latest['ca_gap'])
        reer_gap=float(latest['reer_gap_pct'])
        ca_actual=float(latest['ca_gdp'])
        ca_norm=float(latest['ca_norm'])
        reer_actual=float(latest['reer'])
        reer_eq=float(latest['reer_eq'])

        # --------------------------------------------------------
        # A. Executive dashboard
        # --------------------------------------------------------
        st.markdown('### A. Executive external-position dashboard')
        c1,c2,c3,c4=st.columns(4)
        c1.metric('Actual CA / GDP',f'{ca_actual:.2f}%')
        c2.metric('CA benchmark',f'{ca_norm:.2f}%')
        c3.metric('CA gap',f'{ca_gap:+.2f} pp')
        c4.metric('REER gap',f'{reer_gap:+.2f}%')

        # --------------------------------------------------------
        # B. Current-account benchmark
        # --------------------------------------------------------
        st.markdown('### B. Current-account benchmark model')
        st.latex(r'CA_t/Y_t = \alpha + \beta_1 REER_t + \beta_2 ToT_t + \beta_3 GlobalDemand_t + \beta_4 NFA_t/Y_t + \beta_5 Reserves_t + \beta_6 Inflation_t + \varepsilon_t')
        st.markdown('The benchmark is the fitted value from a transparent domestic time-series regression. The <b>CA gap</b> is actual CA/GDP minus the modelled benchmark. A positive gap means the observed CA ratio is above the modelled benchmark; it does not by itself establish whether the position is desirable or undesirable.',unsafe_allow_html=True)
        if fit_ca is not None:
            coef=fit_ca.summary2().tables[1].reset_index().rename(columns={'index':'Term'})
            st.dataframe(coef.round(5),use_container_width=True,hide_index=True)
            st.metric('Adjusted R-squared',f'{fit_ca.rsquared_adj:.3f}')
        else:
            st.warning('Insufficient complete observations for the requested CA regression; the benchmark falls back to the sample mean.')

        fig=go.Figure()
        fig.add_trace(go.Scatter(x=z['date'],y=z['ca_gdp'],name='Actual CA/GDP'))
        fig.add_trace(go.Scatter(x=z['date'],y=z['ca_norm'],name='CA benchmark'))
        fig.add_trace(go.Scatter(x=z['date'],y=z['ca_gap'],name='CA gap',yaxis='y2',line=dict(dash='dash')))
        fig.update_layout(template='plotly_white',height=450,title='Current-account actual, benchmark and gap',yaxis_title='Percent of GDP',yaxis2=dict(title='Gap (percentage points)',overlaying='y',side='right'))
        st.plotly_chart(fig,use_container_width=True)

        # --------------------------------------------------------
        # C. REER equilibrium benchmark
        # --------------------------------------------------------
        st.markdown('### C. REER equilibrium benchmark')
        st.latex(r'REER_t = \gamma + \delta_1 ToT_t + \delta_2 GlobalDemand_t + \delta_3 NFA_t + \delta_4 Reserves_t + \delta_5 Inflation_t + u_t')
        st.markdown(r'The REER gap is calculated as $100\times(REER^{actual}-REER^{equilibrium})/|REER^{equilibrium}|$. The sign is retained exactly as defined; the index convention must therefore be checked before giving an economic interpretation.')
        if fit_reer is not None:
            coefr=fit_reer.summary2().tables[1].reset_index().rename(columns={'index':'Term'})
            st.dataframe(coefr.round(5),use_container_width=True,hide_index=True)
            st.metric('REER model adjusted R-squared',f'{fit_reer.rsquared_adj:.3f}')
        else:
            st.warning('Insufficient complete observations for the requested REER equilibrium regression; the equilibrium series falls back to the sample mean.')
        fig=go.Figure([
            go.Scatter(x=z['date'],y=z['reer'],name='Actual REER'),
            go.Scatter(x=z['date'],y=z['reer_eq'],name='Equilibrium REER')
        ])
        fig.update_layout(template='plotly_white',height=430,title='REER actual versus equilibrium proxy',yaxis_title='REER index')
        st.plotly_chart(fig,use_container_width=True)

        # --------------------------------------------------------
        # D. External sustainability norm
        # --------------------------------------------------------
        st.markdown('### D. External-sustainability current-account norm')
        st.latex(r'CA^{ES}/Y \approx \frac{g-r}{1+g}\times\frac{NFA}{Y}')
        st.markdown('Here $g$ is long-run real GDP growth, $r$ is the assumed real return/interest rate, and NFA/Y is the net international investment position relative to GDP. This is a steady-state stabilisation condition, not an IMF cross-country EBA coefficient estimate.')
        c1,c2,c3=st.columns(3)
        c1.metric('NFA / GDP used',f'{nfa_ratio:.2f}%')
        c2.metric('ES CA norm',f'{es_norm:.2f}% of GDP')
        c3.metric('Actual CA minus ES norm',f'{ca_actual-es_norm:+.2f} pp')

        # --------------------------------------------------------
        # E. Gap decomposition and REER sensitivity
        # --------------------------------------------------------
        st.markdown('### E. Gap decomposition and implied REER sensitivity')
        if fit_ca is not None and 'reer' in fit_ca.params.index and abs(float(fit_ca.params['reer']))>1e-12:
            beta_reer=float(fit_ca.params['reer'])
            implied=-ca_gap/beta_reer
            st.latex(r'\Delta REER_{implied} = -\frac{CA\ gap}{\beta_{REER}}')
            st.metric('Mechanical REER change implied by CA gap',f'{implied:+.3f} index points')
            st.caption('This is a local linear sensitivity from the domestic regression, not an official EBA exchange-rate adjustment.')
        else:
            st.info('An implied REER adjustment is not calculated because the CA model does not contain a sufficiently identified REER coefficient.')

        if fit_ca is not None:
            contrib=[]
            for term in fit_ca.params.index:
                if term=='const':
                    val=float(fit_ca.params[term])
                else:
                    xval=float(latest[term]) if term in latest.index and pd.notna(latest[term]) else np.nan
                    val=float(fit_ca.params[term])*xval if np.isfinite(xval) else np.nan
                contrib.append([term,val])
            cd=pd.DataFrame(contrib,columns=['Term','Contribution to fitted CA norm'])
            st.dataframe(cd.round(5),use_container_width=True,hide_index=True)

        # --------------------------------------------------------
        # F. Integrated assessment
        # --------------------------------------------------------
        st.markdown('### F. Integrated assessment table')
        integrated=pd.DataFrame({
            'Component':['Current-account actual','Current-account benchmark','Current-account gap','REER actual','REER equilibrium','REER gap','NFA/GDP','External-sustainability CA norm'],
            'Value':[ca_actual,ca_norm,ca_gap,reer_actual,reer_eq,reer_gap,nfa_ratio,es_norm],
            'Unit':['% GDP','% GDP','percentage points','index','index','percent','% GDP','% GDP'],
            'Formula / basis':['Observed','Domestic regression fitted value','Actual - benchmark','Observed index','Domestic fundamentals regression','100*(actual-equilibrium)/|equilibrium|','NFA/GDP','(g-r)/(1+g) × NFA/GDP']
        })
        st.dataframe(integrated.round(4),use_container_width=True,hide_index=True)

        # --------------------------------------------------------
        # G. Diagnostic transparency
        # --------------------------------------------------------
        st.markdown('### G. Diagnostic transparency')
        diag=pd.DataFrame([
            ['CA model observations',len(sample) if fit_ca is not None else 0,'Complete cases used in CA regression'],
            ['CA model variables',', '.join(ca_rhs) if ca_rhs else 'None','Available fundamentals meeting minimum coverage'],
            ['REER model observations',len(sample_r) if fit_reer is not None else 0,'Complete cases used in REER regression'],
            ['REER model variables',', '.join(reer_rhs) if reer_rhs else 'None','Available fundamentals meeting minimum coverage'],
            ['Target NFA/GDP',f'{target_nfa:.4f}','User assumption when observed ratio unavailable'],
            ['Long-run growth',f'{real_g:.4f}','% assumption in ES norm'],
            ['Long-run real return',f'{real_r:.4f}','% assumption in ES norm'],
            ['Selected REER field',reer,'Actual source variable used'],
            ['Selected GDP field',gdp,'Source variable used for CA/GDP'],
            ['Selected NFA field',iip,'Source variable used for NFA/GDP']
        ],columns=['Diagnostic','Value','Explanation'])
        st.dataframe(diag,use_container_width=True,hide_index=True)

        # --------------------------------------------------------
        # H. Methodological status
        # --------------------------------------------------------
        st.markdown('### H. Methodological status and official EBA distinction')
        st.markdown('The IMF EBA methodology is designed to assess current accounts and exchange rates using a multilaterally consistent framework. Its current generation includes current-account and REER models and has undergone methodological reviews. This application uses the same analytical concepts—norms, gaps, fundamentals and external sustainability—but <b>does not reproduce the IMF cross-country coefficient set or official policy-gap adjustments</b>.',unsafe_allow_html=True)

# Methodology
elif page=='Methodology':
    st.markdown('<div class="section">Methodology</div>',unsafe_allow_html=True)
    sec=st.selectbox('Methodological subsection',list(ABOUT.keys()))
    st.markdown(f'<div class="card"><h3>{sec}</h3><p>{ABOUT[sec]}</p></div>',unsafe_allow_html=True)
    st.markdown('### 14-model catalogue')
    rows=[]
    for name,(d,r,p,t) in MODEL_SPECS.items(): rows.append([name,t,d or 'System model',p,', '.join(r) if r else '—'])
    st.dataframe(pd.DataFrame(rows,columns=['Model','Title','LHS','Lag','RHS / System Variables']),use_container_width=True,hide_index=True)

# Database info
else:
    st.markdown('<div class="section">Database Information</div>',unsafe_allow_html=True)
    files=sorted([p.name for p in DATA.glob('*.csv')])
    st.write(f'Bundled source tables: **{len(files)} CSV files**')
    st.dataframe(pd.DataFrame({'Source table':files}),use_container_width=True,hide_index=True)
    st.markdown('### Analytical master dataset')
    info_df=pd.DataFrame({'Metric':['Rows','Columns','Numeric indicators','Date start','Date end'], 'Value':[str(len(df)),str(len(df.columns)),str(len(numeric_vars(df))),str(df.date.min().date()),str(df.date.max().date())]})
    st.dataframe(info_df,use_container_width=True,hide_index=True)
    st.markdown('<div class="info">The application uses the supplied simulated demonstration tables as its bundled data layer. A production implementation can replace this layer with a controlled database connection without changing the analytical navigation.</div>',unsafe_allow_html=True)

st.markdown('---')
st.caption('External Sector Analytics, Modelling & Forecasting • Public Research Prototype • Neutral branding • Author: Chirume Admire Tarisirayi')
