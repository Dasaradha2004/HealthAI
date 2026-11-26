import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity, ArrowRight, Beaker, CheckCircle2, ChevronDown, CircleAlert,
  ClipboardList, Database, HeartPulse, Info, Microscope, ShieldCheck,
  Sparkles, Stethoscope, Waves,
} from "lucide-react";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const CKD_FIELDS = [
  { section: "Patient Information", fields: [
    ["age", "Age", 18, 120, 45, "years"],
    ["blood_pressure", "Blood pressure", 40, 250, 80, "mmHg"],
    ["hypertension", "Hypertension", "no", "yes", "no", ["no", "yes"]],
    ["diabetes_mellitus", "Diabetes mellitus", "no", "yes", "no", ["no", "yes"]],
  ] },
  { section: "Urine & Kidney Indicators", fields: [
    ["specific_gravity", "Specific gravity", 1.0, 1.1, 1.02, ""],
    ["albumin", "Albumin", 0, 5, 0, "0–5"],
    ["sugar", "Urine sugar", 0, 5, 0, "0–5"],
    ["red_blood_cells", "Red blood cells", "normal", "abnormal", "normal", ["normal", "abnormal"]],
    ["pus_cell", "Pus cell", "normal", "abnormal", "normal", ["normal", "abnormal"]],
    ["pus_cell_clumps", "Pus cell clumps", "notpresent", "present", "notpresent", ["notpresent", "present"]],
    ["bacteria", "Bacteria", "notpresent", "present", "notpresent", ["notpresent", "present"]],
  ] },
  { section: "Blood Chemistry", fields: [
    ["blood_glucose_random", "Random blood glucose", 40, 600, 110, "mg/dL"],
    ["blood_urea", "Blood urea", 1, 300, 30, "mg/dL"],
    ["serum_creatinine", "Serum creatinine", 0.1, 30, 1, "mg/dL"],
    ["sodium", "Sodium", 80, 200, 140, "mEq/L"],
    ["potassium", "Potassium", 1, 15, 4.5, "mEq/L"],
  ] },
  { section: "Blood Cell Measurements", fields: [
    ["hemoglobin", "Hemoglobin", 1, 25, 13, "g/dL"],
    ["packed_cell_volume", "Packed cell volume", 5, 80, 40, "%"],
    ["red_blood_cell_count", "Red blood cell count", 0.5, 15, 4.5, "million/µL"],
  ] },
];
const DIABETES_FIELDS = [
  { section: "Patient & Clinical Measurements", fields: [
    ["pregnancies", "Pregnancies", 0, 20, 1, "count"],
    ["glucose", "Glucose", 40, 300, 110, "mg/dL"],
    ["blood_pressure", "Blood pressure", 30, 200, 72, "mmHg"],
    ["skin_thickness", "Skin thickness", 0, 100, 20, "mm"],
    ["insulin", "Insulin", 0, 1000, 80, "µU/mL"],
    ["bmi", "BMI", 10, 70, 27, "kg/m²"],
    ["diabetes_pedigree_function", "Diabetes pedigree function", 0, 5, 0.5, ""],
    ["age", "Age", 18, 120, 35, "years"],
  ] },
];
const PIPELINE = ["User input", "Preprocessing", "ML prediction", "SHAP / LIME", "Plain-language explanation", "Saved history"];

function initialValues(groups) {
  return Object.fromEntries(groups.flatMap(group => group.fields.map(([key, , , , value]) => [key, String(value)])));
}

function RiskPill({ level }) {
  return <span className={`risk-pill ${level?.toLowerCase() || "low"}`}>{level || "—"}</span>;
}

function Header({ page, onNavigate }) {
  const tabs = ["Dashboard", "CKD", "Diabetes", "History"];
  return <>
    <header className="hero">
      <div className="hero-inner">
        <a className="brand" href="#" onClick={event => { event.preventDefault(); onNavigate("Dashboard"); }}>
          <span className="brand-mark"><HeartPulse size={23} /></span>
          <span><strong>HealthAI</strong><small>CLINICAL RISK INTELLIGENCE</small></span>
        </a>
        <div className="hero-copy">
          <div className="eyebrow"><span className="live-dot" /> EXPLAINABLE HEALTH ANALYTICS</div>
          <h1>Explainable AI for<br /><span>kidney & diabetes risk.</span></h1>
          <p>HealthAI early-risk assessment for Chronic Kidney Disease and Diabetes.</p>
        </div>
        <div className="hero-orb orb-one" /><div className="hero-orb orb-two" />
        <div className="hero-decoration"><Activity size={108} strokeWidth={1} /></div>
      </div>
    </header>
    <nav className="nav-wrap">
      <div className="nav-inner">
        <div className="nav-tabs">{tabs.map(tab =>
          <button key={tab} onClick={() => onNavigate(tab)} className={`nav-tab ${page === tab ? "active" : ""}`}>
            {tab === "Dashboard" ? <Activity size={16} /> : tab === "CKD" ? <Waves size={16} /> : tab === "Diabetes" ? <DropletIcon /> : <ClipboardList size={16} />}
            {tab === "CKD" ? "CKD" : tab}
          </button>)}</div>
        <div className="nav-status"><span className="status-dot" /> XAI system online</div>
      </div>
    </nav>
  </>;
}

function DropletIcon() {
  return <Beaker size={16} />;
}

function Dashboard({ health, history, navigate }) {
  return <main className="page-content">
    <div className="page-heading">
      <div><div className="eyebrow muted">YOUR OVERVIEW</div><h2>Assessment dashboard</h2><p>Explore risk estimates with model transparency at every step.</p></div>
      <span className="date-chip"><ShieldCheck size={15} /> Decision support demo</span>
    </div>
    <section className="overview-grid">
      <article className="overview-card kidney-card">
        <div className="card-icon kidney-icon"><Waves size={21} /></div>
        <span className="card-kicker">KIDNEY HEALTH</span><h3>Chronic Kidney Disease</h3>
        <p>Review clinical indicators and explore an explainable CKD risk estimate.</p>
        <div className="mini-metrics"><span><strong>3</strong> ensemble models</span><span><CheckCircle2 size={15} /> SHAP + LIME</span></div>
        <button className="text-link" onClick={() => navigate("CKD")}>Start CKD assessment <ArrowRight size={16} /></button>
      </article>
      <article className="overview-card diabetes-card">
        <div className="card-icon diabetes-icon"><Beaker size={21} /></div>
        <span className="card-kicker">METABOLIC HEALTH</span><h3>Diabetes</h3>
        <p>Explore the factors contributing to a model-based diabetes risk estimate.</p>
        <div className="mini-metrics"><span><strong>3</strong> ensemble models</span><span><CheckCircle2 size={15} /> SHAP + LIME</span></div>
        <button className="text-link" onClick={() => navigate("Diabetes")}>Start diabetes assessment <ArrowRight size={16} /></button>
      </article>
    </section>
    <section className="pipeline-panel panel">
      <div className="section-title"><div><div className="eyebrow muted">HOW IT WORKS</div><h3>Transparent assessment pipeline</h3></div><span className="pipeline-badge"><Sparkles size={14} /> XAI enabled</span></div>
      <div className="pipeline">
        {PIPELINE.map((step, index) => <div className="pipeline-step" key={step}>
          <div className="pipeline-node">{index === 0 ? <Stethoscope size={18} /> : index === 1 ? <Beaker size={18} /> : index === 2 ? <Activity size={18} /> : index === 3 ? <Microscope size={18} /> : index === 4 ? <Sparkles size={18} /> : <Database size={18} />}</div>
          <span>{step}</span>{index < PIPELINE.length - 1 && <ArrowRight className="pipeline-arrow" size={16} />}
        </div>)}
      </div>
    </section>
    <div className="dashboard-bottom">
      <section className="panel model-panel"><div className="section-title"><div><div className="eyebrow muted">PREDICTION ENGINE</div><h3>Ensemble model status</h3></div><span className={`status-label ${health?.models?.ckd && health?.models?.diabetes ? "ready" : ""}`}><span />{health?.models?.ckd && health?.models?.diabetes ? "Ready" : "Models needed"}</span></div>
        <div className="model-list">{[["XGBoost", "Gradient boosting"], ["Random Forest", "Bagged trees"], ["LightGBM", "Gradient boosting"]].map(([name, text]) => <div className="model-row" key={name}><span className="model-symbol"><Activity size={15} /></span><span><strong>{name}</strong><small>{text}</small></span><CheckCircle2 size={17} className="model-check" /></div>)}</div>
      </section>
      <section className="panel recent-panel"><div className="section-title"><div><div className="eyebrow muted">RECENT ACTIVITY</div><h3>Latest assessments</h3></div><button className="subtle-link" onClick={() => navigate("History")}>View all <ArrowRight size={14} /></button></div>
        {history.length ? history.slice(0, 3).map(item => <div className="recent-row" key={item.id}><span className={`recent-icon ${item.condition}`}><HeartPulse size={16} /></span><span><strong>{item.condition === "ckd" ? "CKD assessment" : "Diabetes assessment"}</strong><small>{new Date(item.created_at).toLocaleString()}</small></span><RiskPill level={item.risk_level} /></div>) : <div className="empty-state compact"><ClipboardList size={22} /><span>Your saved assessments will appear here.</span></div>}
      </section>
    </div>
    <div className="notice"><Info size={16} /><span><strong>For demonstration and educational use only.</strong> This system is not clinically validated and cannot diagnose disease. Synthetic training data may be in use. Consult a qualified healthcare professional for medical concerns.</span></div>
  </main>;
}

function AssessmentForm({ kind, onResult }) {
  const groups = kind === "ckd" ? CKD_FIELDS : DIABETES_FIELDS;
  const [values, setValues] = useState(() => initialValues(groups));
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const title = kind === "ckd" ? "Chronic Kidney Disease" : "Diabetes";

  async function submit(event) {
    event.preventDefault();
    setError("");
    const body = Object.fromEntries(Object.entries(values).map(([key, value]) => [
      key, Number.isNaN(Number(value)) ? value : Number(value),
    ]));
    for (const group of groups) for (const [key, label, min, max, , unit] of group.fields) {
      if (typeof min === "number" && (body[key] < min || body[key] > max)) {
        setError(`${label} must be between ${min} and ${max}${unit ? ` ${unit}` : ""}.`);
        return;
      }
    }
    setLoading(true);
    try {
      const response = await fetch(`${API}/api/predict/${kind}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Unable to complete this assessment.");
      onResult(data);
    } catch (requestError) {
      setError(requestError.message || "Could not connect to the HealthAI server.");
    } finally { setLoading(false); }
  }

  return <main className="page-content assessment-page">
    <div className="page-heading">
      <div><div className="eyebrow muted">{kind === "ckd" ? "KIDNEY HEALTH" : "METABOLIC HEALTH"}</div><h2>{title} risk assessment</h2><p>Enter clinical measurements to explore a transparent model estimate.</p></div>
      <span className="secure-chip"><ShieldCheck size={15} /> Your inputs are not stored</span>
    </div>
    <div className="assessment-layout">
      <form className="panel assessment-form" onSubmit={submit}>
        {groups.map((group, groupIndex) => <section className="form-section" key={group.section}>
          <div className="form-section-title"><span className="section-number">0{groupIndex + 1}</span><div><h3>{group.section}</h3><p>Enter the patient's latest available measurements.</p></div></div>
          <div className="field-grid">{group.fields.map(([key, label, min, max, , unit]) => <label className="field" key={key}>
            <span>{label}{unit && !Array.isArray(unit) && <small>{unit}</small>}</span>
            {Array.isArray(unit) ? <span className="select-wrap"><select value={values[key]} onChange={event => setValues({ ...values, [key]: event.target.value })}>{unit.map(option => <option key={option} value={option}>{option === "notpresent" ? "Not present" : option}</option>)}</select><ChevronDown size={16} /></span> :
              <input required type="number" step="any" min={min} max={max} value={values[key]} onChange={event => setValues({ ...values, [key]: event.target.value })} />}
          </label>)}</div>
        </section>)}
        {error && <div className="form-error"><CircleAlert size={17} />{error}</div>}
        <div className="form-submit-row"><span><ShieldCheck size={15} /> Demo only · not a diagnosis</span><button className="primary-button" disabled={loading}>{loading ? <><span className="spinner" /> Analyzing…</> : <>Analyze {kind === "ckd" ? "CKD" : "Diabetes"} Risk <ArrowRight size={16} /></>}</button></div>
      </form>
      <aside className="panel form-aside"><div className="aside-icon"><Info size={20} /></div><h3>About this assessment</h3><p>Three machine-learning models contribute to a weighted ensemble estimate. SHAP and LIME show which entered features influenced this model output.</p><div className="aside-divider" /><div className="aside-detail"><strong>Input ranges</strong><span>Values outside the accepted input range are rejected before analysis.</span></div><div className="aside-detail"><strong>Privacy</strong><span>Clinical measurements are processed for the assessment and are not saved to history.</span></div><div className="aside-detail"><strong>Important</strong><span>Model explanations describe statistical associations, not causes or clinical conclusions.</span></div></aside>
    </div>
  </main>;
}

function Results({ result, onNew }) {
  const [showLime, setShowLime] = useState(false);
  const attributions = showLime ? result.lime_weights : result.shap_features;
  const maxContribution = useMemo(() => Math.max(.001, ...attributions.map(item => Math.abs(item.contribution))), [attributions]);
  return <main className="page-content results-page">
    <div className="results-top"><div><div className="eyebrow muted">ASSESSMENT COMPLETE</div><h2>Risk assessment results</h2><p>Explore the factors contributing to this model estimate.</p></div><button className="secondary-button" onClick={onNew}>New assessment</button></div>
    <section className={`risk-banner ${result.risk_level.toLowerCase()}`}>
      <div className="risk-banner-icon"><HeartPulse size={26} /></div><div className="risk-banner-copy"><span>MODEL-ESTIMATED {result.condition.toUpperCase()} RISK</span><h3>{result.risk_score.toFixed(1)}<small>%</small></h3><p>{result.risk_level} estimate <span className="risk-separator">·</span> not a clinical diagnosis</p></div>
      <div className="risk-meter"><div className="meter-label"><span>Relative risk score</span><strong>{result.risk_level}</strong></div><div className="meter-track"><span style={{ width: `${result.risk_score}%` }} /></div><div className="meter-scale"><span>0%</span><span>LOW</span><span>MODERATE</span><span>HIGH</span><span>100%</span></div></div>
    </section>
    <div className="result-metrics">
      {Object.entries(result.model_probabilities).map(([name, probability]) => <article className="panel metric-card" key={name}><span>{name.replace("_", " ")}</span><strong>{probability.toFixed(1)}%</strong><small>Positive-class probability</small></article>)}
      <article className="panel metric-card"><span>ENSEMBLE FUSION</span><strong>40 / 35 / 25</strong><small>XGBoost / Random Forest / LightGBM</small></article>
    </div>
    <section className="panel explanation-panel">
      <div className="section-title explanation-heading"><div><div className="eyebrow muted">MODEL TRANSPARENCY</div><h3>Why did the model make this prediction?</h3><p>Feature attributions show how inputs influenced the model estimate.</p></div>
        <div className="toggle"><button className={!showLime ? "selected" : ""} onClick={() => setShowLime(false)}>SHAP</button><button className={showLime ? "selected" : ""} onClick={() => setShowLime(true)}>LIME</button></div>
      </div>
      <div className="attribution-legend"><span>FEATURE</span><span>VALUE</span><span>CONTRIBUTION</span><span>DIRECTION</span></div>
      <div className="attribution-list">{attributions.map((item, index) => {
        const positive = item.contribution >= 0;
        const normalized = Math.min(100, Math.abs(item.contribution) / maxContribution * 100);
        return <div className="attribution-row" key={`${item.feature}-${index}`}>
          <div className="attribution-feature"><span className="feature-rank">{String(index + 1).padStart(2, "0")}</span><strong>{item.feature.replaceAll("_", " ")}</strong></div>
          <span className="attribution-value">{item.value ?? "—"}</span>
          <div className="contribution-cell"><strong className={positive ? "positive" : "negative"}>{positive ? "+" : "−"}{Math.abs(item.contribution).toFixed(4)}</strong><div className="contribution-track"><span className={positive ? "positive-bar" : "negative-bar"} style={{ width: `${normalized}%` }} /></div></div>
          <span className={`direction ${positive ? "positive" : "negative"}`}>{positive ? "↑ increases" : "↓ decreases"}</span>
        </div>;
      })}</div>
      <div className="attribution-footnote"><Info size={15} /> Signed contributions are relative to the model output; they are not clinical effects or causal evidence.</div>
    </section>
    <section className="panel natural-language"><div className="nl-icon"><Sparkles size={19} /></div><div><div className="eyebrow muted">PLAIN-LANGUAGE SUMMARY</div><h3>What contributed to this estimate?</h3><p>{result.explanation}</p></div></section>
    {result.demo_data && <div className="notice warning"><CircleAlert size={17} /><span><strong>Synthetic demo model.</strong> This estimate comes from generated demonstration data and is not clinically validated. Do not use it to make healthcare decisions.</span></div>}
  </main>;
}

function History({ entries, loading, error }) {
  return <main className="page-content">
    <div className="page-heading"><div><div className="eyebrow muted">ASSESSMENT LOG</div><h2>Prediction history</h2><p>Review risk summaries saved from completed assessments.</p></div><span className="date-chip"><Database size={15} /> {entries.length} saved {entries.length === 1 ? "record" : "records"}</span></div>
    <section className="panel history-panel">
      {error && <div className="form-error"><CircleAlert size={17} />{error}</div>}
      <div className="history-table-wrap"><table><thead><tr><th>ASSESSMENT</th><th>DATE & TIME</th><th>RISK SCORE</th><th>RISK CATEGORY</th></tr></thead>
      <tbody>{entries.map(item => <tr key={item.id}><td><span className={`history-type-icon ${item.condition}`}><HeartPulse size={17} /></span><strong>{item.condition === "ckd" ? "Chronic Kidney Disease" : "Diabetes"}</strong></td><td>{new Date(item.created_at).toLocaleString()}</td><td className="score-cell">{Number(item.risk_score).toFixed(1)}%</td><td><RiskPill level={item.risk_level} /></td></tr>)}</tbody></table>
      {loading && <div className="empty-state"><span className="spinner dark" /> Loading assessment history…</div>}
      {!loading && !error && !entries.length && <div className="empty-state"><ClipboardList size={30} /><strong>No assessments yet</strong><span>Completed risk assessments will be listed here.</span></div>}</div>
      <div className="history-footnote"><Info size={15} /> History stores only the assessment type, score, category and timestamp—not clinical input values.</div>
    </section>
    <div className="notice"><ShieldCheck size={16} /><span><strong>Private by design.</strong> Clinical measurements are not saved to the history database.</span></div>
  </main>;
}

export default function App() {
  const [page, setPage] = useState("Dashboard");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState("");
  const [health, setHealth] = useState(null);

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError("");
    try {
      const response = await fetch(`${API}/api/history`);
      if (!response.ok) throw new Error("Unable to load assessment history.");
      setHistory(await response.json());
    } catch (error) {
      setHistoryError(error.message);
    } finally { setHistoryLoading(false); }
  }, []);

  useEffect(() => {
    fetch(`${API}/api/health`).then(response => response.ok ? response.json() : null).then(setHealth).catch(() => setHealth(null));
    loadHistory();
  }, [loadHistory]);

  function showResult(data) {
    setResult(data);
    loadHistory();
    setPage("Results");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  function navigate(nextPage) {
    setPage(nextPage);
    if (nextPage !== "Results") setResult(null);
    if (nextPage === "History") loadHistory();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  return <div className="app-shell">
    <Header page={page} onNavigate={navigate} />
    {page === "Dashboard" && <Dashboard health={health} history={history} navigate={navigate} />}
    {page === "CKD" && <AssessmentForm key="ckd" kind="ckd" onResult={showResult} />}
    {page === "Diabetes" && <AssessmentForm key="diabetes" kind="diabetes" onResult={showResult} />}
    {page === "Results" && result && <Results result={result} onNew={() => navigate(result.condition === "ckd" ? "CKD" : "Diabetes")} />}
    {page === "History" && <History entries={history} loading={historyLoading} error={historyError} />}
    <footer className="site-footer"><span>HealthAI <span>·</span> Explainable risk assessment demo</span><span>Not a medical device · Not for clinical use</span></footer>
  </div>;
}
