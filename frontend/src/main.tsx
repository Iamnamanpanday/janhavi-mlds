import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Shield, AlertTriangle, CheckCircle2, Cpu, Activity, Database,
  Terminal, BarChart3, Layers, Sliders, ChevronRight, Lock,
  FileCode, Zap, Info, HelpCircle, RefreshCw, Download, Upload,
  FileSpreadsheet, Play, Check, X, Search, Filter, ShieldCheck,
  ShieldAlert, Eye, ArrowUpRight, ChevronDown, Award
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Cell, Legend, PieChart, Pie
} from 'recharts';
import './style.css';

const API = 'http://localhost:8000';

type Tab = 'testlab' | 'scanner' | 'constraints' | 'benchmarks' | 'viva';
type FilterMode = 'all' | 'correct' | 'mismatch' | 'tp' | 'tn';

interface TestDatasetMeta {
  id: string;
  filename: string;
  name: string;
  description: string;
  total_samples: number;
  malware_count: number;
  benign_count: number;
  threat_ratio: number;
  tags: string[];
}

interface TopDriver {
  feature: string;
  domain: string;
  value: number;
  shap_impact: number;
  direction: 'malware' | 'benign';
}

interface SampleRow {
  row_id: number;
  prediction: string;
  actual_label?: string;
  threat_probability: number;
  status_type: 'TP' | 'TN' | 'FP' | 'FN' | 'SCANNED';
  is_correct?: boolean;
  rf_prediction: string;
  std_prediction: string;
  models_unanimous: boolean;
  diagnostic_reason: string;
  top_driving_features: TopDriver[];
  E_file: number;
  E_text: number;
  sus_sections: number;
  packer: number;
}

interface TestEvaluationResult {
  dataset_name: string;
  total_samples: number;
  has_ground_truth: boolean;
  ground_truth_stats?: {
    total_correct: number;
    total_mismatch: number;
    true_positives: number;
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
    accuracy_pct: number;
    explanation: string;
  };
  malware_detected: number;
  benign_detected: number;
  threat_percentage: number;
  model_evaluations: Record<string, {
    name: string;
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    fpr: number;
    confusion_matrix: { tn: number; fp: number; fn: number; tp: number; };
  }>;
  global_top_features: {
    feature: string;
    domain: string;
    mean_impact: number;
    mean_value: number;
  }[];
  risk_distribution: { range: string; count: number }[];
  sample_preview: SampleRow[];
}

export function App() {
  const [activeTab, setActiveTab] = useState<Tab>('testlab');
  const [serverOnline, setServerOnline] = useState<boolean>(false);
  
  // Test Lab state
  const [testCatalog, setTestCatalog] = useState<TestDatasetMeta[]>([]);
  const [evalResult, setEvalResult] = useState<TestEvaluationResult | null>(null);
  const [evalLoading, setEvalLoading] = useState<boolean>(false);
  const [activeDatasetId, setActiveDatasetId] = useState<string>('');
  const [sampleSearch, setSampleSearch] = useState<string>('');
  const [filterMode, setFilterMode] = useState<FilterMode>('all');
  const [selectedSample, setSelectedSample] = useState<SampleRow | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Scanner state
  const [presets, setPresets] = useState<any[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string>('benign_calc');
  const [features, setFeatures] = useState<Record<string, number>>({});
  const [scanResult, setScanResult] = useState<any>(null);
  const [scanLoading, setScanLoading] = useState<boolean>(false);

  // Benchmarks state
  const [chartData, setChartData] = useState<any>(null);
  const [auditData, setAuditData] = useState<any>(null);

  // Initialize
  useEffect(() => {
    fetch(`${API}/api/health`)
      .then(r => r.json())
      .then(() => setServerOnline(true))
      .catch(() => setServerOnline(false));

    fetch(`${API}/api/test_datasets`)
      .then(r => r.json())
      .then(data => {
        setTestCatalog(data.catalog || []);
        if (data.catalog && data.catalog.length > 0) {
          runDatasetEval(data.catalog[0].filename, data.catalog[0].id);
        }
      })
      .catch(err => console.error('Test catalog error:', err));

    fetch(`${API}/api/presets`)
      .then(r => r.json())
      .then(data => {
        setPresets(data.presets || []);
        if (data.presets && data.presets.length > 0) {
          setFeatures(data.presets[0].data);
        }
      })
      .catch(err => console.error('Presets error:', err));

    fetch(`${API}/api/charts_data`)
      .then(r => r.json())
      .then(data => setChartData(data))
      .catch(err => console.error('Charts error:', err));

    fetch(`${API}/api/overview`)
      .then(r => r.json())
      .then(data => setAuditData(data.audit))
      .catch(err => console.error('Overview error:', err));
  }, []);

  // Run evaluation on a pre-generated test dataset
  const runDatasetEval = async (filename: string, id: string) => {
    setEvalLoading(true);
    setActiveDatasetId(id);
    setSelectedSample(null);
    try {
      const resp = await fetch(`${API}/api/evaluate_preset_test_set/${filename}`, { method: 'POST' });
      const data = await resp.json();
      setEvalResult(data);
      if (data.sample_preview && data.sample_preview.length > 0) {
        setSelectedSample(data.sample_preview[0]);
      }
    } catch (e) {
      console.error('Eval failed:', e);
    } finally {
      setEvalLoading(false);
    }
  };

  // Upload custom CSV
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setEvalLoading(true);
    setActiveDatasetId('custom_upload');
    setSelectedSample(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const resp = await fetch(`${API}/api/upload_test_dataset`, {
        method: 'POST',
        body: formData
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || 'Upload failed');
      setEvalResult(data);
      if (data.sample_preview && data.sample_preview.length > 0) {
        setSelectedSample(data.sample_preview[0]);
      }
    } catch (err: any) {
      alert(`Error evaluating dataset: ${err.message}`);
    } finally {
      setEvalLoading(false);
    }
  };

  // Run Single Scanner
  const triggerScan = async (override?: Record<string, number>) => {
    setScanLoading(true);
    try {
      const resp = await fetch(`${API}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: override || features })
      });
      const data = await resp.json();
      setScanResult(data);
    } catch (e) {
      console.error('Scan error:', e);
    } finally {
      setScanLoading(false);
    }
  };

  const handleSelectPreset = (pId: string) => {
    setSelectedPresetId(pId);
    const p = presets.find(x => x.id === pId);
    if (p) {
      setFeatures({ ...p.data });
      triggerScan(p.data);
    }
  };

  // Filtered samples based on search and filterMode
  const filteredSamples = evalResult?.sample_preview.filter(s => {
    if (filterMode === 'correct' && !s.is_correct) return false;
    if (filterMode === 'mismatch' && s.is_correct) return false;
    if (filterMode === 'tp' && s.status_type !== 'TP') return false;
    if (filterMode === 'tn' && s.status_type !== 'TN') return false;

    if (!sampleSearch) return true;
    const term = sampleSearch.toLowerCase();
    return s.prediction.toLowerCase().includes(term) ||
      (s.actual_label && s.actual_label.toLowerCase().includes(term)) ||
      String(s.row_id).includes(term) ||
      s.diagnostic_reason.toLowerCase().includes(term);
  }) || [];

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      
      {/* Top Console Bar */}
      <header style={{
        background: '#0d1424',
        borderBottom: '1px solid #1a2744',
        padding: '12px 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        zIndex: 100
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{
            background: '#1e293b',
            border: '1px solid #334155',
            borderRadius: '8px',
            padding: '7px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Shield size={22} color="#38bdf8" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '1.2rem', fontWeight: 800, color: '#ffffff', letterSpacing: '-0.02em' }}>
                AEGIS<span style={{ color: '#38bdf8' }}>.PE</span>
              </span>
              <span className="badge-neutral" style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)' }}>
                MODEL TRAINED: 1,800 SAMPLES
              </span>
            </div>
            <p style={{ fontSize: '0.74rem', color: '#64748b' }}>
              Low-Signature Polymorphic Malware Detection & Explainability Console
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          {[
            { id: 'testlab', label: 'Test Lab & Diagnostics', icon: FileSpreadsheet },
            { id: 'scanner', label: 'Single PE Scanner', icon: Zap },
            { id: 'constraints', label: 'Interaction Constraints', icon: Layers },
            { id: 'benchmarks', label: 'Model Benchmarks', icon: BarChart3 },
            { id: 'viva', label: 'Viva Voce Defense', icon: HelpCircle },
          ].map(t => {
            const Icon = t.icon;
            const active = activeTab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id as Tab)}
                className={`btn-secondary ${active ? 'active' : ''}`}
                style={{ fontSize: '0.82rem' }}
              >
                <Icon size={15} />
                {t.label}
              </button>
            );
          })}
        </nav>

        {/* Server Status Indicator */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.78rem', fontFamily: 'var(--font-mono)' }}>
          <span style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: serverOnline ? '#10b981' : '#ef4444',
            display: 'inline-block'
          }} />
          <span style={{ color: serverOnline ? '#10b981' : '#ef4444' }}>
            {serverOnline ? 'BACKEND 8000 ONLINE' : 'OFFLINE'}
          </span>
        </div>
      </header>

      {/* Main Container */}
      <main style={{ flex: 1, padding: '24px', maxWidth: '1440px', margin: '0 auto', width: '100%' }}>

        {/* ======================================================== */}
        {/* TAB 1: TEST LAB & DIAGNOSTICS                            */}
        {/* ======================================================== */}
        {activeTab === 'testlab' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
            
            {/* Top Action Bar: Upload or Pick Test Dataset */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '18px' }}>
              
              {/* Card 1: Custom Test Dataset Upload Zone */}
              <div className="solid-card" style={{ padding: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Upload size={18} color="#38bdf8" />
                    <h3 style={{ fontSize: '0.95rem', fontWeight: 700 }}>Upload Any Test Dataset (CSV)</h3>
                  </div>
                  <span className="badge-neutral">CUSTOM TEST FILE</span>
                </div>
                <p style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: '14px' }}>
                  Upload your own test dataset. The engine maps PE features, verifies against ground-truth (<code style={{ color: '#38bdf8' }}>class</code>), and calculates TreeSHAP feature attributions.
                </p>

                <input
                  type="file"
                  accept=".csv"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  onChange={handleFileUpload}
                />
                <div
                  className="dropzone"
                  onClick={() => fileInputRef.current?.click()}
                  style={{ padding: '22px 14px' }}
                >
                  <FileSpreadsheet size={32} color="#64748b" style={{ margin: '0 auto 8px auto' }} />
                  <div style={{ fontSize: '0.88rem', fontWeight: 600, color: '#f8fafc' }}>
                    Click or Drag & Drop Any Test CSV Here
                  </div>
                  <div style={{ fontSize: '0.74rem', color: '#64748b', marginTop: '4px' }}>
                    Runs instant evaluation, error diagnostics, and feature impact analysis
                  </div>
                </div>
              </div>

              {/* Card 2: Ground Truth Verification Explainer */}
              <div className="solid-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Award size={18} color="#10b981" />
                      <h3 style={{ fontSize: '0.95rem', fontWeight: 700 }}>How Do You Know The Tests Are Right?</h3>
                    </div>
                    <span className="badge-clean">GROUND-TRUTH AUDITED</span>
                  </div>
                  <p style={{ fontSize: '0.8rem', color: '#cbd5e1', lineHeight: '1.5', marginBottom: '12px' }}>
                    Every benchmark file contains a <strong>certified ground-truth label</strong> (0 = Benign, 1 = Malware) from verified execution signatures.
                    Predictions are audited against these labels into 4 clinical quadrants:
                  </p>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '0.74rem' }}>
                    <div style={{ background: '#0b1120', padding: '8px', borderRadius: '6px', borderLeft: '3px solid #10b981' }}>
                      <strong style={{ color: '#10b981' }}>TP (True Positive):</strong> Real malware caught.
                    </div>
                    <div style={{ background: '#0b1120', padding: '8px', borderRadius: '6px', borderLeft: '3px solid #38bdf8' }}>
                      <strong style={{ color: '#38bdf8' }}>TN (True Negative):</strong> Clean file cleared.
                    </div>
                    <div style={{ background: '#0b1120', padding: '8px', borderRadius: '6px', borderLeft: '3px solid #ef4444' }}>
                      <strong style={{ color: '#ef4444' }}>FP (False Alarm):</strong> Benign marked threat.
                    </div>
                    <div style={{ background: '#0b1120', padding: '8px', borderRadius: '6px', borderLeft: '3px solid #f59e0b' }}>
                      <strong style={{ color: '#f59e0b' }}>FN (Missed):</strong> Malware evaded model.
                    </div>
                  </div>
                </div>

                <div style={{ marginTop: '12px', paddingTop: '8px', borderTop: '1px solid #1a2744', fontSize: '0.75rem', color: '#64748b' }}>
                  TreeSHAP isolates the exact features that pushed the score towards Malware (+) or Benign (-).
                </div>
              </div>

            </div>

            {/* 10 Dedicated Test Datasets Gallery */}
            <div className="solid-card" style={{ padding: '20px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px', flexWrap: 'wrap', gap: '10px' }}>
                <div>
                  <h3 style={{ fontSize: '1.05rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Layers size={18} color="#38bdf8" />
                    10 Dedicated Test Datasets (300 Samples Each)
                  </h3>
                  <p style={{ fontSize: '0.78rem', color: '#94a3b8', marginTop: '2px' }}>
                    Click <strong>Run Test</strong> on any dataset to view ground-truth verification and affected features.
                  </p>
                </div>
                {evalLoading && (
                  <span style={{ fontSize: '0.8rem', color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>
                    Evaluating TreeSHAP contributions...
                  </span>
                )}
              </div>

              {/* Grid of 10 test sets */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '12px' }}>
                {testCatalog.map(item => {
                  const isActive = activeDatasetId === item.id;
                  return (
                    <div
                      key={item.id}
                      className={isActive ? 'solid-card-elevated' : 'solid-card'}
                      style={{
                        padding: '14px',
                        borderColor: isActive ? '#38bdf8' : 'var(--border-card)',
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'space-between'
                      }}
                    >
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '6px' }}>
                          <span style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: '#38bdf8', fontWeight: 700 }}>
                            300 SAMPLES
                          </span>
                          <span className={item.threat_ratio > 70 ? 'badge-threat' : item.threat_ratio === 0 ? 'badge-clean' : 'badge-amber'}>
                            {item.threat_ratio}% MALWARE
                          </span>
                        </div>
                        <h4 style={{ fontSize: '0.88rem', fontWeight: 700, color: '#f8fafc', marginBottom: '4px' }}>
                          {item.name}
                        </h4>
                        <p style={{ fontSize: '0.74rem', color: '#94a3b8', lineHeight: '1.4', marginBottom: '10px' }}>
                          {item.description}
                        </p>
                      </div>

                      <div style={{ display: 'flex', gap: '6px', marginTop: '8px' }}>
                        <button
                          onClick={() => runDatasetEval(item.filename, item.id)}
                          className={isActive ? 'btn-primary' : 'btn-secondary'}
                          style={{ flex: 1, fontSize: '0.75rem', padding: '6px 10px', justifyContent: 'center' }}
                        >
                          <Play size={12} />
                          {isActive ? 'Active Test' : 'Run Test'}
                        </button>
                        <a
                          href={`${API}/api/download_test_dataset/${item.filename}`}
                          download
                          className="btn-secondary"
                          style={{ padding: '6px 10px', fontSize: '0.75rem' }}
                          title="Download CSV"
                        >
                          <Download size={13} />
                        </a>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Test Evaluation Results & Ground-Truth Verification */}
            {evalResult && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
                
                {/* Result Header Banner with Ground Truth Verification Stat */}
                <div className="solid-card" style={{ padding: '20px 24px', borderLeft: '5px solid #38bdf8' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span className="badge-neutral" style={{ fontFamily: 'var(--font-mono)' }}>DATASET EVALUATION</span>
                        <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
                          File: <strong style={{ color: '#fff' }}>{evalResult.dataset_name}</strong>
                        </span>
                      </div>
                      <h2 style={{ fontSize: '1.35rem', fontWeight: 800, marginTop: '4px' }}>
                        Verification & Affected Feature Report ({evalResult.total_samples} Total Samples)
                      </h2>
                      {evalResult.ground_truth_stats && (
                        <p style={{ fontSize: '0.84rem', color: '#38bdf8', marginTop: '4px', maxWidth: '850px' }}>
                          {evalResult.ground_truth_stats.explanation}
                        </p>
                      )}
                    </div>

                    {evalResult.ground_truth_stats && (
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '0.72rem', color: '#64748b', textTransform: 'uppercase' }}>Ground-Truth Match Rate</div>
                        <div style={{ fontSize: '2rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: '#10b981' }}>
                          {evalResult.ground_truth_stats.accuracy_pct}%
                        </div>
                        <div style={{ fontSize: '0.72rem', color: '#94a3b8' }}>
                          {evalResult.ground_truth_stats.total_correct} / {evalResult.total_samples} Verified Correct
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Dataset evaluation charts */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px' }}>
                  <div className="solid-card" style={{ padding: '16px' }}>
                    <h3 style={{ fontSize: '0.95rem', fontWeight: 700 }}>Predicted Class Distribution</h3>
                    <p style={{ fontSize: '0.72rem', color: '#64748b', marginBottom: '4px' }}>Classified samples in the selected test dataset</p>
                    <div style={{ height: 230 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={[
                              { name: 'Malware', value: evalResult.malware_detected, color: '#ef4444' },
                              { name: 'Benign', value: evalResult.benign_detected, color: '#10b981' }
                            ]}
                            dataKey="value"
                            nameKey="name"
                            cx="50%"
                            cy="50%"
                            innerRadius={55}
                            outerRadius={82}
                            paddingAngle={3}
                          >
                            {[evalResult.malware_detected, evalResult.benign_detected].map((_, index) => (
                              <Cell key={index} fill={index === 0 ? '#ef4444' : '#10b981'} />
                            ))}
                          </Pie>
                          <Tooltip contentStyle={{ background: '#101827', border: '1px solid #1f2c47', borderRadius: 6 }} />
                          <Legend />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="solid-card" style={{ padding: '16px' }}>
                    <h3 style={{ fontSize: '0.95rem', fontWeight: 700 }}>Threat Probability Distribution</h3>
                    <p style={{ fontSize: '0.72rem', color: '#64748b', marginBottom: '8px' }}>Number of samples in each risk band</p>
                    <div style={{ height: 230 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={evalResult.risk_distribution} margin={{ top: 8, right: 5, left: -20, bottom: 42 }}>
                          <XAxis dataKey="range" stroke="#64748b" fontSize={9} interval={0} angle={-18} textAnchor="end" />
                          <YAxis stroke="#64748b" fontSize={10} allowDecimals={false} />
                          <Tooltip contentStyle={{ background: '#101827', border: '1px solid #1f2c47', borderRadius: 6 }} />
                          <Bar dataKey="count" name="Samples" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {evalResult.model_evaluations?.xgboost_constrained && (
                    <div className="solid-card" style={{ padding: '16px' }}>
                      <h3 style={{ fontSize: '0.95rem', fontWeight: 700 }}>Constrained Model Confusion Matrix</h3>
                      <p style={{ fontSize: '0.72rem', color: '#64748b', marginBottom: '8px' }}>Ground-truth outcomes for this test run</p>
                      <div style={{ height: 230 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={Object.entries(evalResult.model_evaluations.xgboost_constrained.confusion_matrix).map(([key, value]) => ({
                              outcome: key.toUpperCase(), count: value
                            }))}
                            margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
                            layout="vertical"
                          >
                            <XAxis type="number" stroke="#64748b" fontSize={10} allowDecimals={false} />
                            <YAxis type="category" dataKey="outcome" stroke="#94a3b8" fontSize={11} width={45} />
                            <Tooltip contentStyle={{ background: '#101827', border: '1px solid #1f2c47', borderRadius: 6 }} />
                            <Bar dataKey="count" name="Samples" radius={[0, 4, 4, 0]}>
                              {['#10b981', '#ef4444', '#f59e0b', '#38bdf8'].map((color, index) => (
                                <Cell key={index} fill={color} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  )}

                  {evalResult.model_evaluations && Object.keys(evalResult.model_evaluations).length > 0 && (
                    <div className="solid-card" style={{ padding: '16px' }}>
                      <h3 style={{ fontSize: '0.95rem', fontWeight: 700 }}>Model Accuracy Comparison</h3>
                      <p style={{ fontSize: '0.72rem', color: '#64748b', marginBottom: '8px' }}>Accuracy achieved on this test dataset</p>
                      <div style={{ height: 230 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={Object.entries(evalResult.model_evaluations).map(([key, value]) => ({
                              model: key.replace('xgboost_', 'XGB ').replace('random_forest', 'Random Forest'),
                              accuracy: value.accuracy
                            }))}
                            margin={{ top: 10, right: 10, left: -20, bottom: 42 }}
                          >
                            <XAxis dataKey="model" stroke="#64748b" fontSize={9} interval={0} angle={-18} textAnchor="end" />
                            <YAxis domain={[0, 100]} stroke="#64748b" fontSize={10} />
                            <Tooltip contentStyle={{ background: '#101827', border: '1px solid #1f2c47', borderRadius: 6 }} />
                            <Bar dataKey="accuracy" name="Accuracy (%)" fill="#818cf8" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  )}
                </div>

                {/* Global Top Affected Features: Which Points Drive Decisions across this Dataset */}
                {evalResult.global_top_features && evalResult.global_top_features.length > 0 && (
                  <div className="solid-card" style={{ padding: '20px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                      <div>
                        <h3 style={{ fontSize: '0.98rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <BarChart3 size={16} color="#38bdf8" />
                          Which Points/Features Are Most Affected in This Dataset? (TreeSHAP Mean Impact)
                        </h3>
                        <p style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '2px' }}>
                          These specific features exerted the highest mathematical influence on the model's decisions for this test cohort.
                        </p>
                      </div>
                    </div>

                    <div style={{ height: 250, marginBottom: '14px' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          layout="vertical"
                          data={[...evalResult.global_top_features].sort((a, b) => a.mean_impact - b.mean_impact)}
                          margin={{ top: 5, right: 25, left: 95, bottom: 5 }}
                        >
                          <XAxis type="number" stroke="#64748b" fontSize={10} />
                          <YAxis type="category" dataKey="feature" stroke="#94a3b8" fontSize={10} width={95} />
                          <Tooltip contentStyle={{ background: '#101827', border: '1px solid #1f2c47', borderRadius: 6 }} />
                          <Bar dataKey="mean_impact" name="Mean |SHAP Impact|" fill="#38bdf8" radius={[0, 4, 4, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '10px' }}>
                      {evalResult.global_top_features.map((gf, idx) => (
                        <div key={idx} style={{ background: '#0b1120', padding: '10px 14px', borderRadius: '8px', border: '1px solid #1a2744' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <strong style={{ fontSize: '0.82rem', color: '#f8fafc', fontFamily: 'var(--font-mono)' }}>{gf.feature}</strong>
                            <span style={{ fontSize: '0.68rem', color: '#38bdf8', background: 'rgba(56,189,248,0.1)', padding: '2px 6px', borderRadius: '4px' }}>
                              {gf.domain}
                            </span>
                          </div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '0.75rem', color: '#94a3b8' }}>
                            <span>Mean Value: <strong style={{ color: '#fff' }}>{gf.mean_value}</strong></span>
                            <span>SHAP Impact: <strong style={{ color: '#10b981' }}>{gf.mean_impact}</strong></span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Multi-Model Scorecard */}
                {evalResult.has_ground_truth && evalResult.model_evaluations && (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px' }}>
                    {[
                      { key: 'random_forest', title: 'Random Forest Baseline', tag: 'BAGGING BASELINE', color: '#818cf8' },
                      { key: 'xgboost_standard', title: 'Standard XGBoost', tag: 'UNCONSTRAINED', color: '#2dd4bf' },
                      { key: 'xgboost_constrained', title: 'Constrained XGBoost', tag: 'PROPOSED ISOLATED', color: '#38bdf8' }
                    ].map(m => {
                      const score = evalResult.model_evaluations[m.key];
                      if (!score) return null;
                      const cm = score.confusion_matrix;
                      return (
                        <div key={m.key} className="solid-card" style={{ padding: '16px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                            <div>
                              <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: m.color, fontWeight: 700 }}>
                                {m.tag}
                              </span>
                              <h4 style={{ fontSize: '0.98rem', fontWeight: 700, margin: '2px 0 4px 0' }}>{m.title}</h4>
                            </div>
                            <span style={{ fontSize: '1.35rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: m.color }}>
                              {score.accuracy}%
                            </span>
                          </div>

                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '6px', margin: '10px 0', textAlign: 'center' }}>
                            <div style={{ background: '#0b1120', padding: '6px', borderRadius: '6px' }}>
                              <div style={{ fontSize: '0.62rem', color: '#64748b' }}>PRECISION</div>
                              <div style={{ fontSize: '0.82rem', fontWeight: 700 }}>{score.precision}%</div>
                            </div>
                            <div style={{ background: '#0b1120', padding: '6px', borderRadius: '6px' }}>
                              <div style={{ fontSize: '0.62rem', color: '#64748b' }}>RECALL</div>
                              <div style={{ fontSize: '0.82rem', fontWeight: 700 }}>{score.recall}%</div>
                            </div>
                            <div style={{ background: '#0b1120', padding: '6px', borderRadius: '6px' }}>
                              <div style={{ fontSize: '0.62rem', color: '#64748b' }}>F1-SCORE</div>
                              <div style={{ fontSize: '0.82rem', fontWeight: 700 }}>{score.f1_score}</div>
                            </div>
                            <div style={{ background: '#0b1120', padding: '6px', borderRadius: '6px' }}>
                              <div style={{ fontSize: '0.62rem', color: '#64748b' }}>FPR</div>
                              <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#ef4444' }}>{score.fpr}%</div>
                            </div>
                          </div>

                          <div style={{ background: '#090e1a', padding: '6px 10px', borderRadius: '6px', fontSize: '0.72rem', fontFamily: 'var(--font-mono)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8' }}>
                              <span>TN: <strong style={{ color: '#10b981' }}>{cm.tn}</strong></span>
                              <span>FP: <strong style={{ color: '#ef4444' }}>{cm.fp}</strong></span>
                              <span>FN: <strong style={{ color: '#f59e0b' }}>{cm.fn}</strong></span>
                              <span>TP: <strong style={{ color: '#38bdf8' }}>{cm.tp}</strong></span>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* Sample Browser & Live Detail Inspector */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: '18px' }}>
                  
                  {/* Left Column: Interactive Sample Table with Filter Pills */}
                  <div className="solid-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
                      <div>
                        <h3 style={{ fontSize: '0.95rem', fontWeight: 700 }}>Sample Telemetry Audit Table</h3>
                        <p style={{ fontSize: '0.72rem', color: '#64748b' }}>Click any row to inspect feature attribution and root cause.</p>
                      </div>

                      {/* Filter Pills */}
                      <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                        {[
                          { id: 'all', label: 'All' },
                          { id: 'correct', label: 'Verified (Match)' },
                          { id: 'mismatch', label: 'Mismatches' },
                          { id: 'tp', label: 'TP' },
                          { id: 'tn', label: 'TN' },
                        ].map(f => (
                          <button
                            key={f.id}
                            onClick={() => setFilterMode(f.id as FilterMode)}
                            className={`btn-secondary ${filterMode === f.id ? 'active' : ''}`}
                            style={{ fontSize: '0.7rem', padding: '4px 8px' }}
                          >
                            {f.label}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Table */}
                    <div style={{ flex: 1, overflowY: 'auto', maxHeight: '380px' }}>
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>#</th>
                            <th>Prediction</th>
                            {evalResult.has_ground_truth && <th>Ground Truth</th>}
                            <th>Confidence</th>
                            <th>Status</th>
                            <th>Action</th>
                          </tr>
                        </thead>
                        <tbody>
                          {filteredSamples.map(s => {
                            const isSelected = selectedSample?.row_id === s.row_id;
                            return (
                              <tr
                                key={s.row_id}
                                onClick={() => setSelectedSample(s)}
                                style={{
                                  cursor: 'pointer',
                                  background: isSelected ? '#1e293b' : undefined,
                                  borderLeft: isSelected ? '3px solid #38bdf8' : undefined
                                }}
                              >
                                <td style={{ fontFamily: 'var(--font-mono)' }}>{s.row_id}</td>
                                <td>
                                  <span className={s.prediction === 'Malware' ? 'badge-threat' : 'badge-clean'} style={{ fontSize: '0.68rem', padding: '2px 6px' }}>
                                    {s.prediction}
                                  </span>
                                </td>
                                {evalResult.has_ground_truth && (
                                  <td>
                                    <span style={{ fontSize: '0.74rem', color: s.actual_label === 'Malware' ? '#fca5a5' : '#86efac' }}>
                                      {s.actual_label}
                                    </span>
                                  </td>
                                )}
                                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
                                  {s.threat_probability}%
                                </td>
                                <td>
                                  {s.status_type === 'TP' && <span className="badge-clean" style={{ fontSize: '0.65rem' }}>TP (Caught)</span>}
                                  {s.status_type === 'TN' && <span className="badge-clean" style={{ fontSize: '0.65rem', background: '#075985', borderColor: '#0284c7', color: '#bae6fd' }}>TN (Clean)</span>}
                                  {s.status_type === 'FP' && <span className="badge-threat" style={{ fontSize: '0.65rem' }}>FP (False Alarm)</span>}
                                  {s.status_type === 'FN' && <span className="badge-amber" style={{ fontSize: '0.65rem' }}>FN (Missed)</span>}
                                </td>
                                <td>
                                  <Eye size={14} color="#38bdf8" />
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Right Column: Deep-Dive Feature Attribution & Reason Inspector */}
                  {selectedSample && (
                    <div className="solid-card" style={{ padding: '20px', borderLeft: '4px solid #38bdf8' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '10px' }}>
                        <div>
                          <span style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>
                            SAMPLE #{selectedSample.row_id} TELEMETRY DEEP-DIVE
                          </span>
                          <h3 style={{ fontSize: '1.1rem', fontWeight: 800, margin: '2px 0' }}>
                            Why Did The Model Classify This File?
                          </h3>
                        </div>
                        <div style={{ textAlign: 'right' }}>
                          <span className={selectedSample.prediction === 'Malware' ? 'badge-threat' : 'badge-clean'}>
                            {selectedSample.prediction} ({selectedSample.threat_probability}%)
                          </span>
                        </div>
                      </div>

                      {/* Plain English Root Cause Reason */}
                      <div style={{ background: '#090e1a', padding: '12px', borderRadius: '8px', border: '1px solid #1a2744', marginBottom: '14px' }}>
                        <div style={{ fontSize: '0.72rem', color: '#64748b', textTransform: 'uppercase', marginBottom: '4px' }}>
                          Diagnostic Decision Reasoning:
                        </div>
                        <p style={{ fontSize: '0.82rem', color: '#e2e8f0', lineHeight: '1.45' }}>
                          {selectedSample.diagnostic_reason}
                        </p>
                      </div>

                      {/* Top Influencing Features for This File */}
                      <h4 style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: '8px', color: '#94a3b8' }}>
                        Which Points/Features Affected This Decision? (TreeSHAP Values)
                      </h4>

                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '14px' }}>
                        {selectedSample.top_driving_features.map((df, idx) => (
                          <div key={idx} style={{ background: '#0b1120', padding: '8px 12px', borderRadius: '6px', border: '1px solid #1a2744' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <div>
                                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '0.8rem', color: '#f8fafc' }}>
                                  {df.feature} = {df.value}
                                </span>
                                <span style={{ fontSize: '0.68rem', color: '#64748b', marginLeft: '8px' }}>({df.domain})</span>
                              </div>
                              <span style={{
                                fontFamily: 'var(--font-mono)',
                                fontWeight: 700,
                                fontSize: '0.8rem',
                                color: df.direction === 'malware' ? '#f43f5e' : '#10b981'
                              }}>
                                {df.shap_impact > 0 ? `+${df.shap_impact}` : df.shap_impact} {df.direction === 'malware' ? '→ MALWARE' : '→ BENIGN'}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Tri-Model Agreement Note */}
                      <div style={{ fontSize: '0.76rem', color: '#94a3b8', borderTop: '1px solid #1a2744', paddingTop: '10px' }}>
                        <strong>Model Consensus: </strong>
                        RF={selectedSample.rf_prediction} | Std-XGB={selectedSample.std_prediction} | Constrained-XGB={selectedSample.prediction}.
                        {selectedSample.models_unanimous
                          ? ' (All 3 models in 100% agreement).'
                          : ' ⚠️ Models diverge! Notice that Constrained XGBoost prevents non-functional stomped headers from altering the score.'}
                      </div>
                    </div>
                  )}

                </div>

              </div>
            )}

          </div>
        )}

        {/* ======================================================== */}
        {/* TAB 2: SINGLE PE LIVE SCANNER                            */}
        {/* ======================================================== */}
        {activeTab === 'scanner' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            
            {/* Presets Bar */}
            <div className="solid-card" style={{ padding: '16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Cpu size={18} color="#38bdf8" />
                <span style={{ fontSize: '0.88rem', fontWeight: 600 }}>Quick Preset Target:</span>
              </div>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {presets.map(p => (
                  <button
                    key={p.id}
                    onClick={() => handleSelectPreset(p.id)}
                    className={`btn-secondary ${selectedPresetId === p.id ? 'active' : ''}`}
                    style={{ fontSize: '0.8rem' }}
                  >
                    {p.name.split(':')[0]}
                  </button>
                ))}
              </div>
              <button
                onClick={() => triggerScan()}
                className="btn-primary"
                disabled={scanLoading}
                style={{ fontSize: '0.82rem' }}
              >
                <RefreshCw size={14} />
                {scanLoading ? 'Analyzing...' : 'Re-Analyze'}
              </button>
            </div>

            {/* Verdict Hero Banner */}
            {scanResult && (
              <div
                className="solid-card"
                style={{
                  padding: '20px 24px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  borderLeft: scanResult.status === 'clean' ? '5px solid #10b981' : '5px solid #ef4444',
                  flexWrap: 'wrap',
                  gap: '16px'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                  <div style={{
                    width: '56px',
                    height: '56px',
                    borderRadius: '10px',
                    background: scanResult.status === 'clean' ? '#064e3b' : '#7f1d1d',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    {scanResult.status === 'clean' ? (
                      <CheckCircle2 size={32} color="#34d399" />
                    ) : (
                      <AlertTriangle size={32} color="#fca5a5" />
                    )}
                  </div>
                  <div>
                    <span className={scanResult.status === 'clean' ? 'badge-clean' : 'badge-threat'}>
                      {scanResult.status === 'clean' ? 'CLEAN BENIGN' : 'MALWARE DETECTED'}
                    </span>
                    <h2 style={{ fontSize: '1.45rem', fontWeight: 800, marginTop: '4px' }}>
                      {scanResult.verdict}
                    </h2>
                    <p style={{ fontSize: '0.84rem', color: '#94a3b8' }}>
                      {scanResult.status === 'clean'
                        ? 'All models agree: Safe Windows binary headers.'
                        : 'High polymorphic risk: Elevated entropy and suspicious section characteristics.'}
                    </p>
                  </div>
                </div>

                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '0.72rem', color: '#64748b', textTransform: 'uppercase' }}>Threat Probability</div>
                  <div style={{
                    fontSize: '2.2rem',
                    fontWeight: 800,
                    fontFamily: 'var(--font-mono)',
                    color: scanResult.average_risk > 0.5 ? '#ef4444' : '#10b981'
                  }}>
                    {(scanResult.average_risk * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            )}

            {/* Tri-Model Comparison Cards */}
            {scanResult && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px' }}>
                {[
                  { key: 'random_forest', title: 'Random Forest Baseline', tag: 'BAGGING', color: '#818cf8' },
                  { key: 'xgboost_standard', title: 'Standard XGBoost', tag: 'UNCONSTRAINED', color: '#2dd4bf' },
                  { key: 'xgboost_constrained', title: 'Constrained XGBoost', tag: 'PROPOSED ISOLATED', color: '#38bdf8' }
                ].map(m => {
                  const res = scanResult.models[m.key];
                  if (!res) return null;
                  return (
                    <div key={m.key} className="solid-card" style={{ padding: '18px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <div>
                          <span style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: m.color, fontWeight: 700 }}>
                            {m.tag}
                          </span>
                          <h4 style={{ fontSize: '1rem', fontWeight: 700, margin: '2px 0 6px 0' }}>{m.title}</h4>
                        </div>
                        <span className={res.prediction === 'Malware' ? 'badge-threat' : 'badge-clean'}>
                          {res.prediction}
                        </span>
                      </div>

                      <div style={{ margin: '12px 0' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '4px' }}>
                          <span style={{ color: '#94a3b8' }}>Confidence</span>
                          <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{res.risk_percent}%</span>
                        </div>
                        <div style={{ width: '100%', height: '7px', background: '#0b1120', borderRadius: '4px', overflow: 'hidden' }}>
                          <div style={{
                            width: `${res.risk_percent}%`,
                            height: '100%',
                            background: res.risk_percent > 50 ? '#ef4444' : '#10b981',
                            borderRadius: '4px'
                          }} />
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Live Sliders */}
            <div className="solid-card" style={{ padding: '20px' }}>
              <h3 style={{ fontSize: '0.98rem', fontWeight: 700, marginBottom: '16px' }}>
                Fine-Tune Key Polymorphism Indicators:
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '18px' }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '4px' }}>
                    <span>File Entropy (E_file)</span>
                    <strong style={{ color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>{Number(features['E_file'] || 5.0).toFixed(2)}</strong>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="8"
                    step="0.05"
                    value={features['E_file'] || 5.0}
                    onChange={e => {
                      const val = parseFloat(e.target.value);
                      const upd = { ...features, E_file: val };
                      setFeatures(upd);
                      triggerScan(upd);
                    }}
                  />
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '4px' }}>
                    <span>Section Entropy (E_text)</span>
                    <strong style={{ color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>{Number(features['E_text'] || 5.0).toFixed(2)}</strong>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="8"
                    step="0.05"
                    value={features['E_text'] || 5.0}
                    onChange={e => {
                      const val = parseFloat(e.target.value);
                      const upd = { ...features, E_text: val };
                      setFeatures(upd);
                      triggerScan(upd);
                    }}
                  />
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '4px' }}>
                    <span>Suspicious Sections (RWX)</span>
                    <strong style={{ color: '#f59e0b', fontFamily: 'var(--font-mono)' }}>{features['sus_sections'] || 0}</strong>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="8"
                    step="1"
                    value={features['sus_sections'] || 0}
                    onChange={e => {
                      const val = parseInt(e.target.value);
                      const upd = { ...features, sus_sections: val };
                      setFeatures(upd);
                      triggerScan(upd);
                    }}
                  />
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '4px' }}>
                    <span>Runtime Packer Flag</span>
                    <strong style={{ color: features['packer'] ? '#ef4444' : '#10b981' }}>
                      {features['packer'] ? 'PACKED' : 'CLEAN'}
                    </strong>
                  </div>
                  <button
                    onClick={() => {
                      const val = features['packer'] ? 0 : 1;
                      const upd = { ...features, packer: val };
                      setFeatures(upd);
                      triggerScan(upd);
                    }}
                    className="btn-secondary"
                    style={{ width: '100%', justifyContent: 'center' }}
                  >
                    Toggle Packer Signature
                  </button>
                </div>
              </div>
            </div>

          </div>
        )}

        {/* ======================================================== */}
        {/* TAB 3: INTERACTION CONSTRAINTS                           */}
        {/* ======================================================== */}
        {activeTab === 'constraints' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            
            <div className="solid-card" style={{ padding: '24px' }}>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 800, marginBottom: '8px' }}>
                Feature Interaction Constraints & Domain Isolation Proof
              </h2>
              <p style={{ color: '#94a3b8', fontSize: '0.88rem', lineHeight: '1.5' }}>
                Standard gradient boosted trees construct decision paths greedily across the entire feature space.
                In malware detection, this causes trees to learn spurious cross-domain correlations (e.g. coupling compiler timestamps with section entropy).
                Malware authors routinely stomp timestamps or add overlay padding to disrupt these paths.
                Enforcing <strong style={{ color: '#fff' }}>Feature Interaction Constraints</strong> mathematically confines all tree splits to domain-approved boundaries.
              </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '18px' }}>
              
              <div className="solid-card" style={{ padding: '20px', borderLeft: '4px solid #ef4444' }}>
                <span className="badge-threat" style={{ marginBottom: '8px' }}>UNCONSTRAINED SPURIOUS SPLITS</span>
                <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: '6px 0 4px 0' }}>Standard XGBoost</h3>
                <div style={{ fontSize: '2.4rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: '#ef4444' }}>
                  64.8%
                </div>
                <p style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: '4px' }}>
                  745 out of 1,149 tree split transitions cross disparate functional domains.
                </p>
              </div>

              <div className="solid-card" style={{ padding: '20px', borderLeft: '4px solid #10b981' }}>
                <span className="badge-clean" style={{ marginBottom: '8px' }}>MATHEMATICALLY ISOLATED</span>
                <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: '6px 0 4px 0' }}>Constrained XGBoost</h3>
                <div style={{ fontSize: '2.4rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: '#10b981' }}>
                  0.0%
                </div>
                <p style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: '4px' }}>
                  0 out of 1,068 tree split transitions cross domains. 100% domain-bounded trees.
                </p>
              </div>

            </div>

            {/* 4 Groups list */}
            <div className="solid-card" style={{ padding: '20px' }}>
              <h3 style={{ fontSize: '0.98rem', fontWeight: 700, marginBottom: '14px' }}>The 4 Isolated Functional Domains:</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '14px' }}>
                {[
                  { name: 'Group 1: DOS & COFF Header', count: '23 features', list: 'e_cblp, e_cp, NumberOfSections, CreationYear, FH_char0-14' },
                  { name: 'Group 2: Optional Header & Memory', count: '37 features', list: 'AddressOfEntryPoint, ImageBase, SectionAlignment, SizeOfImage, OH_DLLchar0-10' },
                  { name: 'Group 3: Entropy & Polymorphic Profile', count: '5 features', list: 'E_text, E_data, filesize, E_file, fileinfo' },
                  { name: 'Group 4: Packer & Suspicious Sections', count: '4 features', list: 'sus_sections, non_sus_sections, packer, packer_type' },
                ].map((g, i) => (
                  <div key={i} style={{ background: '#0b1120', padding: '14px', borderRadius: '8px', border: '1px solid #1a2744' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                      <strong style={{ fontSize: '0.85rem', color: '#f8fafc' }}>{g.name}</strong>
                      <span style={{ fontSize: '0.72rem', fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>{g.count}</span>
                    </div>
                    <p style={{ fontSize: '0.74rem', color: '#64748b', fontFamily: 'var(--font-mono)' }}>{g.list}</p>
                  </div>
                ))}
              </div>
            </div>

          </div>
        )}

        {/* ======================================================== */}
        {/* TAB 4: MODEL BENCHMARKS                                  */}
        {/* ======================================================== */}
        {activeTab === 'benchmarks' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '14px' }}>
              {[
                { label: 'TRAINING SAMPLES', val: '1,800 PE Files', color: '#38bdf8' },
                { label: 'TEST SAMPLES (BENCHMARK)', val: '300 PE Files', color: '#10b981' },
                { label: 'CONSTRAINED ACCURACY', val: '98.67%', color: '#38bdf8' },
                { label: 'CONSTRAINED ROC-AUC', val: '0.9994', color: '#f59e0b' },
              ].map((m, i) => (
                <div key={i} className="solid-card" style={{ padding: '16px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.68rem', color: '#64748b', fontFamily: 'var(--font-mono)' }}>{m.label}</div>
                  <div style={{ fontSize: '1.45rem', fontWeight: 800, color: m.color, marginTop: '4px', fontFamily: 'var(--font-mono)' }}>
                    {m.val}
                  </div>
                </div>
              ))}
            </div>

            {chartData && (
              <div className="solid-card" style={{ padding: '20px' }}>
                <h3 style={{ fontSize: '0.98rem', fontWeight: 700, marginBottom: '14px' }}>
                  Model Performance Comparison on 300-Sample Benchmark Cohort
                </h3>
                <div style={{ height: '280px', width: '100%' }}>
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={chartData.comparison_bars} margin={{ top: 15, right: 20, left: 0, bottom: 5 }}>
                      <XAxis dataKey="metric" stroke="#64748b" fontSize={11} />
                      <YAxis domain={[96, 100]} unit="%" stroke="#64748b" fontSize={11} />
                      <Tooltip contentStyle={{ background: '#101827', border: '1px solid #1f2c47', borderRadius: '6px' }} />
                      <Legend />
                      <Bar dataKey="Random_Forest" fill="#818cf8" name="Random Forest Baseline" radius={[3, 3, 0, 0]} />
                      <Bar dataKey="Standard_XGBoost" fill="#2dd4bf" name="Standard XGBoost" radius={[3, 3, 0, 0]} />
                      <Bar dataKey="Constrained_XGBoost" fill="#38bdf8" name="Constrained XGBoost" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

          </div>
        )}

        {/* ======================================================== */}
        {/* TAB 5: VIVA VOCE DEFENSE                                 */}
        {/* ======================================================== */}
        {activeTab === 'viva' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            
            <div className="solid-card" style={{ padding: '20px' }}>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 800, marginBottom: '4px' }}>
                🎓 College Project Viva Voce Defense Guide
              </h2>
              <p style={{ color: '#94a3b8', fontSize: '0.84rem' }}>
                Essential technical questions asked by examiners during machine learning and cybersecurity project evaluations.
              </p>
            </div>

            {[
              {
                q: "1. How do you know that the predictions on the test files are right?",
                a: "Every sample in the test files is evaluated against certified ground truth (class: 0 for Benign, 1 for Malware). The platform calculates exact True Positives (real malware detected), True Negatives (clean files verified), False Positives (false alarms), and False Negatives (missed malware) using standard confusion matrices."
              },
              {
                q: "2. How do you explain 'which points/features are affected' in a prediction?",
                a: "We compute TreeSHAP values (pred_contribs in XGBoost). For each file, the engine quantifies the exact positive or negative impact of all 69 features. A positive SHAP value (e.g. E_file = 7.72, impact +0.995) pushed the score towards Malware, while a negative value pulled towards Benign."
              },
              {
                q: "3. How are the models trained in this project?",
                a: "The models are trained on exactly 1,800 PE samples (940 malware, 860 benign) sampled from the peer-reviewed ClaMP benchmark dataset. Evaluation is performed across 10 diverse 300-sample test sets and custom user-uploaded CSV datasets."
              },
              {
                q: "4. What are Feature Interaction Constraints in XGBoost?",
                a: "Standard decision trees split greedily across all features. Feature interaction constraints restrict splits to pre-approved subsets. If a branch starts with an Entropy feature, it can only split on other Entropy features, preventing spurious cross-domain correlations."
              },
              {
                q: "5. What empirical proof do you have that the constraints worked?",
                a: "We audited all trees using model.get_booster().trees_to_dataframe(). In Standard XGBoost, 64.8% of split transitions crossed functional domains. In Constrained XGBoost, exactly 0.0% crossed domains, proving mathematical isolation."
              }
            ].map((qa, i) => (
              <div key={i} className="solid-card" style={{ padding: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                  <HelpCircle size={16} color="#38bdf8" />
                  <strong style={{ fontSize: '0.92rem', color: '#f8fafc' }}>{qa.q}</strong>
                </div>
                <p style={{ fontSize: '0.82rem', color: '#94a3b8', lineHeight: '1.4', paddingLeft: '24px' }}>
                  {qa.a}
                </p>
              </div>
            ))}

          </div>
        )}

      </main>

      {/* Footer */}
      <footer style={{
        background: '#0d1424',
        borderTop: '1px solid #1a2744',
        padding: '12px 24px',
        textAlign: 'center',
        fontSize: '0.75rem',
        color: '#64748b'
      }}>
        AEGIS.PE &copy; 2026 &middot; Ground-Truth Audited &middot; TreeSHAP Feature Attribution &middot; React + Vite + FastAPI
      </footer>

    </div>
  );
}

const rootEl = document.getElementById('root');
if (rootEl) {
  createRoot(rootEl).render(<App />);
}
