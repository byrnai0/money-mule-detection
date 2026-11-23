import { useMemo, useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  Atom,
  Check,
  ChevronLeft,
  ChevronRight,
  FileText,
  GitBranch,
  Info,
  LoaderCircle,
  Radio,
  ShieldCheck,
} from "lucide-react";

/* ------------------------------------------------------------------ */
/* DATA                                                                */
/* ------------------------------------------------------------------ */

const ACCOUNT = { id: "10_800059840", score: 0.5462382869422436, band: "MEDIUM" };

const FEATURE_ATTRIBUTION = [
  { name: "Receiving currency diversity", impact: 0.4600410292297602 },
  { name: "Incoming payment format diversity", impact: -0.2594380503892897 },
  { name: "Incoming log amount mean", impact: 0.3050307331979275 },
  { name: "Unique receiver banks", impact: -0.14864204198122022 },
  { name: "Outgoing transaction count", impact: -0.13255445256829257 },
];

const ENSEMBLE = [
  { model: "GraphSAGE", weight: 0.78 },
  { model: "GATv2", weight: 0.19 },
  { model: "GIN", weight: 0.03 },
];

const ENSEMBLE_TEST_METRICS = [
  { model: "GraphSAGE", prAuc: 0.29439487418194715, rocAuc: 0.9116186694377556, precision: 0.057073600705156456, recall: 0.815748031496063, f1: 0.10668314282772114, f2: 0.22296831955922866 },
  { model: "GATv2", prAuc: 0.21482209858998677, rocAuc: 0.8822367070845711, precision: 0.05425017277125086, recall: 0.7417322834645669, f1: 0.10110550606418375, f2: 0.20985564070575655 },
  { model: "GIN", prAuc: 0.19863876092585583, rocAuc: 0.8704435599577651, precision: 0.03201538548560439, recall: 0.8913385826771654, f1: 0.06181063667139893, f2: 0.13996735743607497 },
];

const NEIGHBORS = [
  { id: "1467_800D10B40", relation: "Neighbor feature ablation", effect: -0.08987050279974929 },
];

const NEIGHBOR_EXPLANATION = {
  baselineScore: 0.5462382869422436,
  scoreAfterFeatureAblation: 0.6361087897419929,
  transactionCount: 1,
  edgeIndex: 1870666,
  amountPaid: 34.97999954223633,
  amountReceived: 34.97999954223633,
  edgeLabel: 0.0,
  timestamp: 1662164,
};

const GLOBAL_IMPORTANCE = [
  { name: "outgoing_log_amount_max", value: 0.27951188929835385 },
  { name: "Incoming payment format diversity", value: 0.17591134504065054 },
  { name: "Receiving currency diversity", value: 0.09418701390013268 },
  { name: "outgoing_log_amount_std", value: 0.06968342855771065 },
  { name: "Out transaction count", value: 0.06308904497075546 },
  { name: "Outgoing payment format diversity", value: 0.0609039884217249 },
];

const CHECKPOINTS = [
  { id: "gcn_sampled_model.pt", group: "Full-feature", inputs: 26, layer1: "GCNConv · 26 → 64", activation: "ReLU · dropout 0.30", layer2: "GCNConv · 64 → 2" },
  { id: "gatv2_sampled_model.pt", group: "Full-feature", inputs: 26, layer1: "GATv2Conv · 4 × 64 heads · 256 concatenated", activation: "ELU · dropout 0.30", layer2: "GATv2Conv · 1 head · 256 → 2" },
  { id: "sage_sampled_model.pt", group: "Full-feature", inputs: 26, layer1: "SAGEConv · 26 → 64", activation: "ReLU · dropout 0.30", layer2: "SAGEConv · 64 → 2" },
  { id: "gin_sampled_model.pt", group: "Full-feature", inputs: 26, layer1: "GINConv MLP · 26 → 64 → 64", activation: "ReLU · dropout 0.30", layer2: "GINConv MLP · 64 → 64 → 2" },
  { id: "cheb_sampled_model.pt", group: "Full-feature", inputs: 26, layer1: "ChebConv K=3 · 26 → 64", activation: "ReLU · dropout 0.30", layer2: "ChebConv K=3 · 64 → 2" },
  { id: "cp_gnn_sampled_model.pt", group: "Full-feature", inputs: 26, layer1: "Node / neighbor projections · 26 → rank 16 → 64", activation: "Element-wise interaction · ReLU · dropout 0.30", layer2: "Node / neighbor projections · 64 → rank 16 → 2" },
  { id: "cp_pooling_gnn_sampled_model.pt", group: "Full-feature", inputs: 26, layer1: "CPLayer · (26 + 1) → rank 16 → 64", activation: "Multiplicative neighbor pooling · ReLU · dropout 0.30", layer2: "CPLayer · (64 + 1) → rank 16 → 2" },
  { id: "gcn_qiea12_model.pt", group: "QIEA-selected", inputs: 12, layer1: "GCNConv · 12 → 64", activation: "ReLU · dropout 0.30", layer2: "GCNConv · 64 → 2" },
  { id: "gat_qiea12_model.pt", group: "QIEA-selected", inputs: 12, layer1: "GATConv · 4 × 64 heads · 256 concatenated", activation: "ELU · dropout 0.30", layer2: "GATConv · 1 head · 256 → 2" },
  { id: "gatv2_qiea12_model.pt", group: "QIEA-selected", inputs: 12, layer1: "GATv2Conv · 4 × 64 heads · 256 concatenated", activation: "ELU · dropout 0.30", layer2: "GATv2Conv · 1 head · 256 → 2" },
  { id: "sage_qiea12_model.pt", group: "QIEA-selected", inputs: 12, layer1: "SAGEConv · 12 → 64", activation: "ReLU · dropout 0.30", layer2: "SAGEConv · 64 → 2" },
  { id: "gin_qiea12_model.pt", group: "QIEA-selected", inputs: 12, layer1: "GINConv MLP · 12 → 64 → 64", activation: "ReLU · dropout 0.30", layer2: "GINConv MLP · 64 → 64 → 2" },
  { id: "cheb_qiea12_model.pt", group: "QIEA-selected", inputs: 12, layer1: "ChebConv K=3 · 12 → 64", activation: "ReLU · dropout 0.30", layer2: "ChebConv K=3 · 64 → 2" },
];

const EXPERIMENTS = [
  { name: "Saved GNN reference", threshold: 0.5, prAuc: 0.29905876081714466, rocAuc: 0.9134609235291513, brier: 0.10950984529833736, precision: 0.05939574273289082, recall: 0.8173228346456692, f1: 0.1107436253067321, f2: 0.23009398829579714, flagged: 8738 },
  { name: "QUBO ensemble", threshold: 0.5, prAuc: 0.29540096989712955, rocAuc: 0.9126703026374563, brier: 0.10753160343044073, precision: 0.061542174490165316, recall: 0.8031496062992126, f1: 0.11432414256893074, f2: 0.23552230534774177, flagged: 8287 },
  { name: "Classical Brier-grid reference", threshold: 0.5, prAuc: 0.29496130365329426, rocAuc: 0.9124318874576341, brier: 0.10752008515947117, precision: 0.06153100775193798, recall: 0.8, f1: 0.11427286019570353, f2: 0.2352723230826232, flagged: null },
  { name: "QIEA + GNN hybrid", threshold: 0.77, prAuc: 0.3017551232676519, rocAuc: 0.9131643749154543, brier: 0.10954079491634307, precision: 0.16766467065868262, recall: 0.48503937007874015, f1: 0.24919093851132687, f2: 0.3518391592414896, flagged: 1837, hybrid: true },
];

const QIEA_PILOT = {
  allFeaturePrAuc: 0.07399143638252076,
  selectedPrAuc: 0.08091269485426893,
  selectedFeatures: 12,
  totalFeatures: 26,
};

const SECTIONS = [
  { key: "overview", label: "Overview" },
  { key: "attribution", label: "Features" },
  { key: "ensemble", label: "Ensemble" },
  { key: "graph", label: "Graph Evidence" },
  { key: "global", label: "Global" },
];

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000/v1").replace(/\/$/, "");

/* ------------------------------------------------------------------ */
/* Shared UI                                                           */
/* ------------------------------------------------------------------ */

function Group({ title, children, className = "", bodyClass = "p-5" }) {
  return (
    <section className={`animate-rise ${className}`}>
      {title && (
        <div className="mb-2 flex items-baseline justify-between px-4">
          <h3 className="text-[13px] font-medium uppercase tracking-wide text-muted-foreground">{title}</h3>
        </div>
      )}
      <div className={`card-ios ${bodyClass}`}>{children}</div>
    </section>
  );
}

function Note({ children }) {
  return (
    <div className="flex gap-3 px-4 text-[13px] leading-relaxed text-muted-foreground">
      <Info className="mt-0.5 h-4 w-4 shrink-0 text-subtle" />
      <p>{children}</p>
    </div>
  );
}

function PillButton({ icon: Icon, children, onClick, primary, active }) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex h-9 items-center gap-1.5 rounded-full px-4 text-[14px] font-medium transition-all active:scale-[0.97] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
        primary
          ? "bg-primary text-primary-foreground hover:brightness-110"
          : active
            ? "bg-accent text-accent-foreground"
            : "bg-surface text-foreground hover:bg-surface-raised"
      }`}
    >
      {Icon && <Icon className="h-4 w-4" />}
      <span>{children}</span>
    </button>
  );
}

function BackLink({ onClick }) {
  return (
    <button
      onClick={onClick}
      className="-ml-1 inline-flex items-center text-[16px] text-primary transition-opacity hover:opacity-80"
    >
      <ChevronLeft className="h-5 w-5" />
      Back to dashboard
    </button>
  );
}

function LargeTitle({ caption, title, right }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        {caption && <p className="text-[13px] font-medium text-muted-foreground">{caption}</p>}
        <h1 className="font-display text-[34px] font-bold leading-tight tracking-tight text-foreground">{title}</h1>
      </div>
      {right}
    </div>
  );
}

function fmtSigned(v) {
  return `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(3)}`;
}

function useSignalCounts() {
  return useMemo(() => {
    const up = FEATURE_ATTRIBUTION.filter((f) => f.impact > 0).length;
    return { up, down: FEATURE_ATTRIBUTION.length - up };
  }, []);
}

/* ------------------------------------------------------------------ */
/* Widgets                                                             */
/* ------------------------------------------------------------------ */

function RiskRing({ score }) {
  const r = 76;
  const c = 2 * Math.PI * r;
  return (
    <div className="relative mx-auto h-52 w-52">
      <svg viewBox="0 0 180 180" className="h-full w-full -rotate-90">
        <circle cx="90" cy="90" r={r} fill="none" strokeWidth="9" className="stroke-surface-raised" />
        <circle
          cx="90"
          cy="90"
          r={r}
          fill="none"
          strokeWidth="9"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - score)}
          className="stroke-primary transition-all duration-700"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-display text-[44px] font-semibold tracking-tight text-foreground">
          {(score * 100).toFixed(1)}
          <span className="text-2xl text-muted-foreground">%</span>
        </span>
        <span className="text-[13px] text-muted-foreground">Score {score.toFixed(4)}</span>
      </div>
    </div>
  );
}

function SignedRows({ items, valueKey = "impact", labelKey = "name", subKey }) {
  const max = Math.max(...items.map((d) => Math.abs(d[valueKey])));
  return (
    <ul className="divide-y divide-border">
      {items.map((d) => {
        const v = d[valueKey];
        const up = v >= 0;
        return (
          <li key={d[labelKey]} className="flex items-center gap-4 px-5 py-3.5">
            <span
              className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${
                up ? "bg-accent text-accent-foreground" : "bg-surface-raised text-muted-foreground"
              }`}
            >
              {up ? <ArrowUp className="h-3.5 w-3.5" /> : <ArrowDown className="h-3.5 w-3.5" />}
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline justify-between gap-3">
                <span className="truncate text-[15px] text-foreground">{d[labelKey]}</span>
                <span className={`text-[15px] tabular-nums ${up ? "text-foreground" : "text-muted-foreground"}`}>
                  {fmtSigned(v)}
                </span>
              </div>
              {subKey && <p className="text-[12px] text-muted-foreground">{d[subKey]}</p>}
              <div className="mt-2 h-1 overflow-hidden rounded-full bg-surface-raised">
                <div
                  className={`h-full rounded-full ${up ? "bg-primary" : "bg-subtle"}`}
                  style={{ width: `${(Math.abs(v) / max) * 100}%` }}
                />
              </div>
            </div>
          </li>
        );
      })}
    </ul>
  );
}

const ENSEMBLE_SHADES = ["bg-chart-1", "bg-chart-2", "bg-chart-3"];

function EnsembleList() {
  return (
    <>
      <div className="mx-5 mt-5 flex h-2 gap-1">
        {ENSEMBLE.map((m, i) => (
          <div key={m.model} className={`rounded-full ${ENSEMBLE_SHADES[i]}`} style={{ width: `${m.weight * 100}%` }} />
        ))}
      </div>
      <ul className="mt-3 divide-y divide-border">
        {ENSEMBLE.map((m, i) => (
          <li key={m.model} className="flex items-center justify-between px-5 py-3">
            <span className="flex items-center gap-3 text-[15px] text-foreground">
              <span className={`h-2.5 w-2.5 rounded-full ${ENSEMBLE_SHADES[i]}`} />
              {m.model}
            </span>
            <span className="text-[15px] tabular-nums text-muted-foreground">{(m.weight * 100).toFixed(0)}%</span>
          </li>
        ))}
      </ul>
    </>
  );
}

/* ------------------------------------------------------------------ */
/* Dashboard sections                                                  */
/* ------------------------------------------------------------------ */

function OverviewSection() {
  const { up, down } = useSignalCounts();
  const [liveResult, setLiveResult] = useState(null);
  const [liveError, setLiveError] = useState("");
  const [liveLoading, setLiveLoading] = useState(false);
  const top = [...FEATURE_ATTRIBUTION].sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact)).slice(0, 5);

  async function runLiveSample() {
    setLiveLoading(true);
    setLiveError("");

    try {
      const response = await fetch(`${API_BASE_URL}/analysis/live-random-account`);
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail ?? "The live sample request failed.");
      }
      setLiveResult(payload);
    } catch (error) {
      setLiveError(error instanceof Error && error.message === "Failed to fetch"
        ? "Cannot reach the detection API. Start the backend and try again."
        : error instanceof Error ? error.message : "The live sample request failed.");
    } finally {
      setLiveLoading(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-5">
      <div className="space-y-6 lg:col-span-2">
        <Group title="Risk score">
          <RiskRing score={ACCOUNT.score} />
          <div className="mt-3 flex justify-center">
            <span className="rounded-full bg-surface-raised px-3 py-1 text-[13px] font-medium text-foreground">
              Medium risk
            </span>
          </div>
        </Group>
        <Group title="Signals" bodyClass="grid grid-cols-2 divide-x divide-border">
          <div className="p-5">
            <p className="text-[13px] text-muted-foreground">Raising score</p>
            <p className="mt-1 font-display text-3xl font-semibold text-foreground">{up}</p>
          </div>
          <div className="p-5">
            <p className="text-[13px] text-muted-foreground">Lowering score</p>
            <p className="mt-1 font-display text-3xl font-semibold text-foreground">{down}</p>
          </div>
        </Group>
        <div className="space-y-3">
          <button
            type="button"
            onClick={runLiveSample}
            disabled={liveLoading}
            className="inline-flex h-10 w-full items-center justify-center gap-2 rounded-full bg-primary px-4 text-sm font-medium text-primary-foreground transition hover:brightness-110 disabled:cursor-wait disabled:opacity-70"
          >
            {liveLoading
              ? <LoaderCircle className="h-4 w-4 animate-spin" />
              : <Radio className="h-4 w-4" />}
            {liveLoading ? "Sampling account..." : liveResult ? "Run another account" : "Run live sample"}
          </button>
          <p className="px-2 text-xs leading-relaxed text-muted-foreground">
            Random account from saved held-out predictions; the ensemble score is recomputed for each request. This is not new transaction data.
          </p>
          {liveError && (
            <p role="alert" className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-foreground">
              {liveError}
            </p>
          )}
          {liveResult && (
            <Group title={`Live sample · ${liveResult.account_id}`} bodyClass="p-4">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-xs text-muted-foreground">Ensemble score</p>
                  <p className="mt-1 font-display text-2xl font-semibold tabular-nums text-foreground">{liveResult.score.toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Prediction · threshold {liveResult.threshold.toFixed(2)}</p>
                  <p className="mt-1 text-sm font-medium capitalize text-foreground">{liveResult.predicted_label}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Saved test label</p>
                  <p className="mt-1 text-sm font-medium capitalize text-foreground">{liveResult.test_label}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Node index</p>
                  <p className="mt-1 font-mono text-sm tabular-nums text-foreground">{liveResult.node_id}</p>
                </div>
              </div>
              <div className="mt-4 border-t border-border pt-3">
                <p className="mb-2 text-xs font-medium text-muted-foreground">Model probabilities</p>
                <div className="grid grid-cols-3 gap-2">
                  {Object.entries(liveResult.model_probabilities).map(([model, probability]) => (
                    <div key={model}>
                      <p className="text-[11px] text-muted-foreground">{model}</p>
                      <p className="mt-0.5 font-mono text-sm tabular-nums text-foreground">{probability.toFixed(4)}</p>
                    </div>
                  ))}
                </div>
              </div>
              <p className="mt-3 text-[11px] leading-relaxed text-subtle">The saved test label is dataset ground truth for evaluation, not a live operational finding.</p>
            </Group>
          )}
        </div>
      </div>

      <div className="space-y-6 lg:col-span-3">
        <Group title="Leading drivers" bodyClass="py-1">
          <SignedRows items={top} />
        </Group>
        <Group title="Ensemble weights" bodyClass="pb-1">
          <EnsembleList />
        </Group>
        <Note>
          Explanations describe how sensitive the score is to perturbations. They do not prove illicit activity, and
          this dashboard does not make a legal or causal determination.
        </Note>
      </div>
    </div>
  );
}

function AttributionSection() {
  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <Group title="Feature attribution · this account" bodyClass="py-1">
        <SignedRows items={FEATURE_ATTRIBUTION} />
      </Group>
      <Note>Values show model sensitivity to perturbing each feature, not causes of the score.</Note>
    </div>
  );
}

function EnsembleSection() {
  return (
    <div className="grid gap-4 md:grid-cols-3">
      {ENSEMBLE.map((m, i) => (
        <Group key={m.model} title={i === 0 ? "Ensemble members" : " "}>
          <p className="text-[15px] font-medium text-foreground">{m.model}</p>
          <p className="mt-6 font-display text-5xl font-semibold tracking-tight text-foreground">
            {(m.weight * 100).toFixed(0)}
            <span className="text-2xl text-muted-foreground">%</span>
          </p>
          <p className="text-[13px] text-muted-foreground">Weight in final score</p>
          <div className="mt-5 h-1 overflow-hidden rounded-full bg-surface-raised">
            <div className="h-full rounded-full bg-primary" style={{ width: `${m.weight * 100}%` }} />
          </div>
        </Group>
      ))}

      <Group title="Member performance · held-out test · threshold 0.50" className="md:col-span-3" bodyClass="overflow-x-auto">
        <table className="w-full min-w-[720px] text-sm">
          <thead>
            <tr className="text-left text-xs uppercase tracking-wide text-muted-foreground">
              <th className="px-3 pb-3 font-medium">Model</th>
              <th className="px-3 pb-3 text-right font-medium">PR-AUC</th>
              <th className="px-3 pb-3 text-right font-medium">ROC-AUC</th>
              <th className="px-3 pb-3 text-right font-medium">Precision</th>
              <th className="px-3 pb-3 text-right font-medium">Recall</th>
              <th className="px-3 pb-3 text-right font-medium">F1</th>
              <th className="px-3 pb-3 text-right font-medium">F2</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {ENSEMBLE_TEST_METRICS.map((metric) => (
              <tr key={metric.model}>
                <th scope="row" className="px-3 py-3 text-left font-medium text-foreground">{metric.model}</th>
                <td className="px-3 py-3 text-right tabular-nums text-foreground">{metric.prAuc.toFixed(4)}</td>
                <td className="px-3 py-3 text-right tabular-nums text-foreground">{metric.rocAuc.toFixed(4)}</td>
                <td className="px-3 py-3 text-right tabular-nums text-muted-foreground">{(metric.precision * 100).toFixed(2)}%</td>
                <td className="px-3 py-3 text-right tabular-nums text-muted-foreground">{(metric.recall * 100).toFixed(2)}%</td>
                <td className="px-3 py-3 text-right tabular-nums text-muted-foreground">{metric.f1.toFixed(4)}</td>
                <td className="px-3 py-3 text-right tabular-nums text-muted-foreground">{metric.f2.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Group>

      <Group title="How the final score is combined" className="md:col-span-3">
        <p className="text-[15px] leading-relaxed text-foreground">
          The validation-tuned mixture is <span className="font-mono">0.78 × GraphSAGE + 0.19 × GATv2 + 0.03 × GIN</span>.
          The percentages above are ensemble weights, not each model's accuracy. Member metrics are standalone rankings on the held-out test split at threshold 0.50.
        </p>
        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            ['Ensemble PR-AUC', EXPERIMENTS[0].prAuc.toFixed(4)],
            ['Ensemble ROC-AUC', EXPERIMENTS[0].rocAuc.toFixed(4)],
            ['Ensemble F2 @ 0.50', EXPERIMENTS[0].f2.toFixed(4)],
            ['Accounts flagged', EXPERIMENTS[0].flagged.toLocaleString()],
          ].map(([label, value]) => (
            <div key={label} className="rounded-lg bg-surface-raised/60 px-3 py-2">
              <p className="text-xs text-muted-foreground">{label}</p>
              <p className="mt-1 font-mono text-sm tabular-nums text-foreground">{value}</p>
            </div>
          ))}
        </div>
      </Group>
    </div>
  );
}

function GraphEvidenceSection() {
  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <Group title="Neighbor feature ablation" bodyClass="py-1">
        <SignedRows items={NEIGHBORS} valueKey="effect" labelKey="id" subKey="relation" />
      </Group>
      <Group title="Score response" bodyClass="p-5">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          {[
            ['Baseline score', NEIGHBOR_EXPLANATION.baselineScore.toFixed(4)],
            ['After feature ablation', NEIGHBOR_EXPLANATION.scoreAfterFeatureAblation.toFixed(4)],
            ['Score increase', `+${(NEIGHBOR_EXPLANATION.scoreAfterFeatureAblation - NEIGHBOR_EXPLANATION.baselineScore).toFixed(4)}`],
          ].map(([label, value]) => (
            <div key={label} className="rounded-lg bg-surface-raised/60 px-3 py-2">
              <p className="text-xs text-muted-foreground">{label}</p>
              <p className="mt-1 font-mono text-sm tabular-nums text-foreground">{value}</p>
            </div>
          ))}
        </div>
      </Group>
      <Group title="Transaction context" bodyClass="divide-y divide-border">
        <div className="grid grid-cols-2 gap-4 p-4 sm:grid-cols-3">
          {[
            ['Edge index', NEIGHBOR_EXPLANATION.edgeIndex],
            ['Amount paid', NEIGHBOR_EXPLANATION.amountPaid.toFixed(2)],
            ['Amount received', NEIGHBOR_EXPLANATION.amountReceived.toFixed(2)],
            ['Edge label', NEIGHBOR_EXPLANATION.edgeLabel.toFixed(1)],
            ['Recorded timestamp', NEIGHBOR_EXPLANATION.timestamp],
            ['Transactions', NEIGHBOR_EXPLANATION.transactionCount],
          ].map(([label, value]) => (
            <div key={label}>
              <p className="text-xs text-muted-foreground">{label}</p>
              <p className="mt-1 font-mono text-sm tabular-nums text-foreground">{value}</p>
            </div>
          ))}
        </div>
      </Group>
      <Note>
        The score change is caused by ablating this neighbor's features; the transaction edge remains in the graph. It measures model sensitivity, not evidence of wrongdoing by either account.
      </Note>
    </div>
  );
}

function GlobalSection() {
  const max = Math.max(...GLOBAL_IMPORTANCE.map((g) => g.value));
  return (
    <div className="mx-auto max-w-3xl">
      <Group title="Validation PR-AUC drop · global importance" bodyClass="py-1">
        <ul className="divide-y divide-border">
          {GLOBAL_IMPORTANCE.map((g, i) => (
            <li key={g.name} className="flex items-center gap-4 px-5 py-3.5">
              <span className="w-5 text-[13px] tabular-nums text-subtle">{i + 1}</span>
              <div className="flex-1">
                <div className="flex justify-between text-[15px]">
                  <span className="text-foreground">{g.name}</span>
                  <span className="tabular-nums text-muted-foreground">{g.value.toFixed(4)}</span>
                </div>
                <div className="mt-2 h-1 overflow-hidden rounded-full bg-surface-raised">
                  <div className="h-full rounded-full bg-primary" style={{ width: `${(g.value / max) * 100}%` }} />
                </div>
              </div>
            </li>
          ))}
        </ul>
      </Group>
      <Note>Permutation importance is computed on validation data. Baseline PR-AUC: 0.345188.</Note>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Separate views                                                      */
/* ------------------------------------------------------------------ */

function AnalystSummary({ onBack }) {
  const { up, down } = useSignalCounts();
  const sorted = [...FEATURE_ATTRIBUTION].sort((a, b) => b.impact - a.impact);
  const raise = sorted.filter((f) => f.impact > 0).slice(0, 3);
  const lower = sorted.filter((f) => f.impact < 0).slice(-2).reverse();

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <BackLink onClick={onBack} />
      <LargeTitle caption={`Account ${ACCOUNT.id}`} title="Analyst Summary" />

      <Group>
        <p className="text-[17px] leading-relaxed text-foreground">
          The ensemble assigned this account a score of <strong>{ACCOUNT.score.toFixed(4)}</strong> (
          {(ACCOUNT.score * 100).toFixed(1)}%), placing it in the <strong>medium</strong> risk band. {up} signals
          raised the score and {down} lowered it.
        </p>
      </Group>

      <Group title="Strongest raising signals" bodyClass="py-1">
        <SignedRows items={raise} />
      </Group>
      <Group title="Strongest lowering signals" bodyClass="py-1">
        <SignedRows items={lower} />
      </Group>

      <Note>
        This summary reflects model sensitivity, not proof of illicit activity. Use it to prioritise review; it is
        not a legal or causal determination.
      </Note>
    </div>
  );
}

function GraphsView({ onBack }) {
  const [selected, setSelected] = useState(CHECKPOINTS[0].id);
  const ckpt = CHECKPOINTS.find((c) => c.id === selected);
  const groups = [
    { name: "Full-feature", sub: "26 input features" },
    { name: "QIEA-selected", sub: "12 input features" },
  ];

  const nodes = [
    { label: "Node input", sub: `${ckpt.inputs} standardized account features`, key: true },
    { label: "Message passing · layer 1", sub: ckpt.layer1 },
    { label: "Hidden transform", sub: ckpt.activation },
    { label: "Message passing · layer 2", sub: ckpt.layer2 },
    { label: "Class output", sub: "2 logits · legitimate / illicit", key: true },
  ];

  return (
    <div className="space-y-6">
      <BackLink onClick={onBack} />
      <LargeTitle caption="13 saved checkpoints" title="Model Graphs" />

      <div className="grid gap-6 lg:grid-cols-[300px_minmax(0,1fr)]">
        <div className="space-y-6">
          {groups.map((g) => (
            <Group key={g.name} title={`${g.name} · ${g.sub}`} bodyClass="py-1">
              <ul className="divide-y divide-border">
                {CHECKPOINTS.filter((c) => c.group === g.name).map((c) => (
                  <li key={c.id}>
                    <button
                      onClick={() => setSelected(c.id)}
                      className="flex w-full items-center justify-between px-5 py-2.5 text-left text-[15px] text-foreground transition-colors hover:bg-surface-raised/50"
                    >
                      <span>{c.id}</span>
                      {c.id === selected ? (
                        <Check className="h-4 w-4 text-primary" />
                      ) : (
                        <ChevronRight className="h-4 w-4 text-subtle" />
                      )}
                    </button>
                  </li>
                ))}
              </ul>
            </Group>
          ))}
        </div>

        <div className="space-y-4">
          <Group title={`Architecture · ${ckpt.id} · ${ckpt.group}`}>
            <div className="flex flex-col items-center gap-0 md:flex-row md:items-stretch">
              {nodes.map((n, i) => (
                <div key={n.label} className="flex w-full flex-col items-center md:flex-1 md:flex-row">
                  <div
                    className={`w-full rounded-2xl px-4 py-4 text-center ${
                      n.key ? "bg-accent ring-1 ring-primary/40" : "bg-surface-raised"
                    }`}
                  >
                    <p className={`text-[14px] font-medium ${n.key ? "text-accent-foreground" : "text-foreground"}`}>
                      {n.label}
                    </p>
                    <p className="mt-0.5 text-[12px] text-muted-foreground">{n.sub}</p>
                  </div>
                  {i < nodes.length - 1 && <div className="h-5 w-px bg-subtle md:h-px md:w-4 md:shrink-0" />}
                </div>
              ))}
            </div>
            <p className="mt-4 text-[12px] text-subtle">Layer dimensions and operators are based on checkpoint tensors and model definitions. Checkpoints do not contain the full transaction graph.</p>
          </Group>
          <Note>
            Diagrams are reconstructed from checkpoint tensor shapes. They show model architecture, not every account
            and transaction edge in the data.
          </Note>
        </div>
      </div>
    </div>
  );
}

function QuboView({ onBack }) {
  return (
    <div className="space-y-6">
      <BackLink onClick={onBack} />
      <LargeTitle caption="Quantum-inspired experiments" title="QUBO / QIEA" />

      <Group title="Method comparison" bodyClass="overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[960px] text-[15px]">
            <thead>
              <tr className="text-left text-[12px] uppercase tracking-wide text-muted-foreground">
                <th className="px-5 pb-3 pt-4 font-medium">Method</th>
                <th className="px-3 pb-3 pt-4 font-medium">Threshold</th>
                <th className="px-3 pb-3 pt-4 text-right font-medium">PR-AUC</th>
                <th className="px-3 pb-3 pt-4 text-right font-medium">ROC-AUC</th>
                <th className="px-3 pb-3 pt-4 text-right font-medium">Brier</th>
                <th className="px-3 pb-3 pt-4 text-right font-medium">Precision</th>
                <th className="px-3 pb-3 pt-4 text-right font-medium">Recall</th>
                <th className="px-3 pb-3 pt-4 text-right font-medium">F1</th>
                <th className="px-3 pb-3 pt-4 text-right font-medium">F2*</th>
                <th className="px-5 pb-3 pt-4 text-right font-medium">Flagged*</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border border-t border-border">
              {EXPERIMENTS.map((e) => (
                <tr key={e.name}>
                  <td className="px-5 py-3.5 text-foreground">{e.name}</td>
                  <td className="px-3 py-3.5">
                    <span
                      className={`rounded-full px-2.5 py-0.5 text-[12px] tabular-nums ${
                        e.hybrid ? "bg-accent text-accent-foreground" : "bg-surface-raised text-muted-foreground"
                      }`}
                    >
                      {e.threshold.toFixed(2)} · {e.hybrid ? "validation-selected" : "fixed"}
                    </span>
                  </td>
                  <td className="px-3 py-3.5 text-right tabular-nums text-foreground">{e.prAuc.toFixed(4)}</td>
                  <td className="px-3 py-3.5 text-right tabular-nums text-foreground">{e.rocAuc.toFixed(4)}</td>
                  <td className="px-3 py-3.5 text-right tabular-nums text-muted-foreground">{e.brier == null ? "N/A" : e.brier.toFixed(4)}</td>
                  <td className="px-3 py-3.5 text-right tabular-nums text-muted-foreground">{(e.precision * 100).toFixed(2)}%</td>
                  <td className="px-3 py-3.5 text-right tabular-nums text-muted-foreground">{(e.recall * 100).toFixed(2)}%</td>
                  <td className="px-3 py-3.5 text-right tabular-nums text-muted-foreground">{e.f1.toFixed(4)}</td>
                  <td className="px-3 py-3.5 text-right tabular-nums text-muted-foreground">{e.f2.toFixed(4)}</td>
                  <td className="px-5 py-3.5 text-right tabular-nums text-muted-foreground">{e.flagged == null ? "N/A" : e.flagged.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="border-t border-border px-5 py-3 text-[12px] text-subtle">
          PR-AUC, ROC-AUC, and Brier score are threshold-independent. Precision, recall, F1, F2, and flagged counts depend on threshold; 0.50 and 0.77 rows are not directly comparable.
        </p>
      </Group>

      <div className="grid gap-6 md:grid-cols-2">
        <Group title="QIEA pilot">
          <p className="font-display text-5xl font-semibold tracking-tight text-foreground">
            {QIEA_PILOT.selectedPrAuc.toFixed(4)}
          </p>
          <p className="mt-1 text-[13px] text-muted-foreground">Selected-feature validation PR-AUC from a logistic-regression screening proxy</p>
          <p className="mt-4 text-[14px] text-muted-foreground">
            All-feature PR-AUC: {QIEA_PILOT.allFeaturePrAuc.toFixed(4)} · selected {QIEA_PILOT.selectedFeatures} of {QIEA_PILOT.totalFeatures} features. This is not the final GNN evaluation.
          </p>
        </Group>
        <Group title="Good to know" bodyClass="divide-y divide-border">
          <p className="p-5 text-[14px] leading-relaxed text-muted-foreground">
            Reference and QUBO rows use a fixed <span className="text-foreground">0.50</span> threshold; the QIEA
            hybrid uses a validation-selected <span className="text-foreground">0.77</span> threshold, so F2 and
            flagged counts are not directly comparable.
          </p>
          <p className="p-5 text-[14px] leading-relaxed text-muted-foreground">
            QUBO and QIEA are quantum-inspired classical methods. No quantum hardware was used and no quantum
            advantage is shown.
          </p>
        </Group>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

export default function AmlDashboard() {
  const [view, setView] = useState("dashboard");
  const [section, setSection] = useState("overview");
  const back = () => setView("dashboard");

  const SectionBody = {
    overview: OverviewSection,
    attribution: AttributionSection,
    ensemble: EnsembleSection,
    graph: GraphEvidenceSection,
    global: GlobalSection,
  }[section];

  return (
    <div className="min-h-screen">
      <header className="glass sticky top-0 z-20 border-b border-border">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <button onClick={back} className="flex items-center gap-2.5">
            <span className="flex h-8 w-8 items-center justify-center rounded-[10px] bg-primary">
              <ShieldCheck className="h-[18px] w-[18px] text-primary-foreground" />
            </span>
            <span className="text-[17px] font-semibold tracking-tight text-foreground">AML </span>
          </button>
          <div className="flex flex-wrap gap-2">
            <PillButton icon={GitBranch} active={view === "graphs"} onClick={() => setView("graphs")}>
              Graphs
            </PillButton>
            <PillButton icon={Atom} active={view === "qubo"} onClick={() => setView("qubo")}>
              QUBO / QIEA
            </PillButton>
            <PillButton icon={FileText} primary onClick={() => setView("summary")}>
              Explain account
            </PillButton>
          </div>
        </div>
      </header>

      <main key={view} className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
        {view === "dashboard" && (
          <>
            <LargeTitle
              caption="Risk & Explainability"
              title={ACCOUNT.id}
              right={
                <div className="text-right">
                  <p className="text-[13px] text-muted-foreground">Medium risk</p>
                  <p className="font-display text-2xl font-semibold tabular-nums text-foreground">
                    {ACCOUNT.score.toFixed(4)}
                  </p>
                </div>
              }
            />

            <nav className="scrollbar-none -mx-4 mb-8 overflow-x-auto px-4 sm:mx-0 sm:px-0">
              <div className="inline-flex rounded-full bg-surface p-1">
                {SECTIONS.map(({ key, label }) => (
                  <button
                    key={key}
                    onClick={() => setSection(key)}
                    className={`whitespace-nowrap rounded-full px-4 py-1.5 text-[14px] font-medium transition-all ${
                      section === key
                        ? "bg-surface-raised text-foreground shadow-pill"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </nav>

            <div key={section}>
              <SectionBody />
            </div>
          </>
        )}
        {view === "summary" && <AnalystSummary onBack={back} />}
        {view === "graphs" && <GraphsView onBack={back} />}
        {view === "qubo" && <QuboView onBack={back} />}
      </main>

      <footer className="mx-auto max-w-6xl px-4 pb-10 text-center text-[12px] text-subtle sm:px-6">
        For analyst review only — not a legal or causal determination.
      </footer>
    </div>
  );
}
