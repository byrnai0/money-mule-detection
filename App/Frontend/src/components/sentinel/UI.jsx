import { motion } from 'framer-motion';
import { ArrowDownRight, ArrowUpRight } from 'lucide-react';

export function PageHeader({
  eyebrow,
  title,
  description,
  children,
}) {
  return (
    <div className="mb-2 flex flex-col justify-between gap-2 border-b border-slate-800/80 pb-2 lg:flex-row lg:items-end">
      <div>
        <p className="text-[10px] font-bold uppercase tracking-[0.28em] text-cyan-400">
          {eyebrow}
        </p>
        <h1 className="mt-2 text-2xl font-semibold leading-tight text-white xl:whitespace-nowrap">
          {title}
        </h1>
        <p className="mt-2 max-w-3xl text-sm text-slate-500">{description}</p>
      </div>

      {children && <div className="flex flex-wrap gap-2">{children}</div>}
    </div>
  );
}

export function Panel({
  title,
  subtitle,
  action,
  children,
  className = '',
  compact = false,
}) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ borderColor: 'rgba(34,211,238,0.18)' }}
      className={`min-w-0 rounded-2xl border border-slate-800/80 bg-[#071520]/80 shadow-[0_20px_60px_rgba(0,0,0,0.18)] ${className}`}
    >
      {(title || action) && (
        <div className={`flex items-start justify-between gap-4 border-b border-slate-800/70 ${compact ? 'px-4 py-2' : 'px-5 py-4'}`}>
          <div>
            <h2 className="text-base font-semibold leading-6 text-white">{title}</h2>
            {subtitle && (
              <p className="mt-1 text-sm leading-5 text-slate-400">{subtitle}</p>
            )}
          </div>
          {action}
        </div>
      )}

      <div className={compact ? 'p-3' : 'p-5'}>{children}</div>
    </motion.section>
  );
}

export function StatCard({
  label,
  value,
  change,
  positive = true,
  icon: Icon,
  color = 'cyan',
}) {
  const colors = {
    cyan: 'border-cyan-400/20 bg-cyan-400/10 text-cyan-300',
    emerald: 'border-emerald-400/20 bg-emerald-400/10 text-emerald-300',
    rose: 'border-rose-400/20 bg-rose-400/10 text-rose-300',
    amber: 'border-amber-400/20 bg-amber-400/10 text-amber-300',
    violet: 'border-violet-400/20 bg-violet-400/10 text-violet-300',
  };

  return (
    <motion.div
      whileHover={{ y: -4 }}
      transition={{ type: 'spring', stiffness: 300 }}
      className="rounded-2xl border border-slate-800/80 bg-[#071520]/80 p-5 transition hover:border-cyan-400/20"
    >
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
          {label}
        </span>

        {Icon && (
          <span
            className={`flex h-9 w-9 items-center justify-center rounded-xl border ${colors[color]}`}
          >
            <Icon size={17} />
          </span>
        )}
      </div>

      <div className="mt-4 text-2xl font-bold text-white">{value}</div>

      {change && (
        <div
          className={`mt-2 flex items-center gap-1 text-xs ${
            positive ? 'text-emerald-400' : 'text-rose-400'
          }`}
        >
          {positive ? (
            <ArrowUpRight size={14} />
          ) : (
            <ArrowDownRight size={14} />
          )}
          {change}
        </div>
      )}
    </motion.div>
  );
}

export function Badge({ children, tone = 'slate' }) {
  const tones = {
    critical: 'border-rose-400/25 bg-rose-400/10 text-rose-300',
    high: 'border-orange-400/25 bg-orange-400/10 text-orange-300',
    medium: 'border-amber-400/25 bg-amber-400/10 text-amber-300',
    low: 'border-emerald-400/25 bg-emerald-400/10 text-emerald-300',
    cyan: 'border-cyan-400/25 bg-cyan-400/10 text-cyan-300',
    violet: 'border-violet-400/25 bg-violet-400/10 text-violet-300',
    slate: 'border-slate-700 bg-slate-800/60 text-slate-300',
  };

  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

export function ProgressBar({ value, color = '#22d3ee' }) {
  return (
    <div className="h-2 overflow-hidden rounded-full bg-slate-800">
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${value}%` }}
        transition={{ duration: 0.8, ease: 'easeOut' }}
        className="h-full rounded-full"
        style={{
          background: color,
          boxShadow: `0 0 12px ${color}55`,
        }}
      />
    </div>
  );
}

export const buttonPrimary =
  'inline-flex items-center justify-center gap-2 rounded-xl border border-cyan-300/20 bg-cyan-400/10 px-4 py-2.5 text-xs font-semibold text-cyan-200 transition hover:-translate-y-0.5 hover:bg-cyan-400/20 hover:shadow-[0_10px_30px_rgba(34,211,238,0.12)]';

export const buttonSecondary =
  'inline-flex items-center justify-center gap-2 rounded-xl border border-slate-700 bg-slate-900/70 px-4 py-2.5 text-xs font-semibold text-slate-300 transition hover:border-slate-600 hover:bg-slate-800 hover:text-white';

export const inputClass =
  'rounded-xl border border-slate-800 bg-slate-950/70 px-4 py-2.5 text-sm text-slate-200 outline-none placeholder:text-slate-600 focus:border-cyan-400/40 focus:ring-4 focus:ring-cyan-400/5';