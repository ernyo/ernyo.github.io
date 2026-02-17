import * as tf from "@tensorflow/tfjs";
import type { TaskName } from "../models/model";

export type WeightMap = Record<string, tf.Tensor>;

export const EPS = 1e-8;

/** ---------------- Checkpoint I/O ---------------- */

export function exportCheckpoint(model: tf.LayersModel): WeightMap {
  const names = model.weights.map(w => w.name);
  const values = model.getWeights();

  const ckpt: WeightMap = {};
  for (let i = 0; i < names.length; i++) ckpt[names[i]] = values[i].clone();

  return ckpt;
}

export function loadCheckpoint(
  model: tf.LayersModel,
  ckpt: WeightMap,
) {
  const names = model.weights.map(w => w.name);
  const current = model.getWeights();

  const next: tf.Tensor[] = new Array(names.length);
  for (let i = 0; i < names.length; i++) {
    const k = names[i];
    const ck = canonicalKey(k);
    const t = ckpt[ck];
    next[i] = t ? t : current[i];
  }

  model.setWeights(next);
}


/** ---------------- Key extraction ---------------- */
function canonicalKey(k: string): string {
  // strips TFJS auto-suffix like "_1/" before the slash
  return k.replace(/_\d+$/, "");
}

export function toCanonicalMap(m: WeightMap): WeightMap {
  const out: WeightMap = {};
  for (const [key, t] of Object.entries(m)) {
    const ck = canonicalKey(key);
    if (out[ck]) {
      // Optional: fail fast on collisions so you don't silently drop weights
      throw new Error(`toCanonicalMap collision: "${ck}" from "${key}"`);
    }
    out[ck] = t; // keep same tensor reference
  }
  return out;
}

export function getEncoderKeys(allKeys: string[]): string[] {
  // matches encoder_stage{X}_conv{Y}/...
  return allKeys.filter(k => k.startsWith("encoder_stage")).sort();
}

export function getDecoderKeysForTask(allKeys: string[], task: TaskName): string[] {
  // matches `${task}_decoder_stage.../...`
  const decPrefix = `${task}_decoder_`; // works because names are `${task}_decoder_stage...`
  return allKeys.filter(k => k.startsWith(decPrefix)).sort();
}

/** For cross-task decoder alignment we strip the task prefix */
export function toRelativeDecoderKey(fullKey: string, task: TaskName): string {
  const decPrefix = `${task}_decoder_`;
  if (!fullKey.startsWith(decPrefix)) throw new Error(`Not a decoder key: ${fullKey}`);
  return `decoder/${fullKey.slice(decPrefix.length)}`; // e.g. decoder/stage0_conv0/depthwise_kernel
}

export function fromRelativeDecoderKey(relKey: string, task: TaskName): string {
  if (!relKey.startsWith("decoder/")) throw new Error(`Bad rel decoder key: ${relKey}`);
  return `${task}_decoder_${relKey.slice("decoder/".length)}`;
}


/** ---------------- Dict ops ---------------- */

export function meanSoup(dicts: WeightMap[], keys: string[]): WeightMap {
  const out: WeightMap = {};
  for (const k of keys) out[k] = tf.tidy(() => tf.stack(dicts.map(d => d[k])).mean(0));
  return out;
}

export function deltaDict(cur: WeightMap, last: WeightMap, keys: string[]): WeightMap {
  const out: WeightMap = {};
  for (const k of keys) out[k] = tf.tidy(() => cur[k].sub(last[k]));
  return out;
}

export function flatten(dict: WeightMap, keys: string[]): tf.Tensor1D {
  return tf.tidy(() => tf.concat(keys.map(k => dict[k].flatten())) as tf.Tensor1D);
}

export function unflatten(vec: tf.Tensor1D, keys: string[], shapes: number[][]): WeightMap {
  return tf.tidy(() => {
    const out: WeightMap = {};
    let start = 0;

    for (let i = 0; i < keys.length; i++) {
      const size = shapes[i].reduce((a, b) => a * b, 1);
      out[keys[i]] = vec.slice([start], [size]).reshape(shapes[i]);
      start += size;
    }
    return out;
  });
}


export function pick(m: WeightMap, keys: string[]): WeightMap {
  const out: WeightMap = {};
  for (const k of keys) out[k] = m[k];
  return out;
}

/** ---------------- Conflict-averse delta ---------------- */

function projectToSimplex(v: Float64Array): Float64Array {
  const n = v.length;
  const u = Array.from(v).sort((a, b) => b - a);
  let cssv = 0, rho = -1;
  for (let i = 0; i < n; i++) {
    cssv += u[i];
    const t = (cssv - 1) / (i + 1);
    if (u[i] - t > 0) rho = i;
  }
  const theta = (u.slice(0, rho + 1).reduce((s, x) => s + x, 0) - 1) / (rho + 1);
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = Math.max(v[i] - theta, 0);
  return w;
}

function matVec(A: number[][], x: Float64Array): Float64Array {
  const n = A.length;
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += A[i][j] * x[j];
    y[i] = s;
  }
  return y;
}

function dotArr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function quadForm(A: number[][], x: Float64Array): number {
  const Ax = matVec(A, x);
  return dotArr(x, Ax);
}

function objCA(A: number[][], b: Float64Array, c: number, x: Float64Array): number {
  const Ab = matVec(A, b);
  return dotArr(x, Ab) + c * Math.sqrt(quadForm(A, x) + EPS);
}

function gradCA(A: number[][], b: Float64Array, c: number, x: Float64Array): Float64Array {
  const Ab = matVec(A, b);
  const Ax = matVec(A, x);
  const denom = Math.sqrt(quadForm(A, x) + EPS);
  const g = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) g[i] = Ab[i] + (c * Ax[i]) / denom;
  return g;
}

export function solveSimplexProjectedGD(A: number[][], c: number, maxIters = 500, tol = 1e-9): Float64Array {
  const n = A.length;
  let x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = 1 / n;
  const b = x;

  let fx = objCA(A, b, c, x);
  let step = 0.1;

  for (let it = 0; it < maxIters; it++) {
    const g = gradCA(A, b, c, x);
    const z = new Float64Array(n);
    for (let i = 0; i < n; i++) z[i] = x[i] - step * g[i];
    const xNew = projectToSimplex(z);
    const fNew = objCA(A, b, c, xNew);

    if (fNew > fx + 1e-12) { step *= 0.5; if (step < 1e-12) break; continue; }

    let diff1 = 0;
    for (let i = 0; i < n; i++) diff1 += Math.abs(xNew[i] - x[i]);
    x.set(xNew); fx = fNew; step = Math.min(step * 1.05, 1.0);
    if (diff1 < tol) break;
  }
  return x;
}


/** -------------------- small tensor helpers -------------------- */

export function tensorOrNumberArray(x: tf.Tensor | number[]): number[] {
  if (Array.isArray(x)) return x.slice();
  // tensor: sync OK for small vectors; for large, use async data()
  return Array.from(x.dataSync());
}

export function disposeWeightMap(wm: WeightMap) {
  for (const k of Object.keys(wm)) wm[k].dispose();
}

/** Clone a block dict safely */
export function cloneBlock(b: WeightMap): WeightMap {
  const out: WeightMap = {};
  for (const [k, t] of Object.entries(b)) out[k] = t.clone();
  return out;
}

export function cloneWeightMap(wm: WeightMap): WeightMap {
  const out: WeightMap = {};
  for (const k of Object.keys(wm)) out[k] = wm[k].clone();
  return out;
}