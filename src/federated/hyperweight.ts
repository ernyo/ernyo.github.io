// hyperweight.ts

import * as tf from "@tensorflow/tfjs";
import { FederatedClient } from "./client";
import { TaskName } from "../models/model";
import { WeightMap, pick, flatten, deltaDict, fromRelativeDecoderKey } from "./utils"

/** ---------------- Hyperweight (TRAINABLE) ---------------- */
export class HyperweightEncoder {
  private alphaVar: tf.Variable; // [N]
  trainableVariables: tf.Variable[];
  optimizer: tf.Optimizer;

  constructor(numClients: number, lr: number, initAlpha: number) {
    this.alphaVar = tf.variable(tf.fill([numClients], initAlpha, "float32"), true, "enc_alpha");
    this.trainableVariables = [this.alphaVar];
    this.optimizer = tf.train.sgd(lr);
  }

  get alpha(): tf.Tensor {
    return this.alphaVar;
  }

  forward(
    flattenLast: tf.Tensor1D[],     // [N] each [D]
    flattenDelta: tf.Tensor1D[],    // [N] each [D]  (homo-averaged)
    flattenDeltaUpdate: tf.Tensor1D // [D]           (CA delta)
  ): tf.Tensor1D[] {
    const N = flattenLast.length;
    const out: tf.Tensor1D[] = new Array(N);

    for (let i = 0; i < N; i++) {
      out[i] = tf.tidy(() => {
        const ai = tf.clipByValue(this.alphaVar.gather(i), 0, 1); // scalar
        // new = last + delta_i + ai * delta_update
        return flattenLast[i]
          .add(flattenDelta[i])
          .add(flattenDeltaUpdate.mul(ai)) as tf.Tensor1D;
      });
    }
    return out;
  }

  dispose() {
    this.alphaVar.dispose();
    this.trainableVariables.forEach(v => v.dispose());
    this.optimizer.dispose();
  }
}


export class HyperweightDecoder {
  private layerToBetaVar = new Map<string, tf.Variable>(); // layerName -> [K]
  trainableVariables: tf.Variable[] = [];
  optimizer: tf.Optimizer;

  private K: number;
  private initialized = false;
  private initBeta: number;

  constructor(K: number, lr: number, initBeta: number) {
    this.K = K;
    this.initBeta = initBeta;
    this.optimizer = tf.train.sgd(lr);
  }

  // Optional UI exposure similar to paper: beta per layer
  get beta(): Record<string, tf.Tensor> {
    const out: Record<string, tf.Tensor> = {};
    for (const [layer, v] of this.layerToBetaVar.entries()) out[layer] = v;
    return out;
  }

  // Optional UI: list of layer names
  get betaNames(): string[] {
    return Array.from(this.layerToBetaVar.keys()).sort();
  }

  private keyToLayerName(relKey: string): string {
    // relKey example: decoder/stage0_conv0/separable_conv2d/depthwise_kernel
    // layerName should be everything except the last token:
    const parts = relKey.split("/");
    if (parts.length <= 1) return relKey;
    return parts.slice(0, -1).join("/");
  }

  private initFromRelKeys(relKeys: string[]) {
    if (this.initialized) return;

    const layerNames = Array.from(new Set(relKeys.map(k => this.keyToLayerName(k)))).sort();
    for (const layer of layerNames) {
      const v = tf.variable(tf.fill([this.K], this.initBeta, "float32"), true, `dec_beta_${layer.replace("/", "_")}`);
      this.layerToBetaVar.set(layer, v);
      this.trainableVariables.push(v);
    }
    this.initialized = true;
  }

  forward(
    lastBlocks: WeightMap[],  // length K
    deltaBlocks: WeightMap[], // length K
    meta: { taskOfBlock: TaskName[]; clientOfBlock: number[]; relKeys: string[] }
  ): WeightMap[] {
    const { relKeys } = meta;
    const K = lastBlocks.length;
    if (K !== this.K) throw new Error(`HyperCrossAttentionDec: expected K=${this.K}, got ${K}.`);
    this.initFromRelKeys(relKeys); // increases memory by 12
    
    const out: WeightMap[] = Array.from({ length: K }, () => ({}));
    
    // group relKeys by "layer name"
    const layerToKeys = new Map<string, string[]>();
    for (const rk of relKeys) {
      const layer = this.keyToLayerName(rk);
      const arr = layerToKeys.get(layer) ?? [];
      arr.push(rk);
      layerToKeys.set(layer, arr);
    }
    
    for (const [layerName, keys] of layerToKeys.entries()) {
      const betaVar = this.layerToBetaVar.get(layerName); // increases memory by 6
      if (!betaVar) continue;
      
      // clamp beta into [0,1] like PyTorch
      const layerBeta = tf.clipByValue(betaVar, 0, 1); // [K]

      for (const rk of keys) {
        // Find a reference tensor for shape
        let ref: tf.Tensor | null = null;
        for (let j = 0; j < K; j++) {
          ref = deltaBlocks[j][rk] ?? lastBlocks[j][rk] ?? null;
          if (ref) break;
        }
        if (!ref) continue;

        // cross_delta = stack([delta_j_flat]) => [K, D]
        const crossDeltaFlat = tf.tidy(() => {
          const deltas: tf.Tensor[] = new Array(K);
          for (let j = 0; j < K; j++) {
            const dj = deltaBlocks[j][rk];
            deltas[j] = dj ? dj.reshape([-1]) : tf.zerosLike(ref!).reshape([-1]);
          }
          return tf.stack(deltas) as tf.Tensor2D; // [K, D]
        });

        const D = crossDeltaFlat.shape[1];
        const scale = 1 / Math.sqrt(D + 1e-8);

        // For each block i: CrossAttention(self_delta_i, cross_delta, cross_delta)
        
        
        
        for (let i = 0; i < K; i++) {
          const base = lastBlocks[i][rk];
          if (!base) continue;
          
          const newParam = tf.tidy(() => {
            const kT = crossDeltaFlat.transpose();                        // [D, K]
            const selfDelta = deltaBlocks[i][rk] ?? tf.zerosLike(ref);
            const q = selfDelta.reshape([1, -1]) as tf.Tensor2D;          // [1, D]
            const attn = tf.softmax(q.matMul(kT).mul(scale), -1);         // [1, K]
            const outFlat = attn.matMul(crossDeltaFlat);                  // [1, D]
            const outDelta = outFlat.reshape(selfDelta.shape);            // original shape
            
            const beta_i = layerBeta.gather(i);                           // scalar
            const newDelta = selfDelta.add(outDelta.mul(beta_i));
            return base.add(newDelta);
          });

          out[i][rk] = newParam; // increases memory by 1
        }
        
        crossDeltaFlat.dispose(); // decrease memory by 1
      }
      
      layerBeta.dispose(); // decrease memory by 1
    }

    return out;
  }

  dispose() {
    this.layerToBetaVar.forEach(v => v.dispose());
    this.layerToBetaVar.clear();
    this.trainableVariables.forEach(v => v.dispose());
    this.optimizer.dispose();
  }
}


type EncCache = {
  encKeys: string[];
  encShapes: number[][];
  lastEnc: tf.Tensor1D[];
  localDelta: tf.Tensor1D[];
  globalDelta: tf.Tensor1D;
};

type DecCache = {
  relKeys: string[];
  taskOfBlock: TaskName[];
  clientOfBlock: number[];
  lastBlocks: WeightMap[];
  deltaBlocks: WeightMap[];
};

export interface Hyperweight {
  enc?: HyperweightEncoder;
  dec?: HyperweightDecoder;

  /** Stored outputs/inputs from previous aggregate (like last_enc_output / last_dec_output) */
  _cache?: {
    enc?: EncCache;
    dec?: DecCache;
  };
}

/** Dispose cache safely (important to avoid leaks between rounds) */
export function disposeEncCache(c?: EncCache) {
  if (!c) return;
  c.lastEnc.forEach(t => t.dispose());
  c.localDelta.forEach(t => t.dispose());
  c.globalDelta.dispose();
}
export function disposeDecCache(c?: DecCache) {
  if (!c) return;
  for (const b of c.lastBlocks) for (const t of Object.values(b)) t.dispose?.();
  for (const b of c.deltaBlocks) for (const t of Object.values(b)) t.dispose?.();
}

export function disposeHyperweight(hyperweight: Hyperweight) {
  if (hyperweight.enc) {
    disposeEncCache(hyperweight._cache?.enc);
    hyperweight.enc.dispose();
  }
  if (hyperweight.dec) {
    disposeDecCache(hyperweight._cache?.dec);
    hyperweight.dec.dispose();
  }
}



export function createHyperweight(clients: FederatedClient[]) : Hyperweight {
  const N = clients.length;
  const K = clients.reduce((acc, c) => acc + c.tasks.length, 0); // blocks for cross_attention

  return {
    enc: new HyperweightEncoder(N, 1e-3, 0.1),
    dec: new HyperweightDecoder(K, 1e-3, 0.1),
    _cache: {}, // optional
  };
}

/**
 * TFJS equivalent of PyTorch update_hyperweight():
 * - uses hyperweight._cache.{enc,dec} produced in the PREVIOUS aggregate()
 * - uses current (saveCkpt, lastCkpt) to build external gradients (diff = last - current)
 * - applies gradients to hyperweight variables
 */
function addNChunked(xs: tf.Scalar[], chunkSize = 7): tf.Scalar {
  if (xs.length === 0) return tf.scalar(0);

  // Reduce in rounds so every tf.addN sees <= chunkSize inputs.
  return tf.tidy(() => {
    let cur: tf.Scalar[] = xs;
    while (cur.length > 1) {
      const next: tf.Scalar[] = [];
      for (let i = 0; i < cur.length; i += chunkSize) {
        const chunk = cur.slice(i, i + chunkSize);
        next.push(chunk.length === 1 ? chunk[0] : (tf.addN(chunk) as tf.Scalar));
      }
      cur = next;
    }
    return cur[0];
  });
}

export async function updateHyperweight(
  clients: FederatedClient[],
  hyperweight: Hyperweight,
  saveCkpt: WeightMap[],
  lastCkpt: WeightMap[],
) {
  hyperweight._cache ??= {};

  // ----- Encoder hyperweight update -----
  if (hyperweight.enc && hyperweight._cache.enc) {
    const { encKeys, lastEnc, localDelta, globalDelta } = hyperweight._cache.enc;
    
    // flattenDiff[i] = (last - current) encoder params  (matches PyTorch)
    const flattenDiff: tf.Tensor1D[] = [];
    for (let i = 0; i < clients.length; i++) {
      const last = pick(lastCkpt[i], encKeys);
      const cur = pick(saveCkpt[i], encKeys);
      const d = deltaDict(cur, last, encKeys);
      flattenDiff.push(flatten(d, encKeys)); // increases tensor memory by 1

      // dispose deltaDict tensors
      Object.values(d).forEach(t => t.dispose());
    }
    
    const vars = hyperweight.enc.trainableVariables;
    
    const { grads } = tf.variableGrads((): tf.Scalar => tf.tidy(() => {
      const outs = hyperweight.enc!.forward(lastEnc, localDelta, globalDelta);
      const terms = outs.map((o, i) => tf.sum(o.mul(flattenDiff[i])) as tf.Scalar);
      return addNChunked(terms) as tf.Scalar;
    }), vars);

    hyperweight.enc.optimizer.applyGradients(grads);

    // dispose is not doing anything!!
    Object.values(grads).forEach(g => g.dispose());
    flattenDiff.forEach(t => t.dispose());
  }

  // ----- Decoder hyperweight update -----
  if (hyperweight.dec && hyperweight._cache.dec) {
    const { relKeys, taskOfBlock, clientOfBlock, lastBlocks, deltaBlocks } = hyperweight._cache.dec;

    // Build diffBlocks (last - current) in the SAME block order as cache
    const diffBlocks: WeightMap[] = [];
    for (let bi = 0; bi < taskOfBlock.length; bi++) {
      const ci = clientOfBlock[bi];
      const task = taskOfBlock[bi];
      const lastM = lastCkpt[ci];
      const curM = saveCkpt[ci];

      const db: WeightMap = {};
      for (const rk of relKeys) {
        const fullKey = fromRelativeDecoderKey(rk, task);
        const lastT = lastM[fullKey];
        const curT = curM[fullKey];
        if (!lastT || !curT) continue;
        db[rk] = tf.tidy(() => lastT.sub(curT));
      }
      diffBlocks.push(db);
    }

    const vars = hyperweight.dec.trainableVariables;

    const { grads } = tf.variableGrads((): tf.Scalar => tf.tidy(() => {
      const outs = hyperweight.dec!.forward(lastBlocks, deltaBlocks, { taskOfBlock, clientOfBlock, relKeys });

      const terms: tf.Scalar[] = [];
      for (let bi = 0; bi < outs.length; bi++) {
        for (const rk of relKeys) {
          const outT = outs[bi][rk];
          const diffT = diffBlocks[bi][rk];
          if (outT && diffT) terms.push(tf.sum(outT.mul(diffT)) as tf.Scalar);
        }
      }
      return addNChunked(terms) as tf.Scalar;
    }), vars);

    hyperweight.dec.optimizer.applyGradients(grads);
    Object.values(grads).forEach(g => g.dispose());
    diffBlocks.forEach(b => Object.values(b).forEach(t => t.dispose()));
  }
}