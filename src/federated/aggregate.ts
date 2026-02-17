import * as tf from "@tensorflow/tfjs";
import type { TaskName } from "../models/model";
import { FederatedClient } from "./client";
import {
  EPS,
  getEncoderKeys,
  getDecoderKeysForTask,
  WeightMap,
  cloneWeightMap,
  pick,
  flatten,
  unflatten,
  deltaDict,
  meanSoup,
  toRelativeDecoderKey,
  fromRelativeDecoderKey,
  solveSimplexProjectedGD,
  loadCheckpoint,
  cloneBlock,
} from "./utils";
import { Hyperweight, disposeDecCache, disposeEncCache } from "./hyperweight";

/** ---------------- Types ---------------- */
export type EncoderAgg = "none" | "fedavg" | "conflict_averse";
export type DecoderAgg = "none" | "fedavg" | "cross_attention";


/** ---------------- Homo aggregation for encoder deltas ---------------- */
function homoAverageEncoderDeltasInPlace(deltas: tf.Tensor1D[], clients: FederatedClient[]) {
  const N = clients.length;
  let start = 0;

  while (start < N) {
    let end = start + 1;
    while (
      end < N &&
      clients[end].dataname === clients[end - 1].dataname &&
      clients[end].tasks.join("|") === clients[end - 1].tasks.join("|")
    ) end++;

    // average deltas[start:end]
    const avg = tf.tidy(() => tf.addN(deltas.slice(start, end)).div(end - start)) as tf.Tensor1D;

    // dispose old tensors and replace with clones
    for (let i = start; i < end; i++) {
      deltas[i].dispose();
      deltas[i] = (i === start) ? avg : avg.clone();
    }

    // we kept avg as deltas[start], so don't dispose it here
    start = end;
  }
}


function getCaDelta(flattenDeltaList: tf.Tensor1D[], alpha: number): tf.Tensor1D {
  return tf.tidy(() => {
    const N = flattenDeltaList.length;
    const grads = tf.stack(flattenDeltaList).transpose();     // [d, N]
    const GG = grads.transpose().matMul(grads);               // [N, N]
    const g0Norm = tf.sqrt(GG.mean().add(EPS));
    const A = GG.arraySync() as number[][];
    const c = alpha * (g0Norm.arraySync() as number) + EPS;

    // solver returns Float64Array
    const ww64 = solveSimplexProjectedGD(A, c) as Float64Array;
    const ww = tf.tensor1d(Array.from(ww64), "float32");      // [N]

    const gw = grads.mul(ww.reshape([1, N])).sum(1);          // [d]
    const lambda = tf.scalar(c).div(tf.norm(gw).add(EPS));
    const g = grads.mean(1).add(gw.mul(lambda));

    // rescale=1
    return g.div(tf.scalar(1 + alpha * alpha)) as tf.Tensor1D;
  });
}


/** ---------------- Main aggregate ---------------- */

export async function aggregate(
  clients: FederatedClient[],
  saveCkpt: WeightMap[],
  lastCkpt: WeightMap[],
  hyperweight: Hyperweight,
  encoderAgg: EncoderAgg,
  decoderAgg: DecoderAgg,
  caC = 0.4,
) {
  const N = clients.length;
  const update: WeightMap[] = saveCkpt.map(m => cloneWeightMap(m)); // increases memory
  const allKeys0 = Object.keys(saveCkpt[0]);

  hyperweight._cache ??= {};

  // ----- Encoder -----
  if (encoderAgg !== "none") {
    const encKeys = getEncoderKeys(allKeys0);
    const encShapes = encKeys.map(k => saveCkpt[0][k].shape.slice());

    if (encoderAgg === "fedavg") {
      const encDicts = saveCkpt.map(m => pick(m, encKeys));
      const encAvg = meanSoup(encDicts, encKeys);
      for (let i = 0; i < N; i++) {
        for (const k of encKeys) {
          const old = update[i][k];
          if (old && old !== encAvg[k]) old.dispose();
          update[i][k] = encAvg[k];
        }
      }
    }

    if (encoderAgg === "conflict_averse") {
      // Compute per-client flattened last encoder and deltas
      const flattenLast: tf.Tensor1D[] = [];
      const flattenDelta: tf.Tensor1D[] = [];

      for (let i = 0; i < N; i++) {
        const cur = pick(saveCkpt[i], encKeys);
        const last = pick(lastCkpt[i], encKeys);
        const d = deltaDict(cur, last, encKeys);           // cur - last
        flattenDelta.push(flatten(d, encKeys));
        flattenLast.push(flatten(last, encKeys));

        // dispose deltaDict tensors
        Object.values(d).forEach(t => t.dispose());
      }

      // Global CA delta (same for all)
      const globalDelta = getCaDelta(flattenDelta, caC);
      
      // Homo aggregation on local deltas (matches PyTorch)
      homoAverageEncoderDeltasInPlace(flattenDelta, clients);
      
      if (hyperweight.enc?.forward) {
        // Replace previous cache (avoid leaks)
        disposeEncCache(hyperweight._cache.enc); // decreases memory by 7
        hyperweight._cache.enc = { // increases memory by 7
          encKeys,
          encShapes,
          lastEnc: flattenLast.map(t => t.clone()),
          localDelta: flattenDelta.map(t => t.clone()),
          globalDelta: globalDelta.clone(),
        };
        
        // Personalized new encoder
        const flattenNew = hyperweight.enc.forward(flattenLast, flattenDelta, globalDelta);
        
        for (let i = 0; i < N; i++) {
          const newEnc = unflatten(flattenNew[i], encKeys, encShapes);
          for (const k of encKeys) {
            const old = update[i][k];
            if (old && old !== newEnc[k]) old.dispose();
            update[i][k] = newEnc[k];
          }
        }
        flattenNew.forEach(t => t.dispose());
      } else {
        // No hyperweight: apply global delta to each
        for (let i = 0; i < N; i++) {
          const newVec = tf.tidy(() => flattenLast[i].add(globalDelta)) as tf.Tensor1D;
          const newEnc = unflatten(newVec, encKeys, encShapes);
          for (const k of encKeys) {
            const old = update[i][k];
            if (old && old !== newEnc[k]) old.dispose();
            update[i][k] = newEnc[k];
          }
        }
      }

      // local temporaries can go away (cache clones persist)
      flattenLast.forEach(t => t.dispose());
      flattenDelta.forEach(t => t.dispose());
      globalDelta.dispose();
    }
  }

  // ----- Decoder + Head per task block -----
  if (decoderAgg !== "none") {
    const blocks: { clientIdx: number; task: TaskName }[] = [];
    const relKeySet = new Set<string>();

    for (let i = 0; i < N; i++) {
      for (const task of clients[i].tasks as TaskName[]) {
        const full = getDecoderKeysForTask(Object.keys(saveCkpt[i]), task);
        for (const fk of full) relKeySet.add(toRelativeDecoderKey(fk, task));
        blocks.push({ clientIdx: i, task });
      }
    }

    const relKeys = Array.from(relKeySet).sort();

    if (decoderAgg === "fedavg") {
      if (blocks.length === 0) return;

      // --- Canonical relKeys come ONLY from the first block (matches Python get_model_soup behavior)
      const b0 = blocks[0];
      const full0 = getDecoderKeysForTask(Object.keys(saveCkpt[b0.clientIdx]), b0.task);
      const relKeys = full0.map(fk => toRelativeDecoderKey(fk, b0.task)).sort();
      
      const dicts: WeightMap[] = blocks.map(b => {
        const m = saveCkpt[b.clientIdx];
        const d: WeightMap = {};
        for (const rk of relKeys) d[rk] = m[fromRelativeDecoderKey(rk, b.task)];
        return d;
      });

      const avg = meanSoup(dicts, relKeys);

      for (const b of blocks) {
        const m = update[b.clientIdx];
        for (const rk of relKeys) {
          const fk = fromRelativeDecoderKey(rk, b.task);
          const old = m[fk];
          if (old && old !== avg[rk]) old.dispose();
          m[fk] = avg[rk].clone();
        }
      }
      Object.values(avg).forEach(t => t.dispose());
    }

    if (decoderAgg === "cross_attention") {
      if (!hyperweight.dec?.forward) {
        throw new Error(`decoderAgg="cross_attention" requires hyperweight.dec.forward(...)`);
      }

      // Build block-wise last + delta (cur - last)
      const lastBlocks: WeightMap[] = [];
      const deltaBlocks: WeightMap[] = [];
      const taskOfBlock: TaskName[] = [];
      const clientOfBlock: number[] = [];

      for (const b of blocks) { // increases memory by 72
        const lastM = lastCkpt[b.clientIdx];
        const curM = saveCkpt[b.clientIdx];

        const lb: WeightMap = {};
        const db: WeightMap = {};

        for (const rk of relKeys) {
          const fullKey = fromRelativeDecoderKey(rk, b.task);
          const lastT = lastM[fullKey];
          const curT = curM[fullKey];
          if (!lastT || !curT) continue;

          lb[rk] = lastT.clone();
          db[rk] = tf.tidy(() => curT.sub(lastT)); // delta = cur - last
        }
        
        lastBlocks.push(lb);
        deltaBlocks.push(db);
        taskOfBlock.push(b.task);
        clientOfBlock.push(b.clientIdx);
      }
      
      // Cache for NEXT ROUND updateHyperweight()
      disposeDecCache(hyperweight._cache.dec); // decreases memory by 72
      
      hyperweight._cache.dec = { // increases memory by 72
        relKeys,
        taskOfBlock,
        clientOfBlock,
        lastBlocks: lastBlocks.map(b => cloneBlock(b)),
        deltaBlocks: deltaBlocks.map(b => cloneBlock(b)),
      };
      
      const newBlocks = hyperweight.dec.forward(lastBlocks, deltaBlocks, { taskOfBlock, clientOfBlock, relKeys }); // increases memory by 48
      
      // Write back
      for (let bi = 0; bi < blocks.length; bi++) {
        const b = blocks[bi];
        const m = update[b.clientIdx];
        const nb = newBlocks[bi];

        for (const rk of Object.keys(nb)) {
          const fk = fromRelativeDecoderKey(rk, b.task);
          const old = m[fk];
          if (old) old.dispose();
          m[fk] = nb[rk].clone();
        }
      }

      for (const nb of newBlocks) { // decreases memory by 36
        for (const t of Object.values(nb)) t.dispose();
      }
      
      // dispose temporaries (cache clones persist)
      lastBlocks.forEach(b => Object.values(b).forEach(t => t.dispose()));
      deltaBlocks.forEach(b => Object.values(b).forEach(t => t.dispose())); // decreases memory by 72
    }
  }

  // ----- Load into TFJS models -----
  for (let i = 0; i < N; i++) loadCheckpoint(clients[i].model.model, update[i]);

  // Dispose staging tensors
  const seen = new Set<number>();
  for (const m of update) {
    for (const t of Object.values(m)) {
      const id = (t as any).id; // TFJS internal; works but not “official”
      if (id != null && seen.has(id)) continue;
      if (id != null) seen.add(id);
      t.dispose();
    }
  }
}
