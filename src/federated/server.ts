// server.ts
import * as tf from "@tensorflow/tfjs";
import { FederatedClient } from "./client";
import { aggregate, EncoderAgg, DecoderAgg } from "./aggregate";
import { exportCheckpoint, WeightMap, disposeWeightMap, tensorOrNumberArray, toCanonicalMap } from "./utils";
import { Hyperweight, updateHyperweight, disposeHyperweight, createHyperweight } from "./hyperweight";
import { computeInterClientDiagnostics } from "./diagnostics";

/**
 * Optional hooks for UI.
 */
export interface FederatedServerCallbacks {
  /** Called after each client finishes local training for a round. */
  onClientTrained?: (payload: {
    round: number;
    clientId: string | number;
    metrics?: Record<string, any>;
  }) => void;

  /** Called after aggregation is applied to clients' models. */
  onAggregated?: (payload: {
    round: number;
    encoderAgg: EncoderAgg;
    decoderAgg: DecoderAgg;
    alpha?: number[];
    beta?: number[];
    delta?: number[][];
  }) => void;

  /** Called periodically with evaluation results you compute. */
  onEvaluated?: (payload: {
    round: number;
    results: Record<string, any>;
  }) => void;

  /** Called when server emits a checkpoint snapshot for UI persistence. */
  onCheckpoint?: (payload: {
    round: number;
    checkpoint: WeightMap[]; // one per client
  }) => void;

  /** Called for progress updates. */
  onProgress?: (payload: {
    round: number;
    phase: "train" | "aggregate" | "eval";
    message?: string;
  }) => void;
}

/**
 * FederatedServer is NOT an "experiment runner".
 * It's a stateful object you can drive from a UI:
 * - change config between rounds
 * - step() one round at a time
 * - run(n) multiple rounds
 * - pause/cancel
 */
export class FederatedServer {
  round = 0;
  lastDiagnostics: ReturnType<typeof computeInterClientDiagnostics> | null = null;

  /** Configurable parameters; can be updated live. */
  epochsPerClient: number = 1;
  encoderAgg: EncoderAgg = "none";
  decoderAgg: DecoderAgg = "none";
  caC: number = 0.0;
  evalEvery: number = 1;
  checkpointEvery: number = 1;
  logAlphaBeta: boolean = false;

  /** Optional hyperweights (if you implement meta-learning). */
  hyperweight: Hyperweight;

  /** Latest hyperweight values exposed for UI display. */
  alpha: number[] = [];
  beta: number[] = [];
  delta: number[][] = [];

  /** History for UI charts (optional). */
  history: {
    alpha: number[][];
    beta: number[][];
    rounds: number[];
  } = { alpha: [], beta: [], rounds: [] };

  /** Internal state. */
  private lastCkpt: WeightMap[] | null = null;

  get snapshotConfig() {
    return {
      epochsPerClient: this.epochsPerClient,
      encoderAgg: this.encoderAgg,
      decoderAgg: this.decoderAgg,
      caC: this.caC,
    }
  }

  constructor(
    hyperweight: Hyperweight,
    private callbacks: FederatedServerCallbacks = {}
  ) {
    this.hyperweight = hyperweight;
  }

  async reset(clients: FederatedClient[]) {
    // clear server bookkeeping
    this.round = 0;
    this.lastDiagnostics = null;

    // dispose old checkpoints
    if (this.lastCkpt) {
      this.lastCkpt.forEach(disposeWeightMap);
      this.lastCkpt = null;
    }

    // clear hyperweight history/derived values
    disposeHyperweight(this.hyperweight);
    this.hyperweight = createHyperweight(clients);
    this.alpha = [];
    this.beta = [];
    this.delta = [];
    this.history = { alpha: [], beta: [], rounds: [] };
  }

  /** One federated round (train -> (optional) hyperweight update -> aggregate -> (optional) eval). */
  async step(clients: FederatedClient[]) {
    console.log("before", tf.memory());
    const r = this.round;
    
    // Initialize lastCkpt on first round
    if (!this.lastCkpt) {
      this.lastCkpt = clients.map(c => toCanonicalMap(exportCheckpoint(c.model.model))); // increases memory by 93 tensors
    }
    this.callbacks.onProgress?.({ round: r, phase: "train", message: "Training clients" });

    // --- 1) Local training (sequential; you can parallelize if your environment allows)
    for (const client of clients) {
      await client.update(this.epochsPerClient);
      // optional metrics hook if your client exposes them
      this.callbacks.onClientTrained?.({
        round: r,
        clientId: client.id,
        metrics: client.lastMetrics,
      });
    }

    // --- 2) Collect checkpoints
    const saveCkpt = clients.map((c) => toCanonicalMap(exportCheckpoint(c.model.model))); // increases memory by 93 tensors
    this.lastDiagnostics = computeInterClientDiagnostics(clients, saveCkpt, this.lastCkpt!);

    // --- 3) Hyperweight update (optional)
    // In your Python loop, hyperweights update starting from cr > 0.
    if (r > 0 && this.hyperweight) {
      await updateHyperweight(clients, this.hyperweight, saveCkpt, this.lastCkpt); // increases memory by 1 tensor excluding first round
    }
    this.refreshAlphaBetaDelta();
    this.maybeLogAlphaBeta();

    // --- 4) Aggregate 
    this.callbacks.onProgress?.({ round: r, phase: "aggregate", message: "Aggregating checkpoints" });
    await aggregate( // increases memory by 319 tensors in first round, then 228 tensors
      clients,
      saveCkpt,
      this.lastCkpt,
      this.hyperweight,
      this.encoderAgg,
      this.decoderAgg,
      this.caC,
    );

    // --- 5) Update lastCkpt for next round
    saveCkpt.forEach(disposeWeightMap); // decreases memory by 93 tensors
    this.lastCkpt.forEach(disposeWeightMap); // decreases memory by 93 tensors
    this.lastCkpt = clients.map(c => toCanonicalMap(exportCheckpoint(c.model.model))); // increases memory by 93 tensors

    // Notify UI that aggregation happened
    this.callbacks.onAggregated?.({
      round: r,
      encoderAgg: this.encoderAgg,
      decoderAgg: this.decoderAgg,
      alpha: this.alpha.length ? [...this.alpha] : undefined,
      beta: this.beta.length ? [...this.beta] : undefined,
      delta: this.delta.length ? this.delta.map((row) => [...row]) : undefined,
    });

    this.round++;
    console.log("after", tf.memory());
  }

  /** Replace callbacks (UI can rebind). */
  setCallbacks(cb: FederatedServerCallbacks) {
    this.callbacks = cb;
  }

  // --------------------- internals ---------------------
  private refreshAlphaBetaDelta() {
    // Expose alpha/beta in plain JS arrays for UI.
    const enc = this.hyperweight?.enc;
    const dec = this.hyperweight?.dec;

    if (enc?.alpha != null) {
      this.alpha = tensorOrNumberArray(enc.alpha);
    }
    if (dec?.beta != null) {
      // common patterns: beta is a tensor, number[], or record of tensors
      if (dec.beta instanceof tf.Tensor || Array.isArray(dec.beta)) {
        this.beta = tensorOrNumberArray(dec.beta as any);
      } else if (typeof dec.beta === "object" && dec.beta) {
        const names = Object.keys(dec.beta);
        const flat: number[] = [];
        for (const n of names) flat.push(...tensorOrNumberArray((dec.beta as any)[n]));
        this.beta = flat;
      }
    }

    // delta: if you have a cross-task routing matrix you want to expose,
    // store it on hyperweight.dec (or elsewhere) and read it here.
    const maybeDelta = (this.hyperweight as any)?.dec?.delta ?? (this.hyperweight as any)?.delta;
    if (maybeDelta) this.delta = maybeDelta;
  }

  private maybeLogAlphaBeta() {
    if (!this.logAlphaBeta) return;
    this.history.rounds.push(this.round);
    this.history.alpha.push([...this.alpha]);
    this.history.beta.push([...this.beta]);
  }
}

// export class ServerSnapshot {
//   round: number;
//   alpha: number[];
//   beta: number[];
//   delta: number[][];
//   encoderAgg: EncoderAgg;
//   decoderAgg: DecoderAgg;
//   caC: number;
// }