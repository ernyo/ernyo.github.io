import * as tf from '@tensorflow/tfjs';
import { Req, requestBatchFromWorker } from './worker';
import { batchToTensors } from '../utils/tensors';
import { TaskName } from '../models/model';
import { Shape } from './constants';

export const NUM_SAMPLES = 100;

export interface DataConfig {
  seed: number;
  numSamples: number;
  numShapes: number;
  typeShapes: Shape[];
  resolution: number;
  batchSize: number;
  percTrain: number;
}

export class Dataloader {
  private config: DataConfig;
  private tasks: TaskName[];
  trainSet: tf.data.Dataset<{ xs: tf.Tensor4D; ys: tf.Tensor[] }>;
  testXs: tf.Tensor4D;
  testYs: tf.Tensor[];
  gtImages: ImageData[]; // ground truth images for test set
  private worker: Worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  private trainGenFactory: () => AsyncGenerator<{ xs: tf.Tensor4D; ys: tf.Tensor[] }, void, unknown>;
  ready: Promise<void>;

  constructor(config: DataConfig, tasks: TaskName[]) {
    this.ready = this.reset(tasks, config);
  }

  private generateBatch(worker: Worker, baseReq: Omit<Req, "seed">, seed: number, tasks: TaskName[]) {
    return async function* () {
      let count = 0;

      while (true) {
        const samplesPerBatch = baseReq.batch;
        const startSampleIdx = (count * samplesPerBatch) % NUM_SAMPLES;

        const req: Req = { ...baseReq, seed: seed + startSampleIdx };
        const batch = await requestBatchFromWorker(worker, req);

        const { xs, ys } = batchToTensors(batch, tasks);

        yield { xs, ys };
        count++;
      }
    };
  }

  private async generateTestData() {
    // Reset old data
    if (this.testXs) this.testXs.dispose();
    if (this.testYs) this.testYs.forEach(t => t.dispose());
  
    let numSamples = Math.floor((100 - this.config.percTrain) * NUM_SAMPLES / 100);
    numSamples = Math.max(1, numSamples);
    const baseReq: Omit<Req, 'seed'> = {
      H: this.config.resolution,
      W: this.config.resolution,
      nShapes: this.config.numShapes,
      typeShapes: this.config.typeShapes,
      batch: numSamples,
    };
    const req: Req = {
      ...baseReq,
      seed: this.config.seed + 12345,
    };
  
    const batch = await requestBatchFromWorker(this.worker, req);
    const { xs, ys } = batchToTensors(batch, this.tasks);
    this.testXs = xs;
    this.testYs = ys;
  }

  async reset(tasks: TaskName[], config: DataConfig) {
    this.tasks = tasks;
    this.config = config;
    this.trainGenFactory = this.generateBatch(this.worker, {
      H: this.config.resolution,
      W: this.config.resolution,
      nShapes: this.config.numShapes,
      typeShapes: this.config.typeShapes,
      batch: this.config.batchSize
    }, this.config.seed, this.tasks);

    this.trainSet = tf.data.generator(this.trainGenFactory).prefetch(1);
    await this.generateTestData();
  }
}