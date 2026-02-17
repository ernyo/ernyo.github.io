// worker.ts
import { generateSceneMulti } from './shapes2d';
import { Scene, Shape } from './constants';

export type Req = { H:number; W:number; nShapes:number; typeShapes: Shape[]; batch:number; seed:number };

/*------------------ Main Thread ---------------------*/
export function requestBatchFromWorker(
  worker: Worker,
  req: Req
): Promise<Scene> {
  return new Promise(resolve => {
    worker.onmessage = (ev: MessageEvent<Scene>) => {
      resolve(ev.data);
    };
    worker.postMessage(req);
  });
} 

/*------------------ Worker Thread ---------------------*/
function isReq(data: any): data is Req {
  return (
    data &&
    typeof data.H === "number" &&
    typeof data.W === "number" &&
    typeof data.nShapes === "number" &&
    typeof data.batch === "number" &&
    typeof data.seed === "number"
  );
}

declare const self: DedicatedWorkerGlobalScope;

self.onmessage = (ev: MessageEvent<Req>) => {
  if (!isReq(ev.data)) {
    return;
  }
  const { H, W, nShapes, typeShapes, batch, seed } = ev.data;
  const scenes: Scene[] = new Array(batch);
  for (let i = 0; i < batch; i++) {
    scenes[i] = generateSceneMulti({ H, W, nShapes, typeShapes, seed: seed + i });
  }

  // Flatten batch into big typed arrays for zero-copy transfer
  const N = batch, C = 3, K = scenes[0].K;
  const HxW = H * W;

  const rgb = new Uint8ClampedArray(N * HxW * C);
  const seg = new Uint8ClampedArray(N * HxW);
  const edge = new Uint8ClampedArray(N * HxW);
  const sal  = new Uint8ClampedArray(N * HxW);
  const depth= new Uint8ClampedArray(N * HxW);
  const normal = new Uint8ClampedArray(N * HxW * C);

  for (let n = 0; n < N; n++) {
    const s = scenes[n];
    rgb.set(s.rgb, n * HxW * C);
    seg.set(s.seg, n * HxW);
    edge.set(s.edge, n * HxW);
    sal.set(s.sal, n * HxW);
    depth.set(s.depth, n * HxW);
    normal.set(s.normal, n * HxW * C);
  }

  (self as any).postMessage(
    { H, W, K, N, rgb, seg, edge, sal, depth, normal },
    [rgb.buffer, seg.buffer, edge.buffer, sal.buffer, depth.buffer, normal.buffer]
  );
};
