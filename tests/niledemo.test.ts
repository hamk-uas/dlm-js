import { defaultDevice, init, DType, Device } from "@jax-js/jax";
import { describe, it } from 'vitest';
import { dlmFit, type DlmMode } from '../src/index';
import { filterKeys, deepAlmostEqual } from './utils';
import * as fs from 'fs';
import * as path from 'path';

// Read input from tests/niledemo-in.json
const inputFile = path.join(__dirname, 'niledemo-in.json');
if (!fs.existsSync(inputFile)) {
  throw new Error(`Input file not found: ${inputFile}`);
}
const nileInput = JSON.parse(fs.readFileSync(inputFile, 'utf-8'));

const referenceFileName = path.join(__dirname, 'niledemo-out-m.json');

describe('niledemo output', () => {
  const runTest = async (mode: DlmMode, label: string) => {
    const outputDir = path.join(__dirname, 'out');
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    const devices = await init();
    let useDevice: Device = 'cpu';
    let useDType = DType.Float64;
    if (devices.includes('webgpu')) {
        useDevice = 'webgpu';
        useDType = DType.Float32;
    } else if (devices.includes('wasm')) {
        useDevice = 'wasm';
    }
    console.log(`Using device: ${useDevice}, dtype: ${useDType}`);
    defaultDevice(useDevice);
    const startTime = performance.now();
    const result = await dlmFit(nileInput.y, nileInput.s, nileInput.w, useDType, mode);
    const endTime = performance.now();
    console.log(`[${label}] Time: ${(endTime - startTime).toFixed(0)}ms`);
    const outputFileName = path.join(outputDir, `niledemo-out-${mode}.json`);
    fs.writeFileSync(outputFileName, JSON.stringify(result, null, 2));
    if (!fs.existsSync(referenceFileName)) {
      throw new Error(`Reference file not found: ${referenceFileName}`);
    }
    const reference = JSON.parse(fs.readFileSync(referenceFileName, 'utf-8'));
    const keysFile = path.join(__dirname, 'niledemo-keys.json');
    let filteredResult = result;
    let filteredReference = reference;
    if (fs.existsSync(keysFile)) {
      const keys: string[] = JSON.parse(fs.readFileSync(keysFile, 'utf-8'));
      filteredResult = filterKeys(result, keys);
      filteredReference = filterKeys(reference, keys);
    }
    const relativeTolerance = 1e-10;
    const cmp = deepAlmostEqual(filteredResult, filteredReference, relativeTolerance);
    if (!cmp.equal) {
      throw new Error(
        `[${label}] Output does not match reference.\n` +
        `First mismatch at: ${cmp.path}\n` +
        `Result value:    ${JSON.stringify(cmp.a)}\n` +
        `Reference value: ${JSON.stringify(cmp.b)}`
      );
    }
  };

  it(`should match reference (using scan)`, async () => {
    await runTest('scan', 'scan');
  });

  it(`should match reference (using jit(scan))`, async () => {
    await runTest('jit', 'jit(scan)');
  });
});
