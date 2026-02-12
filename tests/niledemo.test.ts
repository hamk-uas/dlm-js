import { defaultDevice, init, DType, Device, checkLeaks } from "@jax-js/jax";
import { describe, it, expect } from 'vitest';
import { dlmFit } from '../src/index';
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
  const runTest = async () => {
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
    const result = await dlmFit(nileInput.y, nileInput.s, nileInput.w, useDType);
    const endTime = performance.now();
    console.log(`Time: ${(endTime - startTime).toFixed(0)}ms`);
    const outputFileName = path.join(outputDir, `niledemo-out.json`);
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
    const relativeTolerance = 1e-6;
    const cmp = deepAlmostEqual(filteredResult, filteredReference, relativeTolerance);
    if (!cmp.equal) {
      throw new Error(
        `Output does not match reference.\n` +
        `First mismatch at: ${cmp.path}\n` +
        `Result value:    ${JSON.stringify(cmp.a)}\n` +
        `Reference value: ${JSON.stringify(cmp.b)}`
      );
    }
  };

  it(`should match reference`, async () => {
    checkLeaks.start();
    await runTest();
    const report = checkLeaks.stop();
    expect(report.leaked, report.summary).toBe(0);
  });

  it(`should match reference`, async () => {
    checkLeaks.start();
    await runTest();
    const report = checkLeaks.stop();
    expect(report.leaked, report.summary).toBe(0);
  });
});
