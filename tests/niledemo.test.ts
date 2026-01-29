import { defaultDevice, init } from "@jax-js/jax";
import { describe, it } from 'vitest';
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
const y: number[] = nileInput.y;
const t: number[] = nileInput.t;
const s: number = nileInput.s;
const w: [number, number] = nileInput.w;

const outputDir = path.join(__dirname, 'out');
const outputFileName = path.join(outputDir, 'niledemo-out.json');
const referenceFileName = path.join(__dirname, 'niledemo-out-m.json');

describe('niledemo output', () => {
  it('should match the reference output (niledemo-out-m.json)', async () => {
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    const devices = await init();
    // Use WASM if available, otherwise use CPU
    let useDevice = defaultDevice();
    if (devices.includes('wasm')) {
        useDevice = 'wasm';
    } else if (devices.includes('cpu')) {
        useDevice = 'cpu';
    }
    defaultDevice(useDevice);
    const result = await dlmFit(nileInput.y, nileInput.s, nileInput.w);
    fs.writeFileSync(outputFileName, JSON.stringify(result, null, 2));
    if (!fs.existsSync(referenceFileName)) {
      throw new Error(`Reference file not found: ${referenceFileName}\n\nPlease ensure niledemo-out-m.json exists in the test directory.\nYou can generate it by running the Octave/Matlab script to produce the reference output.`);
    }
    const reference = JSON.parse(fs.readFileSync(referenceFileName, 'utf-8'));
    // Optionally filter by keys from niledemo-keys.json
    const keysFile = path.join(__dirname, 'niledemo-keys.json');
    let filteredResult = result;
    let filteredReference = reference;
    if (fs.existsSync(keysFile)) {
      const keys: string[] = JSON.parse(fs.readFileSync(keysFile, 'utf-8'));
      filteredResult = filterKeys(result, keys);
      filteredReference = filterKeys(reference, keys);
    }
    // Compare the (possibly filtered) result and reference with percentage tolerance
    const relativeTolerance = 1e-10; // Relative tolerance
    const cmp = deepAlmostEqual(filteredResult, filteredReference, relativeTolerance);
    if (!cmp.equal) {
      let reason = '';
      if (cmp.path && cmp.path.includes('key length mismatch')) {
        reason = 'Key length mismatch (object shape difference)';
      } else {
        reason = `Values differ by more than a factor of ${relativeTolerance}`;
      }
      throw new Error(
        `Output does not match reference.\nReason: ${reason}\n` +
        `First mismatch at: ${cmp.path}\n` +
        `Result value:    ${JSON.stringify(cmp.a)}\n` +
        `Reference value: ${JSON.stringify(cmp.b)}`
      );
    }
  });
});
