// Helper: filter object by allowed keys (shallow)
function filterKeys(obj: any, keys: string[]): any {
  if (!obj || typeof obj !== 'object') return obj;
  const filtered: any = {};
  for (const k of keys) {
    if (Object.prototype.hasOwnProperty.call(obj, k)) {
      filtered[k] = obj[k];
    }
  }
  return filtered;
}
// Helper: deep comparison with percentage tolerance and path reporting
function deepAlmostEqual(a: any, b: any, percentTolerance = 0.001, path: string = ''): { equal: boolean, path?: string, a?: any, b?: any } {
  if (typeof a === 'number' && typeof b === 'number') {
    if (isNaN(a) && isNaN(b)) return { equal: true };
    if (!isFinite(a) || !isFinite(b)) return { equal: a === b, path, a, b };
    const diff = Math.abs(a - b);
    const maxAbs = Math.max(Math.abs(a), Math.abs(b), 1e-12);
    if (diff / maxAbs > percentTolerance) {
      return { equal: false, path, a, b };
    }
    return { equal: true };
  }
  if (Array.isArray(a) && Array.isArray(b) && a.length === b.length) {
    for (let i = 0; i < a.length; i++) {
      const res = deepAlmostEqual(a[i], b[i], percentTolerance, `${path}[${i}]`);
      if (!res.equal) return res;
    }
    return { equal: true };
  }
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    const aKeys = Object.keys(a);
    const bKeys = Object.keys(b);
    if (aKeys.length !== bKeys.length) {
      return { equal: false, path: path + ' (key length mismatch)', a: aKeys, b: bKeys };
    }
    for (const k of aKeys) {
      const res = deepAlmostEqual(a[k], b[k], percentTolerance, path ? `${path}.${k}` : k);
      if (!res.equal) return res;
    }
    return { equal: true };
  }
  if (a !== b) {
    return { equal: false, path, a, b };
  }
  return { equal: true };
}


import { describe, it, expect } from 'vitest';
import { dlmFit } from '../src/index';
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
const referenceFileName = path.join(outputDir, 'niledemo-out-m.json');

describe('niledemo output', () => {
  it('should match the reference output (niledemo-out-m.json)', async () => {
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    const jax = await import('@jax-js/jax');
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
    const tolerance = 0.01; // 1% relative tolerance
    const cmp = deepAlmostEqual(filteredResult, filteredReference, tolerance);
    if (!cmp.equal) {
      let reason = '';
      if (cmp.path && cmp.path.includes('key length mismatch')) {
        reason = 'Key length mismatch (object shape difference)';
      } else {
        reason = `Values differ by more than ${tolerance * 100}%`;
      }
      throw new Error(
        `Output does not match reference.\nReason: ${reason}\n` +
        `First mismatch at: ${cmp.path}\n` +
        `Result value: ${JSON.stringify(cmp.a)}\n` +
        `Reference value: ${JSON.stringify(cmp.b)}`
      );
    }
  });
});
