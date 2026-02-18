import { describe, it } from 'vitest';
import { dlmFit } from '../src/index';
import { filterKeys, deepAlmostEqual, withLeakCheck } from './utils';
import { getTestConfigs, applyConfig, type TestConfig } from './test-matrix';
import * as fs from 'fs';
import * as path from 'path';

const inputFile = path.join(__dirname, 'niledemo-in.json');
if (!fs.existsSync(inputFile)) {
  throw new Error(`Input file not found: ${inputFile}`);
}
const nileInput = JSON.parse(fs.readFileSync(inputFile, 'utf-8'));

const referenceFileName = path.join(__dirname, 'niledemo-out-m.json');
if (!fs.existsSync(referenceFileName)) {
  throw new Error(`Reference file not found: ${referenceFileName}`);
}
const reference = JSON.parse(fs.readFileSync(referenceFileName, 'utf-8'));

const keysFile = path.join(__dirname, 'niledemo-keys.json');
const compareKeys: string[] | null = fs.existsSync(keysFile)
  ? JSON.parse(fs.readFileSync(keysFile, 'utf-8'))
  : null;

const runTest = async (config: TestConfig) => {
  applyConfig(config);

  const outputDir = path.join(__dirname, 'out');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const result = await withLeakCheck(() =>
    dlmFit(nileInput.y, nileInput.s, nileInput.w, config.dtype)
  );

  const outputFileName = path.join(outputDir, `niledemo-out-${config.label.replace('/', '-')}.json`);
  fs.writeFileSync(outputFileName, JSON.stringify(result, (_key, value) =>
    ArrayBuffer.isView(value) ? Array.from(value as Float64Array) : value
  , 2));

  let filteredResult: Record<string, unknown> = result as unknown as Record<string, unknown>;
  let filteredReference: Record<string, unknown> = reference;
  if (compareKeys) {
    filteredResult = filterKeys(result, compareKeys) as Record<string, unknown>;
    filteredReference = filterKeys(reference, compareKeys) as Record<string, unknown>;
  }

  const cmp = deepAlmostEqual(
    filteredResult,
    filteredReference,
    config.relativeTolerance,
    '',
    config.absoluteTolerance,
  );
  if (!cmp.equal) {
    throw new Error(
      `[${config.label}] Output does not match reference.\n` +
      `First mismatch at: ${cmp.path}\n` +
      `Result value:    ${JSON.stringify(cmp.a)}\n` +
      `Reference value: ${JSON.stringify(cmp.b)}`
    );
  }
};

describe('niledemo output', async () => {
  const configs = await getTestConfigs();

  for (const config of configs) {
    it(`should match reference (${config.label})`, async () => {
      await runTest(config);
    });

    // Second run exercises warm-cache / JIT path
    it(`should match reference — 2nd run (${config.label})`, async () => {
      await runTest(config);
    });
  }
});
