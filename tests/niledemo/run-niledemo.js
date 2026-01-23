// run-nile.js
// Usage: node run-nile.js
// This script runs the demo(jax) function from niledemo.js and writes the result to niledemo-js.json

(async () => {
  const fs = require('fs');
  const jax = await import('@jax-js/jax');
  const { demo } = require('./niledemo.js');
  const result = await demo(jax);
  const outputFileName = './niledemo-out-js.json';
  fs.writeFileSync(outputFileName, JSON.stringify(result, null, 2));
  console.log(`Output written to ${outputFileName}`);
})();
