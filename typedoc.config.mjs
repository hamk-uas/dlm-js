import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const texmath = require('markdown-it-texmath');
const katex = require('katex');

/** @type {Partial<import("typedoc").TypeDocOptions>} */
export default {
  entryPoints: ['src/index.ts'],
  skipErrorChecking: true,
  out: 'docs',
  name: 'dlm-js',
  readme: 'README.md',
  disableGit: true,
  gitRevision: 'main',
  sourceLinkTemplate:
    'https://github.com/hamk-uas/dlm-js/blob/{gitRevision}/{path}#L{line}',
  searchInDocuments: true,
  searchInComments: true,
  navigation: {
    includeFolders: false,
  },
  plugin: ['typedoc-theme-fresh'],
  theme: 'fresh',
  customCss: 'scripts/katex-import.css',
  markdownItLoader(parser) {
    parser.use(texmath, { engine: katex, delimiters: 'dollars' });
  },
};
